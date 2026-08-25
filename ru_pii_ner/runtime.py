from __future__ import annotations

import json
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence, TypedDict, overload

import torch
from peft import PeftModel
from safetensors.torch import load_file
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer
from transformers import logging as transformers_logging

from . import hub
from .decoding import Crf, spans_from_tags

DEFAULT_SOURCE = "vladlinv/ru-pii-ner"
BIO_TAGS = ("O", "B", "I")
MAX_TOKENS = 512
WEIGHT_KEYS = frozenset(
    {
        "head.weight",
        "head.bias",
        "crf.transitions",
        "crf.start_transitions",
        "crf.end_transitions",
    }
)
DECODER_KEYS = ("bio_constraints", "freeze_o_row", "neg_inf", "trans_scale", "o_to_b_shift")

Shift = float | Sequence[float] | Mapping[str, float]


class Entity(TypedDict):
    entity_group: str
    word: str
    start: int
    end: int
    score: float


def load(
    source: str | Path = DEFAULT_SOURCE,
    *,
    device: str | torch.device | None = None,
    base_model: str | None = None,
    dtype: torch.dtype | None = None,
    verify: bool = True,
) -> RuPiiNer:
    return RuPiiNer(
        source, device=device, base_model=base_model, dtype=dtype, verify=verify
    )


class RuPiiNer:
    """Backbone-only PEFT adapter with a per-type BIO CRF head."""

    def __init__(
        self,
        source: str | Path = DEFAULT_SOURCE,
        *,
        device: str | torch.device | None = None,
        base_model: str | None = None,
        dtype: torch.dtype | None = None,
        verify: bool = True,
    ) -> None:
        self.path = hub.bundle(source)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if dtype in (torch.float16, torch.bfloat16) and self.device.type != "cuda":
            raise ValueError(f"{dtype} requires cuda, got {self.device.type}")

        self.config = _read_json(self.path / "ner_config.json")
        self._validate_config(base_model)

        base = self.config["base"]
        model_id = base_model or base["model_id"]
        revision = None if base_model else base["revision"]

        if verify and not base_model and base.get("weights_sha256"):
            digest = hub.sha256(hub.base_weights(model_id, revision))
            if digest != base["weights_sha256"]:
                raise ValueError(f"base weights sha256 is {digest}")

        verbosity = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        try:
            backbone = AutoModel.from_pretrained(model_id, revision=revision, dtype=dtype)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, revision=revision, use_fast=True
            )
        finally:
            transformers_logging.set_verbosity(verbosity)

        if backbone.config.hidden_size != base["hidden_size"]:
            raise ValueError(
                f"base hidden_size is {backbone.config.hidden_size}, "
                f"expected {base['hidden_size']}"
            )
        if not self.tokenizer.is_fast:
            raise RuntimeError("a fast tokenizer is required for character offsets")

        self.max_tokens = min(MAX_TOKENS, backbone.config.max_position_embeddings - 2)
        self.model = (
            PeftModel.from_pretrained(backbone, self.path, is_trainable=False)
            .to(self.device)
            .eval()
        )

        tensors = load_file(str(self.path / "ner_head_crf.safetensors"))
        self._validate_tensors(tensors)

        self.entities: tuple[str, ...] = tuple(self.config["entities"])
        self.head = nn.Linear(base["hidden_size"], len(self.entities) * len(BIO_TAGS))
        with torch.no_grad():
            self.head.weight.copy_(tensors["head.weight"])
            self.head.bias.copy_(tensors["head.bias"])
        self.head = self.head.to(self.device, dtype or torch.float32).eval()
        self.head.requires_grad_(False)

        decoder = {key: self.config["decoder"][key] for key in DECODER_KEYS}
        decoder["o_to_b_shift"] = self._resolve_shift(decoder["o_to_b_shift"])
        self.crf = Crf(
            transitions=tensors["crf.transitions"].to(self.device),
            start_transitions=tensors["crf.start_transitions"].to(self.device),
            end_transitions=tensors["crf.end_transitions"].to(self.device),
            **decoder,
        )

    @overload
    def predict(self, texts: str, **kwargs: object) -> list[Entity]: ...

    @overload
    def predict(self, texts: Iterable[str], **kwargs: object) -> list[list[Entity]]: ...

    @torch.inference_mode()
    def predict(
        self,
        texts: str | Iterable[str],
        *,
        batch_size: int = 16,
        min_score: float = 0.0,
        o_to_b_shift: Shift | None = None,
    ) -> list[Entity] | list[list[Entity]]:
        """Entities with character offsets and CRF probabilities."""
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        crf = self.crf
        if o_to_b_shift is not None:
            crf = replace(crf, o_to_b_shift=self._resolve_shift(o_to_b_shift))

        results: list[list[Entity]] = []
        for begin in range(0, len(items), batch_size):
            batch = items[begin : begin + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = encoded.pop("offset_mapping")
            if (encoded["attention_mask"].sum(dim=1) >= self.max_tokens).any():
                warnings.warn(
                    f"text is longer than {self.max_tokens} tokens, the tail is dropped",
                    stacklevel=2,
                )
            results.extend(
                _entities(text, spans, min_score)
                for text, spans in zip(batch, self._spans(encoded, offsets, crf))
            )

        return results[0] if single else results

    def _spans(
        self,
        encoded: Mapping[str, Tensor],
        offsets: Tensor,
        crf: Crf,
    ) -> list[list[tuple[str, int, int, float]]]:
        inputs = {name: value.to(self.device) for name, value in encoded.items()}
        hidden = self.model(**inputs).last_hidden_state
        logits = self.head(hidden.to(self.head.weight.dtype))
        emissions = logits.view(*logits.shape[:2], len(self.entities), len(BIO_TAGS))

        mask = inputs["attention_mask"].bool() & (offsets[..., 1] > offsets[..., 0]).to(
            self.device
        )
        tags = crf.decode(emissions, mask)
        spans = spans_from_tags(tags, mask)
        scores = crf.score(emissions, mask, spans).tolist()

        found: list[list[tuple[str, int, int, float]]] = [[] for _ in range(tags.shape[0])]
        for (row, index, first, last), score in zip(spans, scores):
            start = int(offsets[row, first, 0])
            end = int(offsets[row, last, 1])
            if end > start:
                found[row].append((self.entities[index], start, end, score))
        return found

    def _validate_config(self, base_model: str | None) -> None:
        config = self.config
        if str(config.get("format_version", "")).split(".")[0] != "1":
            raise ValueError("unsupported ner_config format")
        if config.get("bio_tags") != list(BIO_TAGS):
            raise ValueError(f"bio_tags must be {list(BIO_TAGS)}")
        if config.get("head_layout") != "type_major":
            raise ValueError("head_layout must be type_major")

        entities = config.get("entities", [])
        if not entities or len(entities) != len(set(entities)):
            raise ValueError("entities must be a non-empty unique list")

        base = config.get("base", {})
        for key in ("model_id", "revision", "hidden_size"):
            if not base.get(key):
                raise ValueError(f"missing base.{key}")

        adapter = _read_json(self.path / "adapter_config.json")
        if not base_model:
            if adapter.get("base_model_name_or_path") != base["model_id"]:
                raise ValueError("base model differs between adapter and NER config")
            if adapter.get("revision") != base["revision"]:
                raise ValueError("base revision differs between adapter and NER config")

        decoder = config.get("decoder", {})
        for key in DECODER_KEYS:
            if key not in decoder:
                raise ValueError(f"missing decoder.{key}")

    def _validate_tensors(self, tensors: Mapping[str, Tensor]) -> None:
        if set(tensors) != WEIGHT_KEYS:
            missing = sorted(WEIGHT_KEYS - set(tensors))
            extra = sorted(set(tensors) - WEIGHT_KEYS)
            raise ValueError(f"invalid weight keys; missing={missing}, extra={extra}")

        types = len(self.config["entities"])
        hidden = self.config["base"]["hidden_size"]
        expected = {
            "head.weight": (types * len(BIO_TAGS), hidden),
            "head.bias": (types * len(BIO_TAGS),),
            "crf.transitions": (types, len(BIO_TAGS), len(BIO_TAGS)),
            "crf.start_transitions": (types, len(BIO_TAGS)),
            "crf.end_transitions": (types, len(BIO_TAGS)),
        }
        for name, shape in expected.items():
            if tuple(tensors[name].shape) != shape:
                raise ValueError(
                    f"{name} has shape {tuple(tensors[name].shape)}, expected {shape}"
                )

    def _resolve_shift(self, shift: Shift) -> float | list[float]:
        if isinstance(shift, Mapping):
            unknown = set(shift) - set(self.entities)
            if unknown:
                raise ValueError(f"unknown entity shifts: {sorted(unknown)}")
            return [float(shift.get(entity, 0.0)) for entity in self.entities]
        if isinstance(shift, Sequence):
            if len(shift) != len(self.entities):
                raise ValueError(f"expected {len(self.entities)} shifts")
            return [float(item) for item in shift]
        return float(shift)


def _entities(
    text: str, spans: Sequence[tuple[str, int, int, float]], min_score: float
) -> list[Entity]:
    found: list[Entity] = [
        {
            "entity_group": name,
            "word": text[start:end],
            "start": start,
            "end": end,
            "score": score,
        }
        for name, start, end, score in spans
        if score >= min_score
    ]
    found.sort(key=lambda item: (item["start"], item["end"], item["entity_group"]))
    return found


def _read_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
