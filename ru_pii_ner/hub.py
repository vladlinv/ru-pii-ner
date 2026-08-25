from __future__ import annotations

import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

BUNDLE_FILES = (
    "ner_config.json",
    "adapter_config.json",
    "adapter_model.safetensors",
    "ner_head_crf.safetensors",
)
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def bundle(source: str | Path, revision: str | None = None) -> Path:
    """Local directory for an adapter bundle, downloading it if needed."""
    path = Path(source)
    if path.is_dir():
        return path

    local = None
    for name in BUNDLE_FILES:
        local = hf_hub_download(str(source), name, revision=revision)
    return Path(local).parent


def base_weights(model_id: str, revision: str | None = None) -> Path:
    """Local weight file of the base model."""
    path = Path(model_id)
    if path.is_dir():
        for name in WEIGHT_FILES:
            if (path / name).is_file():
                return path / name
        raise FileNotFoundError(f"no weight file in {path}")

    for name in WEIGHT_FILES:
        try:
            return Path(hf_hub_download(model_id, name, revision=revision))
        except EntryNotFoundError:
            continue
    raise FileNotFoundError(f"no weight file in {model_id}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
