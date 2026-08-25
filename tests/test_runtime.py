from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from ru_pii_ner import hub, load
from ru_pii_ner.runtime import RuPiiNer, _entities

BUNDLE = Path(__file__).resolve().parent.parent / "adapter"
E2E = pytest.mark.skipif(
    os.environ.get("RU_PII_NER_E2E") != "1" or not BUNDLE.is_dir(),
    reason="set RU_PII_NER_E2E=1 to run against the real model",
)


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    config = {
        "format_version": "1.0",
        "base": {"model_id": "org/base", "revision": "abc", "hidden_size": 1024},
        "entities": ["PERSON", "PHONE"],
        "bio_tags": ["O", "B", "I"],
        "head_layout": "type_major",
        "decoder": {
            "bio_constraints": True,
            "freeze_o_row": True,
            "neg_inf": -1e4,
            "trans_scale": 1.0,
            "o_to_b_shift": 0.0,
        },
    }
    (tmp_path / "ner_config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "org/base", "revision": "abc"}),
        encoding="utf-8",
    )
    return tmp_path


def edit(bundle: Path, name: str, **changes: object) -> None:
    path = bundle / name
    config = json.loads(path.read_text(encoding="utf-8"))
    config.update(changes)
    path.write_text(json.dumps(config), encoding="utf-8")


def test_bundle_accepts_a_local_directory(tmp_path: Path) -> None:
    assert hub.bundle(tmp_path) == tmp_path


def test_bundle_downloads_a_repo_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asked = []

    def fake(repo_id: str, filename: str, revision: str | None = None) -> str:
        asked.append((repo_id, filename, revision))
        (tmp_path / filename).write_text("", encoding="utf-8")
        return str(tmp_path / filename)

    monkeypatch.setattr(hub, "hf_hub_download", fake)
    assert hub.bundle("org/model", revision="v1") == tmp_path
    assert [name for _, name, _ in asked] == list(hub.BUNDLE_FILES)
    assert {revision for *_, revision in asked} == {"v1"}


def test_sha256(tmp_path: Path) -> None:
    path = tmp_path / "weights.bin"
    path.write_bytes(b"")
    expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert hub.sha256(path) == expected


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"format_version": "2.0"}, "unsupported"),
        ({"bio_tags": ["O", "B"]}, "bio_tags"),
        ({"head_layout": "tag_major"}, "head_layout"),
        ({"entities": ["PERSON", "PERSON"]}, "unique"),
        ({"base": {"model_id": "org/base", "revision": "abc"}}, "hidden_size"),
    ],
)
def test_config_is_validated(bundle: Path, changes: dict, message: str) -> None:
    edit(bundle, "ner_config.json", **changes)
    with pytest.raises(ValueError, match=message):
        RuPiiNer(bundle, verify=False)


def test_adapter_must_match_the_config(bundle: Path) -> None:
    edit(bundle, "adapter_config.json", revision="other")
    with pytest.raises(ValueError, match="revision"):
        RuPiiNer(bundle, verify=False)

    edit(bundle, "adapter_config.json", base_model_name_or_path="org/other", revision="abc")
    with pytest.raises(ValueError, match="base model"):
        RuPiiNer(bundle, verify=False)


def test_half_precision_needs_cuda(bundle: Path) -> None:
    with pytest.raises(ValueError, match="cuda"):
        RuPiiNer(bundle, device="cpu", dtype=torch.float16)


def test_missing_bundle_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read"):
        RuPiiNer(tmp_path, verify=False)


def test_entities_filtered_by_score() -> None:
    spans = [("PERSON", 5, 9, 0.4), ("PHONE", 0, 3, 0.95)]
    assert [item["entity_group"] for item in _entities("Канье Уэст", spans, 0.5)] == ["PHONE"]
    assert len(_entities("Канье Уэст", spans, 0.0)) == 2


@E2E
def test_offsets_and_scores_are_consistent() -> None:
    model = load(BUNDLE)
    texts = [
        "Иванов Иван Иванович, телефон +7 (999) 123-45-67, почта ivan@test.ru.",
        "Договор № 12/45 от 3 марта 2021 года, г. Нижний Новгород.",
        "",
        "   ",
    ]
    for text, found in zip(texts, model.predict(texts)):
        for item in found:
            assert text[item["start"] : item["end"]] == item["word"]
            assert item["word"] == item["word"].strip()
            assert 0.0 <= item["score"] <= 1.0
            assert item["entity_group"] in model.entities


@E2E
def test_long_text_warns() -> None:
    model = load(BUNDLE)
    text = "Обычный текст. " * 400
    assert len(model.tokenizer(text)["input_ids"]) > model.max_tokens
    with pytest.warns(UserWarning, match="longer than"):
        model.predict(text)


@E2E
def test_batching_does_not_change_results() -> None:
    model = load(BUNDLE)
    texts = [
        "Петров П.П., +7 900 000-11-22",
        "Короткий текст.",
        "Адрес: 620137, Свердловская область, г. Екатеринбург, ул. Блюхера, д. 67.",
    ]
    reference = model.predict(texts, batch_size=1)
    assert model.predict(texts, batch_size=8) == reference
    assert model.predict(texts[0]) == reference[0]


@E2E
def test_min_score_only_removes_entities() -> None:
    model = load(BUNDLE)
    text = "Иванов Иван Иванович, телефон +7 (999) 123-45-67, почта ivan@test.ru."
    everything = model.predict(text)
    filtered = model.predict(text, min_score=0.5)
    assert filtered == [item for item in everything if item["score"] >= 0.5]
