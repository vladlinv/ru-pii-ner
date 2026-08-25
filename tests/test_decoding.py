from __future__ import annotations

import itertools
import math

import pytest
import torch

from ru_pii_ner.decoding import Crf, decode_spans, spans_from_tags

STATES = 3


def build(seed: int, length: int = 5, types: int = 2, **options) -> tuple[torch.Tensor, Crf]:
    generator = torch.Generator().manual_seed(seed)

    def sample(*shape):
        return torch.randn(*shape, generator=generator)

    emissions = sample(1, length, types, STATES)
    crf = Crf(
        transitions=sample(types, STATES, STATES),
        start_transitions=sample(types, STATES),
        end_transitions=sample(types, STATES),
        bio_constraints=options.pop("bio_constraints", False),
        freeze_o_row=options.pop("freeze_o_row", False),
        **options,
    )
    return emissions, crf


def path_score(crf: Crf, emissions: torch.Tensor, index: int, tags: tuple[int, ...]) -> float:
    score = crf.start_transitions[index, tags[0]] + emissions[0, index, tags[0]]
    for position in range(1, len(tags)):
        score += crf.transitions[index, tags[position - 1], tags[position]]
        score += emissions[position, index, tags[position]]
    return float(score + crf.end_transitions[index, tags[-1]])


def all_paths(length: int) -> list[tuple[int, ...]]:
    return list(itertools.product(range(STATES), repeat=length))


@pytest.mark.parametrize("seed", range(5))
def test_decode_matches_brute_force(seed: int) -> None:
    emissions, crf = build(seed)
    length, types = emissions.shape[1], emissions.shape[2]
    tags = crf.decode(emissions, torch.ones(1, length, dtype=torch.bool))

    for index in range(types):
        scores = [path_score(crf, emissions[0], index, path) for path in all_paths(length)]
        chosen = tuple(tags[0, :, index].tolist())
        assert math.isclose(
            path_score(crf, emissions[0], index, chosen), max(scores), rel_tol=1e-5
        )


@pytest.mark.parametrize("seed", range(5))
def test_span_score_matches_brute_force(seed: int) -> None:
    emissions, crf = build(seed)
    length, types = emissions.shape[1], emissions.shape[2]
    mask = torch.ones(1, length, dtype=torch.bool)

    spans = [
        (0, index, first, last)
        for index in range(types)
        for first in range(length)
        for last in range(first, length)
    ]
    got = crf.score(emissions, mask, spans)

    for span, value in zip(spans, got.tolist()):
        _, index, first, last = span
        scores = torch.tensor(
            [path_score(crf, emissions[0], index, path) for path in all_paths(length)]
        )
        keep = torch.tensor(
            [
                path[first] == 1
                and all(state == 2 for state in path[first + 1 : last + 1])
                and (last + 1 == length or path[last + 1] != 2)
                for path in all_paths(length)
            ]
        )
        expected = (scores[keep].logsumexp(0) - scores.logsumexp(0)).exp()
        assert math.isclose(value, float(expected), rel_tol=1e-4, abs_tol=1e-6)


def test_mask_compacts_the_chain() -> None:
    emissions, crf = build(seed=7, length=7)
    mask = torch.tensor([[False, True, True, False, True, True, False]])
    kept = [1, 2, 4, 5]

    tags = crf.decode(emissions, mask)
    reference = crf.decode(emissions[:, kept], torch.ones(1, len(kept), dtype=torch.bool))

    assert torch.equal(tags[:, kept], reference)
    assert not tags[:, [0, 3, 6]].any()


def test_span_score_ignores_masked_tokens() -> None:
    emissions, crf = build(seed=11, length=7)
    mask = torch.tensor([[False, True, True, False, True, True, False]])
    kept = [1, 2, 4, 5]

    got = crf.score(emissions, mask, [(0, 0, 2, 4), (0, 1, 1, 5)])
    reference = crf.score(
        emissions[:, kept],
        torch.ones(1, len(kept), dtype=torch.bool),
        [(0, 0, 1, 2), (0, 1, 0, 3)],
    )
    assert torch.allclose(got, reference, atol=1e-6)


def test_bio_constraints_hold() -> None:
    emissions, crf = build(seed=3, length=9, bio_constraints=True, freeze_o_row=True)
    mask = torch.ones(1, emissions.shape[1], dtype=torch.bool)
    tags = crf.decode(emissions, mask)

    for index in range(emissions.shape[2]):
        sequence = tags[0, :, index].tolist()
        assert sequence[0] != 2
        assert all(
            not (previous == 0 and current == 2)
            for previous, current in zip(sequence, sequence[1:])
        )


def test_shift_raises_on_wrong_length() -> None:
    emissions, crf = build(seed=1)
    crf = Crf(
        crf.transitions, crf.start_transitions, crf.end_transitions, o_to_b_shift=[0.0] * 5
    )
    with pytest.raises(ValueError):
        crf.decode(emissions, torch.ones(1, emissions.shape[1], dtype=torch.bool))


def test_empty_inputs() -> None:
    emissions, crf = build(seed=1)
    mask = torch.zeros(1, emissions.shape[1], dtype=torch.bool)

    assert not crf.decode(emissions, mask).any()
    assert crf.score(emissions, mask, []).numel() == 0


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        ([0, 0, 0], []),
        ([1, 2, 2], [(0, 2)]),
        ([1, 1, 1], [(0, 0), (1, 1), (2, 2)]),
        ([0, 1, 2, 0, 1], [(1, 2), (4, 4)]),
        ([2, 2, 0], [(0, 1)]),
        ([0, 0, 1], [(2, 2)]),
    ],
)
def test_decode_spans(sequence: list[int], expected: list[tuple[int, int]]) -> None:
    assert decode_spans(sequence) == expected


def test_masked_token_does_not_split_a_span() -> None:
    # Токен из одного пробела получает нулевую ширину смещения и выпадает из
    # маски — сущность вокруг него обязана остаться целой.
    tags = torch.tensor([[[1], [2], [0], [2], [2]]])
    mask = torch.tensor([[True, True, False, True, True]])
    assert spans_from_tags(tags, mask) == [(0, 0, 0, 4)]


def test_spans_from_tags_skips_empty_rows() -> None:
    tags = torch.tensor([[[1], [2]], [[0], [0]]])
    mask = torch.tensor([[True, True], [False, False]])
    assert spans_from_tags(tags, mask) == [(0, 0, 0, 1)]
