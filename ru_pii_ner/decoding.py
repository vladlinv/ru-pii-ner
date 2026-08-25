from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

Span = tuple[int, int, int, int]

_BLOCKED = -1e9


@dataclass(frozen=True)
class Crf:
    """Independent per-type BIO chains. Transition layout is [from, to]."""

    transitions: Tensor
    start_transitions: Tensor
    end_transitions: Tensor
    bio_constraints: bool = True
    freeze_o_row: bool = True
    neg_inf: float = -1e4
    trans_scale: float = 1.0
    o_to_b_shift: float | Sequence[float] = 0.0

    @torch.inference_mode()
    def decode(self, emissions: Tensor, mask: Tensor) -> Tensor:
        """Best tag per token, shaped [batch, length, types]."""
        packed, alive, indices, transitions, start, end = self._pack(emissions, mask)
        chains, packed_length, states = packed.shape

        score = start + packed[:, 0]
        backpointers = []
        identity = torch.arange(states, device=packed.device).expand(chains, states)

        for position in range(1, packed_length):
            best, previous = (score.unsqueeze(2) + transitions).max(dim=1)
            active = alive[:, position].unsqueeze(1)
            score = torch.where(active, best + packed[:, position], score)
            backpointers.append(torch.where(active, previous, identity))

        current = (score + end).argmax(dim=1)
        tags = torch.zeros(chains, packed_length, dtype=torch.long, device=packed.device)
        for position in range(packed_length - 1, 0, -1):
            tags[:, position] = current
            current = backpointers[position - 1].gather(1, current[:, None]).squeeze(1)
        tags[:, 0] = current

        batch, length, types, _ = emissions.shape
        full = torch.zeros(chains, length, dtype=torch.long, device=packed.device)
        full.scatter_(1, indices, torch.where(alive, tags, torch.zeros_like(tags)))
        return full.view(batch, types, length).permute(0, 2, 1).contiguous()

    @torch.inference_mode()
    def score(self, emissions: Tensor, mask: Tensor, spans: Sequence[Span]) -> Tensor:
        """Probability of each (batch, type, first, last) span existing exactly."""
        if not spans:
            return torch.zeros(0, device=emissions.device)

        packed, alive, indices, transitions, start, end = self._pack(emissions, mask)
        chains, packed_length, states = packed.shape
        types = emissions.shape[2]
        device = packed.device

        total = _forward(packed, alive, transitions, start, end)

        placement = torch.zeros(chains, emissions.shape[1], dtype=torch.long, device=device)
        placement.scatter_(
            1, indices, torch.arange(packed_length, device=device).expand(chains, -1)
        )

        rows = torch.tensor([b * types + t for b, t, _, _ in spans], device=device)
        first = placement[rows, torch.tensor([s[2] for s in spans], device=device)]
        last = placement[rows, torch.tensor([s[3] for s in spans], device=device)]

        position = torch.arange(packed_length, device=device)[None, :]
        state = torch.arange(states, device=device)[None, None, :]
        inside = (position >= first[:, None]) & (position <= last[:, None])
        required = torch.where(position == first[:, None], 1, 2)

        blocked = inside[:, :, None] & (state != required[:, :, None])
        blocked |= (position == last[:, None] + 1)[:, :, None] & (state == 2)

        clamped = packed[rows].masked_fill(blocked & alive[rows][:, :, None], _BLOCKED)
        constrained = _forward(
            clamped, alive[rows], transitions[rows], start[rows], end[rows]
        )
        return (constrained - total[rows]).exp().clamp(0.0, 1.0)

    def _pack(
        self, emissions: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        emissions = emissions.float()
        mask = mask.bool().to(emissions.device)
        batch, length, types, states = emissions.shape
        device = emissions.device

        chains = emissions.permute(0, 2, 1, 3).reshape(batch * types, length, states)
        chain_mask = mask[:, None, :].expand(batch, types, length).reshape(-1, length)
        lengths = chain_mask.sum(dim=1)
        packed_length = max(int(lengths.max()), 1)

        # Живые токены сдвигаются в начало, порядок сохраняется — цепочка CRF
        # не должна видеть служебные токены и паддинг.
        indices = (~chain_mask).long().argsort(dim=1, stable=True)[:, :packed_length]
        packed = chains.gather(1, indices[:, :, None].expand(-1, -1, states))
        alive = torch.arange(packed_length, device=device)[None, :] < lengths[:, None]

        transitions = self.transitions.float().to(device) * self.trans_scale
        start = self.start_transitions.float().to(device) * self.trans_scale
        end = self.end_transitions.float().to(device) * self.trans_scale

        if self.freeze_o_row:
            transitions = transitions.clone()
            transitions[:, 0, :] = 0

        shift = torch.as_tensor(self.o_to_b_shift, dtype=torch.float32, device=device)
        if shift.numel() not in (1, types):
            raise ValueError(f"expected one O->B shift or {types} values")
        if shift.any():
            transitions = transitions.clone()
            transitions[:, 0, 1] += shift.reshape(-1)

        if self.bio_constraints:
            transitions = transitions.clone()
            start = start.clone()
            transitions[:, 0, 2] += self.neg_inf
            start[:, 2] += self.neg_inf

        return (
            packed,
            alive,
            indices,
            _repeat(transitions, batch),
            _repeat(start, batch),
            _repeat(end, batch),
        )


def _forward(
    packed: Tensor, alive: Tensor, transitions: Tensor, start: Tensor, end: Tensor
) -> Tensor:
    score = start + packed[:, 0]
    for position in range(1, packed.shape[1]):
        total = (score.unsqueeze(2) + transitions).logsumexp(dim=1)
        active = alive[:, position].unsqueeze(1)
        score = torch.where(active, total + packed[:, position], score)
    return (score + end).logsumexp(dim=1)


def _repeat(tensor: Tensor, batch: int) -> Tensor:
    return tensor.unsqueeze(0).expand(batch, *tensor.shape).reshape(-1, *tensor.shape[1:])


def decode_spans(sequence: Sequence[int]) -> list[tuple[int, int]]:
    """Token index ranges of BIO spans, inclusive on both ends."""
    spans = []
    start = None
    for index, state in enumerate(sequence):
        if state == 1:
            if start is not None:
                spans.append((start, index - 1))
            start = index
        elif state == 2:
            if start is None:
                start = index
        elif start is not None:
            spans.append((start, index - 1))
            start = None
    if start is not None:
        spans.append((start, len(sequence) - 1))
    return spans


def spans_from_tags(tags: Tensor, mask: Tensor) -> list[Span]:
    """Spans over live tokens only, reported in the original token frame.

    Masked tokens carry no tag, so they must not split an entity — the CRF chain
    never saw them either.
    """
    found: list[Span] = []
    for row in range(tags.shape[0]):
        positions = mask[row].nonzero(as_tuple=True)[0]
        if not positions.numel():
            continue
        for index in range(tags.shape[2]):
            found.extend(
                (row, index, int(positions[first]), int(positions[last]))
                for first, last in decode_spans(tags[row, positions, index].tolist())
            )
    return found
