"""Judgment overlay merge (Task 20).

Collects the `JudgmentRequest`s the specialists emitted, and merges back
agent/human `Judgment`s: it validates each judgment, replaces the affected
`NOT_SCORABLE` metric, re-scores the target dimension, recomputes coverage
and the output hash. A judgment for an unknown request is an error; a
judgment missing an evidence class or source is rejected.
"""

from __future__ import annotations

import copy

from wbj.schemas.overlay import Judgment
from wbj.specialists.common import (
    JudgmentRequest,
    SpecialistOutput,
    category_from_dimensions,
    compute_output_hash,
)


def collect_requests(outputs: list[SpecialistOutput]) -> list[JudgmentRequest]:
    """Flatten every specialist's open judgment requests."""
    requests: list[JudgmentRequest] = []
    for out in outputs:
        requests.extend(out.judgment_requests)
    return requests


def _index_requests(outputs: list[SpecialistOutput]) -> dict[str, tuple[int, JudgmentRequest]]:
    index: dict[str, tuple[int, JudgmentRequest]] = {}
    for i, out in enumerate(outputs):
        for req in out.judgment_requests:
            index[req.request_id] = (i, req)
    return index


def merge_overlay(
    outputs: list[SpecialistOutput], judgments: list[Judgment]
) -> list[SpecialistOutput]:
    """Return new outputs with the judgments applied.

    Raises `ValueError` for an unknown `request_id` or a judgment missing its
    evidence class / source."""
    merged = [out.model_copy(deep=True) for out in outputs]
    index = _index_requests(outputs)

    for j in judgments:
        if not j.evidence_class or not j.source:
            raise ValueError(f"judgment {j.request_id!r} missing evidence_class/source")
        if j.request_id not in index:
            raise ValueError(f"unknown judgment request_id: {j.request_id!r}")

        out_idx, req = index[j.request_id]
        out = merged[out_idx]

        target_dim = j.target_dimension or req.target_dimension
        if target_dim is not None and j.score_10 is not None:
            for dim in out.dimensions:
                if dim.name == target_dim:
                    dim.score_10 = j.score_10
                    dim.awarded_points = dim.max_points * j.score_10 / 10.0
                    dim.rationale = f"Set by judgment {j.request_id}: {j.rationale}"[:200]
                    break
            out.category = category_from_dimensions(
                out.dimensions, out.category.max_points, out.category.confidence
            )

        # The request is now answered: drop it and bump coverage.
        answered = [r for r in out.dimensions]  # noqa: F841 (kept for clarity)
        out.judgment_requests = [r for r in out.judgment_requests if r.request_id != j.request_id]
        out.assumptions.append(f"judgment applied: {j.request_id} ({j.evidence_class})")
        # Coverage rises as open judgment requests are resolved.
        scored_dims = sum(1 for d in out.dimensions if d.score_10 > 0)
        out.coverage = min(1.0, scored_dims / max(len(out.dimensions), 1))

    for out in merged:
        out.output_hash = compute_output_hash(out)
    return merged
