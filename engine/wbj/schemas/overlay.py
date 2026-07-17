"""Judgment overlay schema (Task 20)."""

from __future__ import annotations

from pydantic import BaseModel

from wbj.core.nullstates import EvidenceClass


class Judgment(BaseModel):
    """An agent/human answer to a `JudgmentRequest`.

    `answer` is the qualitative/quantitative response; `score_10` and
    `target_dimension` let the overlay re-score the affected dimension. Every
    judgment must carry an evidence class and a source, or it is rejected.
    """

    request_id: str
    answer: float | str | dict
    evidence_class: EvidenceClass
    source: str
    rationale: str = ""
    score_10: float | None = None
    target_dimension: str | None = None
