"""Shared specialist envelope and scoring helpers (Task 14).

Every specialist emits a `SpecialistOutput` in the shape fixed by
`Cerebro/shared/OUTPUT_CONTRACT.md` and each agent's `OUTPUT_SCHEMA.md`.
Scoring is deterministic and auditable: a metric's BAD/GOOD/EXCELLENT band
maps to fixed points, dimensions average their metrics' band scores, and the
category rolls up as `10 * awarded_points / max_points`. Judgment-only
metrics emit `Value.null(NOT_SCORABLE)` plus a `JudgmentRequest` — never a
guessed number.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from wbj.core.nullstates import Value

# Band -> 0-10 dimension contribution (midpoints of the 0-3 / 4-6 / 7-10
# buckets in each agent's SCORING.md), and Band -> core-diagnostic points.
BAND_SCORE_10 = {"BAD": 1.5, "GOOD": 5.0, "EXCELLENT": 9.0}
CORE_POINTS = {"BAD": 0, "GOOD": 1, "EXCELLENT": 2}


class MetricRow(BaseModel):
    """One audited metric row per the Output Contract."""

    metric_id: str
    value: Value
    unit: str = ""
    period: str | None = None
    formula: str = ""
    score: float | None = None  # 0-10 or None for NOT_SCORABLE
    evidence_class: str | None = None
    source: str | None = None
    confidence: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    band: str | None = None  # BAD / GOOD / EXCELLENT for diagnostic metrics


class Dimension(BaseModel):
    """A scored sub-dimension of a category."""

    name: str
    max_points: float
    score_10: float
    awarded_points: float
    rationale: str = ""


class CategoryScore(BaseModel):
    """The rolled-up category score."""

    max_points: float
    awarded_points: float
    score_10: float
    confidence: float


class JudgmentRequest(BaseModel):
    """A qualitative metric the engine cannot score without a human/agent
    judgment (returned alongside a `NOT_SCORABLE` metric)."""

    request_id: str
    agent_id: str
    metric_id: str
    question: str
    schema_hint: str = ""


class SpecialistOutput(BaseModel):
    """The common envelope shared by all six specialists."""

    agent_id: str
    version: str = "2.0.0"
    status: str = "COMPLETE"
    security: dict = Field(default_factory=dict)
    knowledge_timestamp: str | None = None
    category: CategoryScore
    coverage: float
    dimensions: list[Dimension] = Field(default_factory=list)
    metrics: list[MetricRow] = Field(default_factory=list)
    mandatory_flags: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    judgment_requests: list[JudgmentRequest] = Field(default_factory=list)
    source_lineage: list[str] = Field(default_factory=list)
    validation_tests: dict = Field(
        default_factory=lambda: {"passed": 0, "failed": 0, "warnings": 0}
    )


# --- band classification -----------------------------------------------------


def band_higher_better(v: float, good_lo: float, good_hi: float) -> str:
    """BAD if v < good_lo, GOOD if good_lo <= v <= good_hi, EXCELLENT if v > good_hi.

    Upper edge is inclusive of GOOD (e.g. yoy=0.10 with good_hi=0.10 -> GOOD,
    matching the registry's `EXCELLENT >10%` exclusive convention)."""
    if v < good_lo:
        return "BAD"
    if v <= good_hi:
        return "GOOD"
    return "EXCELLENT"


def band_lower_better(v: float, good_lo: float, good_hi: float) -> str:
    """EXCELLENT if v < good_lo, GOOD if good_lo <= v <= good_hi, BAD if v > good_hi."""
    if v < good_lo:
        return "EXCELLENT"
    if v <= good_hi:
        return "GOOD"
    return "BAD"


# --- aggregation -------------------------------------------------------------


def dimension_from_bands(name: str, max_points: float, bands: list[str], rationale: str = "") -> Dimension:
    """Build a Dimension by averaging its metrics' band scores (0-10). An
    empty band list scores 0 (missing evidence is never neutral)."""
    if bands:
        score_10 = sum(BAND_SCORE_10[b] for b in bands) / len(bands)
    else:
        score_10 = 0.0
    awarded = max_points * score_10 / 10.0
    return Dimension(name=name, max_points=max_points, score_10=score_10, awarded_points=awarded, rationale=rationale)


def category_from_dimensions(dimensions: list[Dimension], max_points: float, confidence: float) -> CategoryScore:
    """Sum dimension points into the category; score_10 = 10·awarded/max."""
    awarded = sum(d.awarded_points for d in dimensions)
    score_10 = 10.0 * awarded / max_points if max_points else 0.0
    return CategoryScore(
        max_points=max_points, awarded_points=awarded, score_10=score_10, confidence=confidence
    )


def core_diagnostic(bands: list[str]) -> dict:
    """Core-N diagnostic: each valid metric BAD=0/GOOD=1/EXCELLENT=2 ->
    percent = points/(2·valid)·100, score10 = percent/10."""
    valid = len(bands)
    points = sum(CORE_POINTS[b] for b in bands)
    maximum = 2 * valid
    percent = (points / maximum * 100.0) if maximum else 0.0
    return {
        "valid_count": valid,
        "points": points,
        "maximum_valid_points": maximum,
        "percent": percent,
        "score_10": percent / 10.0,
    }
