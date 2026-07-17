"""Final report schema (Task 21), per Cerebro/00_main_agent/FINAL_REPORT_SCHEMA.md."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProfileResult(BaseModel):
    """The gated research classification for the security."""

    profile: str
    raw_score: float
    band: str
    reasons: list[str] = Field(default_factory=list)
    gate_evaluations: dict = Field(default_factory=dict)


class Override(BaseModel):
    """A mandatory override triggered during aggregation."""

    override_id: str
    condition: str
    action: str
    note: str = ""


class CategoryScorecardRow(BaseModel):
    category: str
    max_points: float
    awarded_points: float
    score_10: float
    confidence: float
    coverage: float


class FinalReport(BaseModel):
    """The auditable final report handed to the renderer."""

    report_version: str = "2.0.0"
    security: dict = Field(default_factory=dict)
    knowledge_timestamp: str | None = None
    profile: ProfileResult
    category_scorecard: list[CategoryScorecardRow] = Field(default_factory=list)
    executive_thesis: list[str] = Field(default_factory=list)  # 7 sentences
    important_levels: list[dict] = Field(default_factory=list)
    valuation_scenarios: list[dict] = Field(default_factory=list)
    reverse_dcf: dict = Field(default_factory=dict)
    thesis_killers: list[str] = Field(default_factory=list)
    monitoring_triggers: list[str] = Field(default_factory=list)
    overrides: list[Override] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    insider_activity: list[dict] = Field(default_factory=list)
    institutional_holders: list[dict] = Field(default_factory=list)
    missing_or_conflicted_data: list[str] = Field(default_factory=list)
    per_share_suppressed: bool = False
    revisit_date_or_event: str | None = None
    audit: dict = Field(default_factory=dict)
