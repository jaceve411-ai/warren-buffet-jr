"""Pydantic models for the important-levels engine (Task 12)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ZoneType = Literal["support", "resistance"]
Timeframe = Literal["daily", "weekly"]
ZoneStatus = Literal["candidate", "confirmed", "strong", "broken", "role_reversed"]


class Touch(BaseModel):
    """One independent interaction of price with a zone."""

    date: str
    pivot_price: float
    rejection_atr: float
    volume_ratio: float
    age_sessions: int


class Zone(BaseModel):
    """A clustered support/resistance zone with its evidence and status."""

    zone_id: str
    type: ZoneType
    lower: float
    center: float
    upper: float
    timeframe: Timeframe = "daily"
    status: ZoneStatus = "candidate"
    strength_0_100: float = 0.0
    touches: list[Touch] = Field(default_factory=list)
    distance_percent: float | None = None
    distance_atr: float | None = None
    confluence_count: int = 0
    liquidity_confidence: float = 0.5
    confirmation_rule: str = ""
    invalidation_rule: str = ""


class Gap(BaseModel):
    """An earnings gap and how it held."""

    date: str
    gap_percent: float
    material: bool
    day1_hold: float | None = None
    day5_hold: float | None = None
    day20_hold: float | None = None


class LevelsOutput(BaseModel):
    """Ranked levels handed to the technical specialist and price synthesis."""

    support: list[Zone] = Field(default_factory=list)
    resistance: list[Zone] = Field(default_factory=list)
    moving_averages: dict[str, float] = Field(default_factory=dict)
    avwaps: dict[str, float] = Field(default_factory=dict)
    gaps: list[Gap] = Field(default_factory=list)
