"""Pydantic models for the valuation engine (Task 13)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from wbj.core.nullstates import Value


class DcfResult(BaseModel):
    """Output of a FCFF DCF: enterprise value plus its decomposition.

    `ev` is `NOT_MEANINGFUL` when the model refuses to value (e.g. terminal
    growth >= WACC). `warnings` carries non-fatal notes such as an excessive
    terminal-value share.
    """

    ev: Value
    pv_explicit: float | None = None
    pv_terminal: float | None = None
    terminal_share: float | None = None
    warnings: list[str] = Field(default_factory=list)


class ScenarioValue(BaseModel):
    """One scenario's probability and resulting per-share (or equity) value."""

    name: str
    probability: float
    value: float


class ScenarioResult(BaseModel):
    """A set of probability-weighted scenarios and their blended value."""

    scenarios: list[ScenarioValue]
    weighted: float


class MonteCarloResult(BaseModel):
    """Percentile summary of a Monte Carlo valuation run (reproducible by seed)."""

    p10: float
    p25: float
    median: float
    p75: float
    p90: float
    seed: int
    trials: int


class EnsembleResult(BaseModel):
    """Reliability-weighted blend of several model values, with dispersion."""

    value: float
    dispersion: float
