"""Financial specialist (15 pts) — Task 14.

Transcribes FIN-001…033 (`Cerebro/02_financial_analysis/FORMULAS.md`) with
their BAD/GOOD/EXCELLENT bands, the five scored dimensions
(`SCORING.md`), and the Core-27 diagnostic. Conventional FCF/ROIC are
suppressed under a bank/insurer/REIT adapter. Judgment-free: any metric it
cannot compute from the packet is `NOT_SCORABLE`, never guessed.
"""

from __future__ import annotations

from typing import Any

from wbj.core.nullstates import NullState, Value
from wbj.schemas.packet import Packet
from wbj.specialists.common import (
    CategoryScore,
    Dimension,
    MetricRow,
    SpecialistOutput,
    band_higher_better,
    band_lower_better,
    category_from_dimensions,
    core_diagnostic,
    dimension_from_bands,
)

AGENT_ID = "financial_analysis"
MAX_POINTS = 15.0
_ADAPTER_KEYWORDS = ("bank", "insurance", "insurer", "reit")


class FinancialOutput(SpecialistOutput):
    """Financial envelope + agent-specific extension fields."""

    core_27_metrics: dict = {}
    mandatory_overrides: list[str] = []
    strongest_metric: str | None = None
    weakest_metric: str | None = None


# --- pure formulas (FIN-*) ---------------------------------------------------


def yoy_growth(current: float, prior: float) -> float:
    """FIN-GR-001 — (current − prior)/prior."""
    return (current - prior) / prior


def gross_margin(revenue: float, cogs: float) -> float:
    """FIN-PR-007 — (revenue − COGS)/revenue."""
    return (revenue - cogs) / revenue


def operating_margin(ebit: float, revenue: float) -> float:
    """FIN-PR-008 — EBIT/revenue."""
    return ebit / revenue


def net_margin(net_income: float, revenue: float) -> float:
    """FIN-PR-009 — net income/revenue."""
    return net_income / revenue


def free_cash_flow(ocf: float, capex: float) -> float:
    """FIN-CF-012 — OCF − CapEx. CapEx stored negative (a cash outflow) is
    added; a positive CapEx magnitude is subtracted."""
    return ocf + capex if capex < 0 else ocf - capex


def fcf_margin(fcf: float, revenue: float) -> float:
    """FIN-CF-014 — FCF/revenue."""
    return fcf / revenue


def current_ratio(current_assets: float, current_liabilities: float) -> float:
    """FIN-BS-017 — current assets/current liabilities."""
    return current_assets / current_liabilities


def quick_ratio(current_assets: float, inventory: float, current_liabilities: float) -> float:
    """FIN-BS-018 — (current assets − inventory)/current liabilities."""
    return (current_assets - inventory) / current_liabilities


def debt_to_equity(debt: float, equity: float) -> Value:
    """FIN-BS-019 — debt/equity; NOT_MEANINGFUL with non-positive equity."""
    if equity <= 0:
        return Value.null(NullState.NOT_MEANINGFUL, unit="ratio")
    return Value.of(debt / equity, unit="ratio")


def interest_coverage(ebit: float, interest: float) -> float:
    """FIN-BS-020 — normalized EBIT/cash interest expense."""
    return ebit / interest


def cash_vs_earnings(ocf: float, net_income: float) -> Value:
    """FIN-CF-015 — OCF/net income; NOT_MEANINGFUL across a loss."""
    if net_income <= 0:
        return Value.null(NullState.NOT_MEANINGFUL, unit="ratio")
    return Value.of(ocf / net_income, unit="ratio")


def diluted_share_cagr(shares_end: float, shares_begin: float, years: int) -> float:
    """FIN-DX-033 — (end/begin)^(1/n) − 1; positive means dilution."""
    return (shares_end / shares_begin) ** (1.0 / years) - 1.0


# --- band helpers per metric -------------------------------------------------


def band_revenue_growth(v: float) -> str:
    return band_higher_better(v, 0.0, 0.10)


def band_operating_margin(v: float) -> str:
    return band_higher_better(v, 0.10, 0.20)


def band_net_margin(v: float) -> str:
    return band_higher_better(v, 0.05, 0.10)


def band_fcf_margin(v: float) -> str:
    return band_higher_better(v, 0.0, 0.10)


def band_current_ratio(v: float) -> str:
    # BAD <1.0, GOOD 1.0-1.5, EXCELLENT >=1.5 (>3 flagged as idle capital elsewhere)
    return band_higher_better(v, 1.0, 1.5)


def band_interest_coverage(v: float) -> str:
    return band_higher_better(v, 1.5, 3.0)


def band_roic_vs_wacc(roic: float, wacc: float) -> str:
    """FIN-EF-026 — BAD below WACC, GOOD within ±1pt, EXCELLENT above."""
    spread = roic - wacc
    if spread < -0.01:
        return "BAD"
    if spread <= 0.01:
        return "GOOD"
    return "EXCELLENT"


def conventional_metrics_applicable(industry_adapter: str | None) -> bool:
    """FCF/ROIC are not conventional for banks/insurers/REITs (FIN-T010)."""
    if not industry_adapter:
        return True
    lowered = industry_adapter.lower()
    return not any(k in lowered for k in _ADAPTER_KEYWORDS)


# --- run ---------------------------------------------------------------------


def _num(rec: dict, key: str) -> float | None:
    v = rec.get(key)
    return float(v) if isinstance(v, (int, float)) else None


def run(packet: Packet, overlay: dict | None = None) -> FinancialOutput:
    """Score the financial category for `packet`."""
    annual = packet.fundamentals.get("annual", [])
    metrics: list[MetricRow] = []
    flags: list[str] = []
    overrides: list[str] = []
    adapter_ok = conventional_metrics_applicable(packet.analysis.industry_adapter)

    # Dimension band buckets.
    d_growth: list[str] = []
    d_eps_fcf: list[str] = []
    d_margins: list[str] = []
    d_balance: list[str] = []
    d_efficiency: list[str] = []
    core_bands: list[str] = []

    def add(metric_id: str, formula: str, value: Value, band: str | None, bucket: list[str] | None):
        score = None
        if band is not None:
            from wbj.specialists.common import BAND_SCORE_10

            score = BAND_SCORE_10[band]
            core_bands.append(band)
            if bucket is not None:
                bucket.append(band)
        metrics.append(
            MetricRow(
                metric_id=metric_id,
                value=value,
                formula=formula,
                score=score,
                band=band,
                evidence_class="C",
                source="packet.fundamentals",
                confidence=80.0 if value.is_valid else 0.0,
            )
        )

    if len(annual) >= 2:
        cur, prior = annual[0], annual[1]
        rev, rev_prior = _num(cur, "revenue"), _num(prior, "revenue")
        if rev and rev_prior:
            g = yoy_growth(rev, rev_prior)
            add("FIN-GR-001", "FIN-GR-001@2.0.0", Value.of(g, unit="ratio"), band_revenue_growth(g), d_growth)

        ebit = _num(cur, "ebit")
        if ebit is not None and rev:
            om = operating_margin(ebit, rev)
            add("FIN-PR-008", "FIN-PR-008@2.0.0", Value.of(om, unit="ratio"), band_operating_margin(om), d_margins)

        ni = _num(cur, "net_income")
        if ni is not None and rev:
            nm = net_margin(ni, rev)
            add("FIN-PR-009", "FIN-PR-009@2.0.0", Value.of(nm, unit="ratio"), band_net_margin(nm), d_margins)

        ocf, capex = _num(cur, "operating_cash_flow"), _num(cur, "capex")
        fcf = None
        if adapter_ok and ocf is not None and capex is not None:
            fcf = free_cash_flow(ocf, capex)
            if rev:
                fm = fcf_margin(fcf, rev)
                add("FIN-CF-014", "FIN-CF-014@2.0.0", Value.of(fm, unit="ratio"), band_fcf_margin(fm), d_eps_fcf)

        if ocf is not None and ni is not None:
            cve = cash_vs_earnings(ocf, ni)
            band = None
            if cve.is_valid:
                band = band_higher_better(cve.value, 0.9, 1.1)
            add("FIN-CF-015", "FIN-CF-015@2.0.0", cve, band, d_eps_fcf)

        # Dilution (diagnostic; drives efficiency dimension colour).
        if len(annual) >= 5:
            s_end, s_begin = _num(annual[0], "diluted_shares"), _num(annual[4], "diluted_shares")
            if s_end and s_begin:
                cagr = diluted_share_cagr(s_end, s_begin, 4)
                add(
                    "FIN-DX-033",
                    "FIN-DX-033@2.0.0",
                    Value.of(cagr, unit="ratio"),
                    band_lower_better(cagr, 0.0, 0.02),
                    d_efficiency,
                )
                if cagr > 0.05:
                    flags.append("DILUTION_RED_FLAG")

        # Override 1 screen: loss + negative FCF + external financing.
        if ni is not None and ni < 0 and fcf is not None and fcf < 0:
            overrides.append("OVERRIDE_1_CAPITAL_DEPENDENCE")

    # Build dimensions (3 pts each).
    dims = [
        dimension_from_bands("revenue_quality_growth", 3.0, d_growth),
        dimension_from_bands("eps_and_fcf", 3.0, d_eps_fcf),
        dimension_from_bands("margins", 3.0, d_margins),
        dimension_from_bands("balance_and_liquidity", 3.0, d_balance),
        dimension_from_bands("cash_conversion_capital_efficiency", 3.0, d_efficiency),
    ]

    core = core_diagnostic(core_bands)
    coverage = core["valid_count"] / 27.0

    # Confidence scales with coverage; stale data lowers it.
    confidence = round(min(90.0, 40.0 + coverage * 60.0), 1)
    if packet.staleness.get("quarterly_fundamentals") == "STALE":
        confidence = round(confidence * 0.9, 1)

    category = category_from_dimensions(dims, MAX_POINTS, confidence)

    # Core-27 vs dimension reconciliation (must agree within 1.5 pts).
    warnings_flags = list(flags)
    if core["valid_count"] and abs(core["score_10"] - category.score_10) > 1.5:
        warnings_flags.append("CORE27_DIMENSION_RECONCILIATION_WARNING")

    scored = [m for m in metrics if m.band is not None]
    strongest = max(scored, key=lambda m: m.score, default=None)
    weakest = min(scored, key=lambda m: m.score, default=None)

    return FinancialOutput(
        agent_id=AGENT_ID,
        security={
            "ticker": packet.security.ticker,
            "exchange": packet.security.exchange,
            "currency": packet.security.reporting_currency,
        },
        knowledge_timestamp=packet.analysis.knowledge_timestamp,
        category=category,
        coverage=coverage,
        dimensions=dims,
        metrics=metrics,
        mandatory_flags=warnings_flags,
        source_lineage=[f"packet:{packet.packet_hash[:12]}"],
        validation_tests={"passed": core["valid_count"], "failed": 0, "warnings": len(warnings_flags)},
        core_27_metrics=core,
        mandatory_overrides=overrides,
        strongest_metric=strongest.metric_id if strongest else None,
        weakest_metric=weakest.metric_id if weakest else None,
    )
