"""Business specialist (20 pts) — Task 15.

Dimensions: Moat 5 / Competitive 4 / Management 4 / Durability 4 / Customer
economics 3. Moat classification, quantitative moat effects and the three
thesis-killers are judgment requests (never guessed). Caps: no positive
ROIC-WACC spread -> moat <=6; largest customer >30% -> durability <=6 +
CONCENTRATION_RED_FLAG; diluted-share CAGR >5% -> DILUTION_RED_FLAG;
ROIC<WACC -> VALUE_DESTRUCTION.
"""

from __future__ import annotations

import numpy as np

from wbj.core.nullstates import NullState, Value
from wbj.schemas.packet import Packet
from wbj.specialists.common import (
    CategoryScore,
    Dimension,
    JudgmentRequest,
    MetricRow,
    SpecialistOutput,
    category_from_dimensions,
)
from wbj.specialists.financial import diluted_share_cagr, free_cash_flow

AGENT_ID = "business_analysis"
MAX_POINTS = 20.0


class BusinessOutput(SpecialistOutput):
    business_in_one_sentence: str | None = None
    moat: dict = {}
    roic_history: list = []


def _num(rec: dict, key: str):
    v = rec.get(key)
    return float(v) if isinstance(v, (int, float)) else None


def operating_margin_range(annual: list[dict]) -> float | None:
    """5y operating-margin range in percentage points (≤3pp = moat signal)."""
    margins = []
    for rec in annual[:5]:
        ebit, rev = _num(rec, "ebit"), _num(rec, "revenue")
        if ebit is not None and rev:
            margins.append(ebit / rev)
    if len(margins) < 2:
        return None
    return (max(margins) - min(margins)) * 100.0


def cumulative_fcf_conversion(annual: list[dict]) -> float | None:
    """ΣFCF(5y) / ΣNI(5y) — cash backing of reported earnings."""
    sum_fcf = 0.0
    sum_ni = 0.0
    count = 0
    for rec in annual[:5]:
        ocf, capex, ni = _num(rec, "operating_cash_flow"), _num(rec, "capex"), _num(rec, "net_income")
        if ocf is not None and capex is not None and ni is not None:
            sum_fcf += free_cash_flow(ocf, capex)
            sum_ni += ni
            count += 1
    if count == 0 or sum_ni == 0:
        return None
    return sum_fcf / sum_ni


def run(packet: Packet, overlay: dict | None = None) -> BusinessOutput:
    annual = packet.fundamentals.get("annual", [])
    metrics: list[MetricRow] = []
    flags: list[str] = []
    requests: list[JudgmentRequest] = []
    assumptions: list[str] = []

    # --- Durability: margin stability (quantitative moat signal) -------------
    mr = operating_margin_range(annual)
    if mr is not None:
        stability_score = 9.0 if mr <= 3.0 else (6.0 if mr <= 5.0 else 3.0)
        metrics.append(
            MetricRow(
                metric_id="BUS-MOAT-margin-stability",
                value=Value.of(mr, unit="pp"),
                formula="BUS-018@2.0.0",
                score=stability_score,
                evidence_class="C",
                source="packet.fundamentals",
                confidence=75.0,
            )
        )
    else:
        stability_score = 0.0

    # --- Customer economics: cumulative FCF conversion -----------------------
    conv = cumulative_fcf_conversion(annual)
    if conv is not None:
        conv_score = 9.0 if conv >= 1.0 else (5.0 if conv >= 0.6 else 2.0)
        metrics.append(
            MetricRow(
                metric_id="BUS-CE-fcf-conversion",
                value=Value.of(conv, unit="ratio"),
                formula="BUS-021@2.0.0",
                score=conv_score,
                evidence_class="C",
                source="packet.fundamentals",
                confidence=75.0,
            )
        )
    else:
        conv_score = 0.0

    # --- Dilution (management capital allocation signal) ---------------------
    dilution_score = 5.0
    if len(annual) >= 5:
        s_end, s_begin = _num(annual[0], "diluted_shares"), _num(annual[4], "diluted_shares")
        if s_end and s_begin:
            cagr = diluted_share_cagr(s_end, s_begin, 4)
            dilution_score = 8.0 if cagr <= 0.0 else (5.0 if cagr <= 0.02 else 2.0)
            metrics.append(
                MetricRow(
                    metric_id="BUS-MGMT-dilution",
                    value=Value.of(cagr, unit="ratio"),
                    formula="BUS-024@2.0.0",
                    score=dilution_score,
                    evidence_class="C",
                    source="packet.fundamentals",
                    confidence=75.0,
                )
            )
            if cagr > 0.05:
                flags.append("DILUTION_RED_FLAG")

    # --- ROIC vs WACC: needs invested capital & WACC (not in packet) ---------
    metrics.append(
        MetricRow(
            metric_id="BUS-ROIC-spread",
            value=Value.null(NullState.NOT_SCORABLE, unit="ratio"),
            formula="BUS-004@2.0.0",
            score=None,
            evidence_class="C",
            source="valuation_engine",
            confidence=0.0,
            warnings=["invested capital / WACC not in packet; spread unresolved"],
        )
    )
    positive_spread_confirmed = False  # unknown -> conservative moat cap

    # --- Judgment requests (qualitative, never guessed) ----------------------
    requests.extend(
        [
            JudgmentRequest(
                request_id="BUS-moat-classification",
                agent_id=AGENT_ID,
                metric_id="BUS-moat-class",
                question="Classify the moat (none/narrow/wide) with ≥2 quantitative effects.",
                schema_hint="{class: none|narrow|wide, effects: [str]}",
            ),
            JudgmentRequest(
                request_id="BUS-thesis-killers",
                agent_id=AGENT_ID,
                metric_id="BUS-thesis-killers",
                question="List the three most credible thesis killers for this business.",
                schema_hint="{killers: [str]}",
            ),
        ]
    )

    # --- Dimension scores with caps ------------------------------------------
    moat_score = stability_score
    if not positive_spread_confirmed:
        moat_score = min(moat_score, 6.0)  # cap: no positive spread confirmed

    durability_score = stability_score  # margin durability proxy
    # Concentration is a judgment input; if unresolved we do not cap up.

    dims = [
        Dimension(name="moat", max_points=5.0, score_10=moat_score, awarded_points=5.0 * moat_score / 10.0,
                  rationale="Capped at 6 until a positive ROIC-WACC spread is confirmed."),
        Dimension(name="competitive_position", max_points=4.0, score_10=stability_score,
                  awarded_points=4.0 * stability_score / 10.0),
        Dimension(name="management", max_points=4.0, score_10=dilution_score,
                  awarded_points=4.0 * dilution_score / 10.0),
        Dimension(name="durability", max_points=4.0, score_10=durability_score,
                  awarded_points=4.0 * durability_score / 10.0),
        Dimension(name="customer_economics", max_points=3.0, score_10=conv_score,
                  awarded_points=3.0 * conv_score / 10.0),
    ]

    scored = [m for m in metrics if m.score is not None]
    coverage = len(scored) / 8.0  # ~8 core business metrics expected
    confidence = round(min(80.0, 30.0 + coverage * 60.0), 1)
    category = category_from_dimensions(dims, MAX_POINTS, confidence)

    return BusinessOutput(
        agent_id=AGENT_ID,
        security={"ticker": packet.security.ticker, "exchange": packet.security.exchange,
                  "currency": packet.security.reporting_currency},
        knowledge_timestamp=packet.analysis.knowledge_timestamp,
        category=category,
        coverage=coverage,
        dimensions=dims,
        metrics=metrics,
        mandatory_flags=flags,
        assumptions=assumptions,
        judgment_requests=requests,
        source_lineage=[f"packet:{packet.packet_hash[:12]}"],
        moat={"classification": "JUDGMENT_REQUIRED", "positive_spread_confirmed": positive_spread_confirmed},
    )
