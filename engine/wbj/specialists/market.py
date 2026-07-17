"""Market & Growth specialist (20 pts) — Task 16.

Dimensions: TAM 5 / Revisions 4 / Catalysts 4 / Runway & share 4 / Operating
leverage 3. Revision breadth needs ≥5 estimates (else NOT_SCORABLE — on the
prohibited-imputation list). Catalyst impact = P·Impact·EvidenceQuality·
TimeDecay with TimeDecay = exp(-ln2·months/12); P/Impact/EvidenceQuality and
the TAM tier are judgment requests. Narrative-only catalysts cap that
dimension at 3; TAM confidence <60 caps TAM at 6.
"""

from __future__ import annotations

import math

from wbj.core.nullstates import NullState, Value
from wbj.schemas.packet import Packet
from wbj.specialists.common import (
    Dimension,
    JudgmentRequest,
    MetricRow,
    SpecialistOutput,
    category_from_dimensions,
)

AGENT_ID = "market_analysis"
MAX_POINTS = 20.0
_LN2 = math.log(2.0)

# TAM tier -> confidence (tier assignment itself is a judgment request).
TAM_TIER_CONFIDENCE = {1: 100, 2: 85, 3: 70, 4: 45, 5: 0}


class MarketOutput(SpecialistOutput):
    tam: dict = {}
    revisions: dict = {}
    catalysts: list = []


def time_decay(months_to_event: float) -> float:
    """Catalyst time decay = exp(-ln2 · months / 12); 12 months -> 0.5."""
    return math.exp(-_LN2 * months_to_event / 12.0)


def catalyst_impact(p: float, impact: float, evidence_quality: float, months_to_event: float) -> float:
    """P × Impact × EvidenceQuality × TimeDecay."""
    return p * impact * evidence_quality * time_decay(months_to_event)


def runway_years(current: float, target: float, g: float) -> float:
    """ln(target/current)/ln(1+g) — years of growth headroom to the target."""
    return math.log(target / current) / math.log(1.0 + g)


def revision_breadth(estimates: list) -> Value:
    """Fraction of estimates revised up. Requires ≥5 estimates, else
    NOT_SCORABLE (estimates are on the prohibited-imputation list)."""
    if not isinstance(estimates, list) or len(estimates) < 5:
        return Value.null(NullState.NOT_SCORABLE, unit="ratio",
                          warnings=["<5 estimates; revision breadth not scorable"])
    ups = sum(1 for e in estimates if isinstance(e, dict) and e.get("revision", 0) > 0)
    return Value.of(ups / len(estimates), unit="ratio")


def run(packet: Packet, overlay: dict | None = None) -> MarketOutput:
    metrics: list[MetricRow] = []
    flags: list[str] = []
    requests: list[JudgmentRequest] = []

    # --- Revisions dimension (data-driven; NOT_SCORABLE without ≥5) ----------
    est = packet.estimates.get("analyst_estimates")
    est_list = est if isinstance(est, list) else []
    breadth = revision_breadth(est_list)
    if breadth.is_valid:
        rev_score = 8.0 if breadth.value >= 0.6 else (5.0 if breadth.value >= 0.4 else 2.0)
    else:
        rev_score = 0.0
    metrics.append(
        MetricRow(metric_id="MKT-REV-breadth", value=breadth, formula="MKT-011@2.0.0",
                  score=(rev_score if breadth.is_valid else None), evidence_class="E",
                  source="packet.estimates", confidence=(60.0 if breadth.is_valid else 0.0))
    )

    # --- TAM, Catalysts, Runway & share: judgment-driven ---------------------
    requests.extend([
        JudgmentRequest(request_id="MKT-tam-tier", agent_id=AGENT_ID, metric_id="MKT-tam-tier",
                        question="Assign the TAM evidence tier (1-5).",
                        schema_hint="{tier: 1|2|3|4|5, source: str}"),
        JudgmentRequest(request_id="MKT-catalysts", agent_id=AGENT_ID, metric_id="MKT-catalysts",
                        question="For each catalyst give P, Impact, EvidenceQuality and months_to_event.",
                        schema_hint="{catalysts: [{p, impact, evidence_quality, months}]}"),
    ])

    tam_confidence = 0  # unknown until TAM tier judgment supplied
    tam_score = min(0.0, 6.0)  # no tier -> 0; cap 6 applies once scored
    catalyst_score = 3.0  # narrative-only cap until quantified catalysts arrive
    runway_score = 0.0

    # --- Operating leverage: incremental margin from 2 most recent years -----
    annual = packet.fundamentals.get("annual", [])
    oplev_score = 0.0
    if len(annual) >= 2:
        cur, prior = annual[0], annual[1]
        rev0, rev1 = cur.get("revenue"), prior.get("revenue")
        ebit0, ebit1 = cur.get("ebit"), prior.get("ebit")
        if all(isinstance(x, (int, float)) for x in (rev0, rev1, ebit0, ebit1)) and rev0 != rev1:
            incremental_margin = (ebit0 - ebit1) / (rev0 - rev1)
            oplev_score = 8.0 if incremental_margin >= 0.4 else (5.0 if incremental_margin >= 0.2 else 2.0)
            metrics.append(
                MetricRow(metric_id="MKT-OPLEV-incremental-margin",
                          value=Value.of(incremental_margin, unit="ratio"), formula="MKT-020@2.0.0",
                          score=oplev_score, evidence_class="C", source="packet.fundamentals",
                          confidence=70.0)
            )

    dims = [
        Dimension(name="tam", max_points=5.0, score_10=tam_score, awarded_points=5.0 * tam_score / 10.0,
                  rationale="TAM tier is a judgment request; capped at 6 if confidence <60."),
        Dimension(name="revisions", max_points=4.0, score_10=rev_score, awarded_points=4.0 * rev_score / 10.0),
        Dimension(name="catalysts", max_points=4.0, score_10=catalyst_score,
                  awarded_points=4.0 * catalyst_score / 10.0,
                  rationale="Narrative-only catalysts cap the dimension at 3."),
        Dimension(name="runway_and_share", max_points=4.0, score_10=runway_score,
                  awarded_points=4.0 * runway_score / 10.0),
        Dimension(name="operating_leverage", max_points=3.0, score_10=oplev_score,
                  awarded_points=3.0 * oplev_score / 10.0),
    ]

    scored = [m for m in metrics if m.score is not None]
    coverage = len(scored) / 5.0
    confidence = round(min(70.0, 25.0 + coverage * 50.0), 1)
    category = category_from_dimensions(dims, MAX_POINTS, confidence)

    return MarketOutput(
        agent_id=AGENT_ID,
        security={"ticker": packet.security.ticker, "exchange": packet.security.exchange,
                  "currency": packet.security.reporting_currency},
        knowledge_timestamp=packet.analysis.knowledge_timestamp,
        category=category, coverage=coverage, dimensions=dims, metrics=metrics,
        mandatory_flags=flags, judgment_requests=requests,
        source_lineage=[f"packet:{packet.packet_hash[:12]}"],
        tam={"tier": "JUDGMENT_REQUIRED", "confidence": tam_confidence},
    )
