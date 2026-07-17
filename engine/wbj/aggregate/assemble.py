"""Assemble the FinalReport from the six frozen specialist outputs (Task 21).

Ties together gates, overrides, contradictions and price synthesis into the
auditable `FinalReport`. Applies the mandatory report rules: insider trades
count as significant only above $1,000,000 total, per-share is suppressed on
an unresolved facts-table conflict, and an Avoid classification must carry a
revisit date/event.
"""

from __future__ import annotations

from wbj.aggregate.contradiction import contradictions
from wbj.aggregate.gates import apply_gates
from wbj.aggregate.overrides import apply_overrides
from wbj.aggregate.synthesis import synthesize_levels
from wbj.schemas.final_report import CategoryScorecardRow, FinalReport
from wbj.schemas.packet import Packet
from wbj.specialists.common import SpecialistOutput

INSIDER_SIGNIFICANCE_USD = 1_000_000.0


def _insider_total(row: dict) -> float:
    for key in ("value", "transactionValue", "total", "amount"):
        v = row.get(key)
        if isinstance(v, (int, float)):
            return abs(float(v))
    shares = row.get("shares") or row.get("securitiesTransacted")
    price = row.get("price") or row.get("transactionPrice")
    if isinstance(shares, (int, float)) and isinstance(price, (int, float)):
        return abs(float(shares) * float(price))
    return 0.0


def build_final_report(
    outputs: dict[str, SpecialistOutput], packet: Packet
) -> FinalReport:
    """Aggregate the six category outputs and the packet into a FinalReport."""
    cats = {name: out.category.awarded_points for name, out in outputs.items()}
    confidences = {name: out.category.confidence for name, out in outputs.items()}
    coverages = {name: out.coverage for name, out in outputs.items()}
    total_confidence = sum(confidences.values()) / len(confidences) if confidences else 0.0

    overrides = apply_overrides(outputs, packet)
    profile = apply_gates(cats, confidences, coverages, overrides, total_confidence)

    scorecard = [
        CategoryScorecardRow(
            category=name, max_points=out.category.max_points,
            awarded_points=out.category.awarded_points, score_10=out.category.score_10,
            confidence=out.category.confidence, coverage=out.coverage,
        )
        for name, out in outputs.items()
    ]

    price_v = packet.facts_table.get("price")
    price = price_v.value if price_v and price_v.is_valid else 0.0
    tech, val = outputs.get("technical"), outputs.get("valuation")
    atr = 0.0
    if tech is not None:
        atr = tech.indicators.get("atr", 0.0) if hasattr(tech, "indicators") else 0.0
    levels = synthesize_levels(tech, val, price, atr) if tech and val else []

    per_share_suppressed = any(o.action == "SUPPRESS_PER_SHARE" for o in overrides)

    # Missing / conflicted data.
    missing: list[str] = []
    for k, v in packet.facts_table.items():
        if v.is_null and v.state:
            missing.append(f"{k}: {v.state.value}")
    for out in outputs.values():
        for m in out.metrics:
            if m.value.is_null and m.value.state and m.value.state.value == "NOT_SCORABLE":
                missing.append(f"{out.agent_id}:{m.metric_id}: NOT_SCORABLE")

    # Insider activity (>$1M flagged significant).
    insider_activity = []
    for row in packet.insiders:
        if isinstance(row, dict):
            total = _insider_total(row)
            insider_activity.append({**row, "total_usd": total,
                                     "significant": total > INSIDER_SIGNIFICANCE_USD})

    thesis_killers: list[str] = []
    monitoring: list[str] = []
    for out in outputs.values():
        for req in out.judgment_requests:
            if "thesis" in req.metric_id.lower() or "killer" in req.request_id.lower():
                thesis_killers.append(req.question)
            else:
                monitoring.append(req.question)

    revisit = None
    if profile.profile in ("Avoid/Wait", "Speculative"):
        revisit = "Next quarterly earnings release or a confirmed technical breakout."

    raw = profile.raw_score
    thesis = [
        f"{packet.security.ticker} scores {raw:.1f}/100 raw, landing in the '{profile.band}' band.",
        f"Research classification is '{profile.profile}'.",
        f"Business {cats.get('business', 0):.1f}/20 and financial {cats.get('financial', 0):.1f}/15 anchor the fundamental view.",
        f"Market {cats.get('market', 0):.1f}/20 and technical {cats.get('technical', 0):.1f}/20 frame the growth and tape.",
        f"Risk {cats.get('risk', 0):.1f}/15 (higher is safer) and valuation {cats.get('valuation', 0):.1f}/10 bound the downside and price.",
        f"Aggregate confidence is {total_confidence:.0f}/100; {len(missing)} data points are missing or conflicted.",
        ("Per-share value is suppressed by an unresolved facts-table conflict."
         if per_share_suppressed else "Per-share reference values are shown by scenario, never as a single number."),
    ]

    return FinalReport(
        security={"ticker": packet.security.ticker, "exchange": packet.security.exchange,
                  "currency": packet.security.reporting_currency},
        knowledge_timestamp=packet.analysis.knowledge_timestamp,
        profile=profile,
        category_scorecard=scorecard,
        executive_thesis=thesis,
        important_levels=levels,
        valuation_scenarios=getattr(val, "scenarios_detail", []) if val else [],
        reverse_dcf=getattr(val, "reverse_dcf", {}) if val else {},
        thesis_killers=thesis_killers,
        monitoring_triggers=monitoring,
        overrides=overrides,
        contradictions=contradictions(cats),
        insider_activity=insider_activity,
        institutional_holders=[h for h in packet.institutional_holders if isinstance(h, dict)],
        missing_or_conflicted_data=missing,
        per_share_suppressed=per_share_suppressed,
        revisit_date_or_event=revisit,
        audit={
            "packet_hash": packet.packet_hash,
            "output_hashes": {name: out.output_hash for name, out in outputs.items()},
            "formula_versions": "2.0.0",
            "validation_summary": {name: out.validation_tests for name, out in outputs.items()},
        },
    )
