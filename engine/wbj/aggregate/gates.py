"""Scoring gates and descriptive bands (Task 21).

Implements `Cerebro/00_main_agent/SCORING_AND_GATES.md`'s verbatim gate
table and the descriptive bands. Gates read frozen category points, their
confidences and coverages, and any mandatory overrides; they never mutate a
score.
"""

from __future__ import annotations

from wbj.schemas.final_report import Override, ProfileResult

CATEGORY_KEYS = ("business", "financial", "market", "technical", "risk", "valuation")
_MIN_COVERAGE = 0.70


def raw_total(category_points: list[float]) -> float:
    """MAIN-002 — the raw 100-point total is the sum of category points."""
    return sum(category_points)


def descriptive_band(raw: float) -> str:
    """90 Elite / 80 Strong / 70 Conditional / 60 Mixed / 50 Weak / <50 Avoid."""
    if raw >= 90:
        return "Elite"
    if raw >= 80:
        return "Strong"
    if raw >= 70:
        return "Conditional"
    if raw >= 60:
        return "Mixed"
    if raw >= 50:
        return "Weak"
    return "Avoid"


def evaluate_momentum(cats: dict, confidences: dict) -> tuple[bool, list[str]]:
    raw = sum(cats.values())
    reasons: list[str] = []
    if raw < 78:
        reasons.append("raw_total<78")
    if cats["technical"] < 17:
        reasons.append("technical<17")
    if cats["market"] < 16:
        reasons.append("market<16")
    if cats["business"] + cats["financial"] < 28:
        reasons.append("business+financial<28")
    if cats["risk"] < 8:
        reasons.append("risk<8")
    if confidences.get("technical", 0) < 70:
        reasons.append("technical_confidence<70")
    return (not reasons, reasons)


def evaluate_quality(cats: dict, confidences: dict) -> tuple[bool, list[str]]:
    raw = sum(cats.values())
    reasons: list[str] = []
    if raw < 80:
        reasons.append("raw_total<80")
    if cats["business"] < 16:
        reasons.append("business<16")
    if cats["financial"] < 11:
        reasons.append("financial<11")
    if cats["risk"] < 10:
        reasons.append("risk<10")
    if cats["valuation"] < 5:
        reasons.append("valuation<5")
    if cats["technical"] < 12:
        reasons.append("technical<12")
    return (not reasons, reasons)


def evaluate_value(cats: dict, confidences: dict) -> tuple[bool, list[str]]:
    raw = sum(cats.values())
    reasons: list[str] = []
    if raw < 75:
        reasons.append("raw_total<75")
    if cats["valuation"] < 8:
        reasons.append("valuation<8")
    if cats["business"] < 13:
        reasons.append("business<13")
    if cats["risk"] < 10:
        reasons.append("risk<10")
    if cats["technical"] < 9:
        reasons.append("technical<9")
    return (not reasons, reasons)


def apply_gates(
    cats: dict,
    confidences: dict,
    coverages: dict,
    overrides: list[Override] | None = None,
    total_confidence: float = 100.0,
    runway_unfunded: bool = False,
    critical_incomplete: bool = False,
) -> ProfileResult:
    """Resolve the research profile from the gate table, honoring overrides."""
    overrides = overrides or []
    raw = sum(cats.values())
    band = descriptive_band(raw)
    reasons: list[str] = []

    momentum = evaluate_momentum(cats, confidences)
    quality = evaluate_quality(cats, confidences)
    value = evaluate_value(cats, confidences)
    evaluations = {
        "momentum": {"passed": momentum[0], "failing": momentum[1]},
        "quality": {"passed": quality[0], "failing": quality[1]},
        "value": {"passed": value[0], "failing": value[1]},
    }

    override_actions = {o.action for o in overrides}
    low_coverage = [k for k in CATEGORY_KEYS if coverages.get(k, 1.0) < _MIN_COVERAGE]

    def result(profile: str) -> ProfileResult:
        return ProfileResult(
            profile=profile, raw_score=raw, band=band, reasons=reasons, gate_evaluations=evaluations
        )

    # 1. Hard avoid.
    if "WAIT_AVOID" in override_actions or "CAP_AVOID_SPECULATIVE" in override_actions or raw < 50:
        reasons.append("override or raw<50 forces Avoid/Wait")
        return result("Avoid/Wait")

    # 2. Speculative caps.
    if cats["risk"] <= 4 or total_confidence < 60 or critical_incomplete or runway_unfunded or "CAP_SPECULATIVE" in override_actions:
        reasons.append("risk<=4 / low confidence / incomplete / unfunded runway -> Speculative")
        return result("Speculative")

    # 3. Gate eligibility requires >=0.70 coverage in every category.
    if low_coverage:
        reasons.append(f"gate-ineligible: coverage<0.70 in {low_coverage}")
        return result("Conditional/Watch" if raw >= 60 else "Speculative")

    # 4-6. Major gates in priority.
    if momentum[0]:
        return result("Momentum")
    if quality[0] and "NO_ELITE_QUALITY" not in override_actions:
        return result("Quality")
    if value[0]:
        return result("Value")

    # 7-8. Fallbacks.
    if raw >= 60:
        reasons.append("no major gate met; raw>=60")
        return result("Conditional/Watch")
    reasons.append("no gate met; raw<60")
    return result("Avoid/Wait")
