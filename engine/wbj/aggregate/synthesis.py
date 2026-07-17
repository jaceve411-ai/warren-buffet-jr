"""Price-level synthesis (Task 21), per PRICE_LEVEL_SYNTHESIS.md.

Combines technical zones and intrinsic (valuation) references into one
ranked table. It NEVER averages a technical level with an intrinsic value —
they are different kinds of evidence — but flags *confluence* when they
overlap within `max(0.50·ATR, 0.0075·price)` and at least one side is
technical.
"""

from __future__ import annotations


def confluence_tolerance(atr: float, price: float) -> float:
    """max(0.50·ATR, 0.0075·price)."""
    return max(0.50 * atr, 0.0075 * price)


def _distance(price: float, level: float, atr: float) -> dict:
    return {
        "distance_percent": (level - price) / price * 100.0 if price else None,
        "distance_atr": (level - price) / atr if atr else None,
    }


def synthesize_levels(technical_output, valuation_output, price: float, atr: float) -> list[dict]:
    """Return the merged, confluence-tagged level table.

    Each entry keeps its own price and `source` (`technical` or `intrinsic`);
    values from the two lenses are never blended."""
    tol = confluence_tolerance(atr, price)
    levels: list[dict] = []

    # Technical zones (support + resistance) from the levels engine output.
    tech_levels: list[float] = []
    important = getattr(technical_output, "important_levels", {}) or {}
    for side in ("support", "resistance"):
        for zone in important.get(side, []):
            center = zone.get("center")
            if center is None:
                continue
            tech_levels.append(center)
            levels.append({
                "source": "technical", "kind": side, "price": center,
                "status": zone.get("status"), "strength": zone.get("strength_0_100"),
                "confluence": False, **_distance(price, center, atr),
            })

    # Intrinsic references from valuation scenarios.
    intrinsic_levels: list[float] = []
    for scen in getattr(valuation_output, "scenarios_detail", []) or []:
        val = scen.get("value")
        if val is None:
            continue
        intrinsic_levels.append(val)
        levels.append({
            "source": "intrinsic", "kind": f"scenario_{scen.get('name')}", "price": val,
            "confluence": False, **_distance(price, val, atr),
        })

    # Confluence: an intrinsic level within tolerance of a technical level.
    for entry in levels:
        if entry["source"] == "intrinsic":
            if any(abs(entry["price"] - tl) <= tol for tl in tech_levels):
                entry["confluence"] = True
        else:  # technical
            if any(abs(entry["price"] - il) <= tol for il in intrinsic_levels):
                entry["confluence"] = True

    levels.sort(key=lambda e: abs(e["price"] - price))
    return levels
