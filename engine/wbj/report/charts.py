"""Report charts (Task 22), honoring the four visualization rules in the
root CLAUDE.md:

1. Never a single line — always show a range/band.
2. Label every scenario's assumptions (growth, margin).
3. Historical solid, projected dotted.
4. The chart only illustrates the computed logic.

All functions take data plus an `out_path`, save a 150-dpi PNG via the Agg
backend, and return the path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def price_levels_chart(df, levels: list[dict], smas: dict, out_path: Path) -> Path:
    """Price with shaded support/resistance zone bands and SMA overlays."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(df)), df["close"], color="#1f77b4", lw=1.2, label="Close (historical)")
    for name, series in smas.items():
        ax.plot(range(len(series)), series, lw=1.0, ls="--", label=name)
    for lvl in levels:
        lower, upper = lvl.get("lower"), lvl.get("upper")
        if lower is not None and upper is not None:
            ax.axhspan(lower, upper, alpha=0.15,
                       color="#2ca02c" if lvl.get("type") == "support" else "#d62728")
    ax.set_title("Price with important-level zones")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def scenario_fan_chart(history: list[float], scenarios: list[dict], out_path: Path) -> Path:
    """History as a solid line; each scenario as a dotted projected BAND with
    its assumptions labeled on-chart.

    Raises `ValueError("single-line projection prohibited")` if any scenario
    lacks a band width (low == high) — a single projected line "lies with
    confidence" (rule 1)."""
    for s in scenarios:
        if s.get("low") is None or s.get("high") is None or s["low"] == s["high"]:
            raise ValueError("single-line projection prohibited")

    fig, ax = plt.subplots(figsize=(10, 5))
    hx = list(range(len(history)))
    ax.plot(hx, history, color="#1f77b4", lw=1.5, label="Historical")

    start = len(history) - 1
    for s in scenarios:
        fx = [start, start + s.get("horizon", 12)]
        low = [history[-1], s["low"]]
        high = [history[-1], s["high"]]
        ax.plot(fx, low, ls=":", color=s.get("color", "#ff7f0e"))
        ax.plot(fx, high, ls=":", color=s.get("color", "#ff7f0e"))
        ax.fill_between(fx, low, high, alpha=0.12, color=s.get("color", "#ff7f0e"))
        label = f"{s.get('name', 'scenario')}: growth={s.get('growth')}, margin={s.get('margin')}"
        ax.annotate(label, xy=(fx[1], s["high"]), fontsize=7)

    ax.set_title("Scenario fan (historical solid, projections dotted)")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def scorecard_chart(category_points: dict, maxima: dict, out_path: Path) -> Path:
    """Horizontal bars of awarded vs maximum category points."""
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(category_points.keys())
    y = range(len(names))
    ax.barh(list(y), [maxima[n] for n in names], color="#dddddd", label="Max")
    ax.barh(list(y), [category_points[n] for n in names], color="#1f77b4", label="Awarded")
    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    ax.set_title("Category scorecard")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def football_field_chart(reference_bands: list[dict], current_price: float, out_path: Path) -> Path:
    """Valuation ranges per model/scenario as horizontal bars + a price line."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, band in enumerate(reference_bands):
        ax.barh(i, band["high"] - band["low"], left=band["low"], height=0.5,
                color="#9467bd", alpha=0.6)
    ax.set_yticks(range(len(reference_bands)))
    ax.set_yticklabels([b.get("name", f"model_{i}") for i, b in enumerate(reference_bands)])
    ax.axvline(current_price, color="#d62728", lw=1.5, label=f"Price {current_price}")
    ax.set_title("Valuation football field")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
