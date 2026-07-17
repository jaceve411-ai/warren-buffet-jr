"""Cross-category contradiction lookup (Task 21).

`contradictions` reports tension between category scores per
`Cerebro/00_main_agent/CONTRADICTION_RESOLUTION.md`. It only describes — it
never mutates a score.
"""

from __future__ import annotations


def contradictions(cats: dict) -> list[str]:
    """Return descriptive contradiction notes for the six category points."""
    notes: list[str] = []
    b, f, m = cats["business"], cats["financial"], cats["market"]
    t, r, v = cats["technical"], cats["risk"], cats["valuation"]

    if b >= 16 and t < 10:
        notes.append("Quality watch/wait: strong business, weak technical tape.")
    if t >= 16 and b < 10:
        notes.append("Momentum without moat: tradeable, not investable on fundamentals.")
    if v >= 8 and r < 8:
        notes.append("Cheap but fragile: attractive valuation, elevated risk.")
    if r >= 12 and v < 4:
        notes.append("Safe but expensive: resilient balance sheet, rich price.")
    if m >= 16 and f < 8:
        notes.append("Growth without profitability: strong market, weak financials.")
    if b >= 16 and v < 4:
        notes.append("Great business, poor entry price: revisit on a pullback.")
    return notes
