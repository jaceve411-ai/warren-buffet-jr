"""Generate a self-contained static HTML snapshot of the WBJ analysis page.

Runs the REAL engine (compute, scorecard, targets, narrative) on a rich
in-memory packet — no network, no SEC. Produces the exact same UI as the
live webapp by reusing its PAGE template, with the analysis result injected
so the page renders with zero backend. For preview only: the company is a
synthetic demo ("DEMO CO"), numbers are genuine engine output for that packet.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from webapp import PAGE  # noqa: E402  (reuse the live template verbatim)

from wbj.cli import _compute  # noqa: E402
from wbj.targets import narrative, price_targets  # noqa: E402


def _series(vals):
    return [{"end": e, "val": v, "form": "10-K", "fp": "FY"} for e, v in vals]


def build_packet() -> dict:
    years = ["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"]
    return {
        "ticker": "DEMO",
        "entity": "DEMO CO (empresa de ejemplo)",
        "as_of": "2026-07-17",
        "annual": {
            "revenue": _series(list(zip(years, [42e9, 55e9, 71e9, 92e9, 118e9]))),
            "net_income": _series(list(zip(years, [9e9, 13e9, 18e9, 24e9, 32e9]))),
            "operating_cash_flow": _series(list(zip(years, [11e9, 15e9, 21e9, 28e9, 37e9]))),
            "capex": _series(list(zip(years, [3e9, 4e9, 5e9, 6e9, 7e9]))),
            "long_term_debt": _series([(years[-1], 18e9)]),
            "equity": _series(list(zip(years[-2:], [52e9, 68e9]))),
            "operating_income": _series([(years[-1], 34e9)]),
            "gross_profit": _series([(years[-1], 72e9)]),
            "interest_expense": _series([(years[-1], 0.6e9)]),
            "diluted_shares": _series([(years[-1], 2.4e9)]),
        },
    }


def history(packet: dict) -> list[dict]:
    rev = {r["end"]: r["val"] for r in packet["annual"]["revenue"]}
    ni = {r["end"]: r["val"] for r in packet["annual"]["net_income"]}
    rows = []
    for end in sorted(rev)[-6:]:
        rows.append({
            "year": end[:4],
            "revenue": rev[end],
            "margin": (ni[end] / rev[end]) if end in ni and rev[end] else None,
        })
    return rows


def synth_chart(price: float, days: int = 252) -> list[dict]:
    """Deterministic, smooth 1y daily series ending at `price` (illustrative)."""
    import datetime
    import math

    start = datetime.date(2025, 7, 17)
    out = []
    base = price * 0.72
    for i in range(days):
        t = i / (days - 1)
        trend = base + (price - base) * t
        wave = math.sin(t * math.pi * 5) * price * 0.03
        wave2 = math.sin(t * math.pi * 17) * price * 0.012
        v = trend + wave + wave2
        d = start + datetime.timedelta(days=int(i * 365 / days))
        out.append({"time": d.isoformat(), "value": round(v, 2)})
    out[-1]["value"] = round(price, 2)
    return out


def main() -> None:
    packet = build_packet()
    price = 148.0
    result = _compute(packet)
    result["entity"] = packet["entity"]
    result["targets"] = price_targets(packet, price)
    result["narrative"] = narrative(packet, result["scorecard"], result["targets"])
    result["history"] = history(packet)
    result["chart"] = synth_chart(price)

    d_json = json.dumps(result)

    # Reuse the live template; swap the network fetch for injected data and
    # auto-run on load. Everything else (CSS, render fns) stays identical.
    page = PAGE
    page = page.replace(
        "    const r = await fetch('/api/analyze?ticker=' + encodeURIComponent(t));\n"
        "    if (!r.ok) throw new Error((await r.json()).error || r.status);\n"
        "    const d = await r.json();",
        "    const d = window.__D;",
    )
    # Disable the (backend-less) search + discover network calls gracefully and
    # auto-render the injected analysis once the page loads.
    boot = (
        "\n<script>\n"
        f"window.__D = {d_json};\n"
        "window.addEventListener('load', () => {\n"
        "  const b = document.createElement('div');\n"
        "  b.style.cssText = 'max-width:1040px;margin:0 auto 4px;padding:10px 14px;"
        "border-radius:12px;background:#fef4e2;color:#8a5a00;font:13px system-ui;"
        "font-weight:600';\n"
        "  b.innerHTML = '\\u26a0\\ufe0f Vista previa est\\u00e1tica \\u2014 empresa de ejemplo "
        "(DEMO CO), n\\u00fameros reales del motor. La b\\u00fasqueda en vivo requiere correr "
        "el webapp localmente.';\n"
        "  document.querySelector('.wrap').prepend(b);\n"
        "  run(window.__D.ticker);\n"
        "});\n"
        "</script>\n"
    )
    page = page.replace("</body></html>", boot + "</body></html>")

    out = Path(__file__).resolve().parents[2] / "scratch_preview.html"
    if len(sys.argv) > 1:
        out = Path(sys.argv[1])
    out.write_text(page, encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
