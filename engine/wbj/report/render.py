"""Final report renderer (Task 23).

Writes `report.json` (schema-validated) and an English `report.md` with the
mandatory sections from the root CLAUDE.md §"Contenido obligatorio del
reporte final" and FINAL_REPORT_SCHEMA.md. Enforced rules: an Avoid
classification must carry a revisit date/event; insider trades are flagged
"significant" only above $1,000,000 total; no forbidden certainty language.
"""

from __future__ import annotations

import json
from pathlib import Path

from wbj.schemas.final_report import FinalReport

INSIDER_SIGNIFICANCE_USD = 1_000_000.0
_AVOID_PROFILES = ("Avoid/Wait", "Avoid")
_FORBIDDEN = ("guaranteed target", "must hold", "certain floor")


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _markdown(final: FinalReport) -> str:
    s = final.security
    p = final.profile
    lines: list[str] = []

    lines.append(f"# Warren Buffett Jr — Research Report: {s.get('ticker', '?')}")
    lines.append(f"*Report version {final.report_version} · knowledge timestamp {final.knowledge_timestamp}*\n")

    lines.append("## Executive summary")
    lines.extend(f"- {sent}" for sent in final.executive_thesis)
    lines.append("")

    lines.append("## Research classification")
    lines.append(f"- Classification: **{p.profile}** (raw {p.raw_score:.1f}/100, band {p.band})")
    for reason in p.reasons:
        lines.append(f"- {reason}")
    if p.profile in _AVOID_PROFILES or p.profile == "Speculative":
        lines.append(f"- **Revisit:** {final.revisit_date_or_event}")
    lines.append("")

    lines.append("## Category scorecard")
    lines.append(_table(
        ["Category", "Awarded", "Max", "Score/10", "Confidence", "Coverage"],
        [[r.category, f"{r.awarded_points:.1f}", f"{r.max_points:.0f}",
          f"{r.score_10:.1f}", f"{r.confidence:.0f}", f"{r.coverage:.2f}"]
         for r in final.category_scorecard],
    ))
    lines.append("")

    lines.append("## Price scenario ranges (assumptions declared, never a single price)")
    if final.per_share_suppressed:
        lines.append("- Per-share value **suppressed** by an unresolved facts-table conflict.")
    elif final.valuation_scenarios:
        lines.append(_table(
            ["Scenario", "Value", "Probability"],
            [[sc.get("name"), f"{sc.get('value', 'n/a')}", sc.get("probability", "n/a")]
             for sc in final.valuation_scenarios],
        ))
    else:
        lines.append("- Insufficient data to reach an investment conclusion for scenario values.")
    if final.reverse_dcf:
        lines.append(f"- Reverse DCF implied growth: {final.reverse_dcf.get('implied_growth')}")
    lines.append("")

    lines.append("## Important levels")
    if final.important_levels:
        lines.append(_table(
            ["Source", "Kind", "Price", "Distance %", "Distance ATR", "Confluence"],
            [[lv.get("source"), lv.get("kind"), f"{lv.get('price')}",
              f"{lv.get('distance_percent')}", f"{lv.get('distance_atr')}", lv.get("confluence")]
             for lv in final.important_levels],
        ))
    else:
        lines.append("- No confirmed levels available.")
    lines.append("")

    lines.append("## Notable 13F holders & management track record")
    if final.institutional_holders:
        lines.append(_table(["Holder", "Shares", "As of"],
                            [[h.get("holder"), h.get("shares"), h.get("dateReported")]
                             for h in final.institutional_holders]))
    else:
        lines.append("- No 13F holders in packet.")
    lines.append("")

    lines.append("## Insider activity (significant = total > $1,000,000)")
    significant = [t for t in final.insider_activity if t.get("total_usd", 0) > INSIDER_SIGNIFICANCE_USD]
    if significant:
        lines.append(_table(["Insider", "Total USD", "Significant"],
                            [[t.get("holder") or t.get("name") or t.get("reportingName", "?"),
                              f"{t.get('total_usd', 0):,.0f}", "YES"] for t in significant]))
    else:
        lines.append("- No insider trades exceeding $1,000,000.")
    lines.append("")

    lines.append("## Thesis killers & monitoring triggers")
    for k in final.thesis_killers:
        lines.append(f"- Thesis killer: {k}")
    for m in final.monitoring_triggers:
        lines.append(f"- Monitor: {m}")
    lines.append("")

    lines.append("## Profile fit (Victor Gonzalez)")
    lines.append("- Capital $25,000; max position 30–60%; horizon 3–5 years; US-only, no forex.")
    lines.append("")

    lines.append("## Missing or conflicted data")
    if final.missing_or_conflicted_data:
        lines.extend(f"- {d}" for d in final.missing_or_conflicted_data)
    else:
        lines.append("- None.")
    if not final.valuation_scenarios and not final.important_levels:
        lines.append("- Insufficient data to reach an investment conclusion.")
    lines.append("")

    lines.append("## Audit appendix")
    lines.append(f"- Packet hash: `{final.audit.get('packet_hash', '')[:16]}…`")
    lines.append(f"- Formula versions: {final.audit.get('formula_versions')}")
    lines.append(f"- Overrides triggered: {[o.override_id for o in final.overrides]}")
    lines.append(f"- Contradictions: {final.contradictions}")
    lines.append("")

    return "\n".join(lines)


def render(final: FinalReport, charts: dict[str, Path], out_dir: Path) -> Path:
    """Write report.json and report.md into `out_dir`. Returns the .md path."""
    if final.profile.profile in _AVOID_PROFILES and not final.revisit_date_or_event:
        raise ValueError("Avoid classification requires a revisit date/event")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(final.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )

    md = _markdown(final)
    lowered = md.lower()
    for phrase in _FORBIDDEN:
        if phrase in lowered:
            raise ValueError(f"forbidden certainty language in report: {phrase!r}")

    md_path = out_dir / "report.md"
    md_path.write_text(md)
    return md_path
