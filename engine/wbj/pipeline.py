"""Staged analysis pipeline (Task 24).

Wires the full v2.0.0 pipeline: packet -> 6 specialists (frozen) -> judgment
overlay -> aggregation/gates -> charts + report. Kept separate from the MVP
commands in `cli.py` (see RESUME.md Task-24 note); the MVP stays available as
`wbj analyze`, the full pipeline is `wbj full`.

`offline=True` loads the committed golden NVDA packet instead of hitting the
network, so the end-to-end path is testable without any API access.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from wbj.aggregate.assemble import build_final_report
from wbj.overlay.merge import merge_overlay
from wbj.report import charts as chartmod
from wbj.report.render import render
from wbj.schemas.final_report import FinalReport
from wbj.schemas.overlay import Judgment
from wbj.schemas.packet import Packet
from wbj.specialists import business, financial, market, risk, technical, valuation
from wbj.specialists.common import SpecialistOutput, compute_output_hash

_ENGINE_DIR = Path(__file__).resolve().parent.parent
_GOLDEN_PACKET = _ENGINE_DIR / "tests" / "fixtures" / "packet" / "NVDA_packet.json"

_SPECIALISTS = {
    "business": business,
    "financial": financial,
    "market": market,
    "technical": technical,
    "risk": risk,
    "valuation": valuation,
}


def stage_packet(ticker: str, settings, offline: bool = False, packet: Packet | None = None) -> Packet:
    """Build the analysis packet. Offline mode loads the committed golden
    packet (NVDA); a live build is out of scope for offline tests."""
    if packet is not None:
        return packet
    if offline:
        cached = Path(settings.cache_dir) / ticker.upper() / "packet.json"
        source = cached if cached.exists() else _GOLDEN_PACKET
        return Packet.model_validate(json.loads(source.read_text()))
    raise NotImplementedError("Live packet build requires network providers; use offline=True.")


def stage_compute(packet: Packet, artifacts_dir: Path | None = None) -> dict[str, SpecialistOutput]:
    """Run the six specialists independently and freeze their outputs."""
    outputs: dict[str, SpecialistOutput] = {}
    for name, module in _SPECIALISTS.items():
        out = module.run(packet)
        out.output_hash = compute_output_hash(out)
        outputs[name] = out

    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        for name, out in outputs.items():
            (artifacts_dir / f"{name}.json").write_text(
                json.dumps(out.model_dump(mode="json"), indent=2, sort_keys=True)
            )
        requests = [r.model_dump() for out in outputs.values() for r in out.judgment_requests]
        (artifacts_dir / "judgment_requests.json").write_text(json.dumps(requests, indent=2))
    return outputs


def stage_aggregate(
    outputs: dict[str, SpecialistOutput], packet: Packet, overlay_path: Path | None = None
) -> FinalReport:
    """Apply overlay (if any), overrides, gates and synthesis into a report."""
    if overlay_path is not None and Path(overlay_path).exists():
        raw = json.loads(Path(overlay_path).read_text())
        judgments = [Judgment.model_validate(j) for j in raw]
        merged = merge_overlay(list(outputs.values()), judgments)
        outputs = dict(zip(outputs.keys(), merged))
    return build_final_report(outputs, packet)


def stage_report(final: FinalReport, packet: Packet, out_dir: Path) -> Path:
    """Render charts + report.md/json into `out_dir/charts` and `out_dir`."""
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    chart_paths: dict[str, Path] = {}

    # Scorecard chart.
    pts = {r.category: r.awarded_points for r in final.category_scorecard}
    mx = {r.category: r.max_points for r in final.category_scorecard}
    if pts:
        chart_paths["scorecard"] = chartmod.scorecard_chart(pts, mx, charts_dir / "scorecard.png")

    # Football field from valuation scenarios.
    scenario_vals = [s.get("value") for s in final.valuation_scenarios if s.get("value") is not None]
    if scenario_vals:
        bands = [{"name": "scenarios", "low": min(scenario_vals), "high": max(scenario_vals)}]
        price = 0.0
        pv = packet.facts_table.get("price")
        if pv and pv.is_valid:
            price = pv.value
        chart_paths["football"] = chartmod.football_field_chart(bands, price, charts_dir / "football.png")

    # Price levels from packet daily.
    daily = [row.model_dump() for row in packet.market_data.daily]
    if daily:
        df = pd.DataFrame(daily).iloc[::-1].reset_index(drop=True)
        levels = [{"type": lv.get("kind"), "lower": lv.get("price", 0) * 0.99, "upper": lv.get("price", 0) * 1.01}
                  for lv in final.important_levels]
        chart_paths["price_levels"] = chartmod.price_levels_chart(df, levels, {}, charts_dir / "price_levels.png")

    return render(final, chart_paths, out_dir)


def run_all(
    ticker: str, settings, overlay_path: Path | None = None, offline: bool = False,
    packet: Packet | None = None,
) -> FinalReport:
    """Run every stage and write the report under Reportes/<T>/<date>/."""
    pkt = stage_packet(ticker, settings, offline=offline, packet=packet)
    out_dir = Path(settings.reports_dir) / ticker.upper() / date.today().isoformat()
    artifacts = Path(settings.cache_dir) / ticker.upper() / "artifacts"
    outputs = stage_compute(pkt, artifacts)
    final = stage_aggregate(outputs, pkt, overlay_path)
    stage_report(final, pkt, out_dir)
    return final
