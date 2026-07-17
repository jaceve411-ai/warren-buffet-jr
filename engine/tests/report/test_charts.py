"""Tests for report charts (Task 22)."""

from __future__ import annotations

import pandas as pd
import pytest

from wbj.report import charts


def test_scenario_fan_rejects_single_line(tmp_path):
    scenarios = [{"name": "base", "low": 100.0, "high": 100.0, "growth": 0.08, "margin": 0.3}]
    with pytest.raises(ValueError, match="single-line projection prohibited"):
        charts.scenario_fan_chart([90.0, 95.0, 100.0], scenarios, tmp_path / "fan.png")


def test_scenario_fan_creates_nonempty_file_with_assumptions(tmp_path):
    scenarios = [
        {"name": "bear", "low": 80.0, "high": 95.0, "growth": 0.02, "margin": 0.25, "color": "#d62728"},
        {"name": "bull", "low": 120.0, "high": 160.0, "growth": 0.15, "margin": 0.35, "color": "#2ca02c"},
    ]
    out = charts.scenario_fan_chart([90.0, 95.0, 100.0], scenarios, tmp_path / "fan.png")
    assert out.exists() and out.stat().st_size > 0


def test_scorecard_chart(tmp_path):
    pts = {"business": 12.0, "financial": 9.0}
    mx = {"business": 20.0, "financial": 15.0}
    out = charts.scorecard_chart(pts, mx, tmp_path / "sc.png")
    assert out.exists() and out.stat().st_size > 0


def test_price_levels_chart(tmp_path):
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0]})
    levels = [{"type": "support", "lower": 99.0, "upper": 100.0},
              {"type": "resistance", "lower": 103.0, "upper": 104.0}]
    out = charts.price_levels_chart(df, levels, {}, tmp_path / "levels.png")
    assert out.exists() and out.stat().st_size > 0


def test_football_field_chart(tmp_path):
    bands = [{"name": "DCF", "low": 90.0, "high": 130.0}, {"name": "EP", "low": 95.0, "high": 125.0}]
    out = charts.football_field_chart(bands, current_price=110.0, out_path=tmp_path / "ff.png")
    assert out.exists() and out.stat().st_size > 0
