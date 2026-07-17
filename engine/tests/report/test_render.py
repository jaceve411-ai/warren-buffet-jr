"""Tests for the report renderer (Task 23)."""

from __future__ import annotations

import json

import pytest

from wbj.report.render import render
from wbj.schemas.final_report import CategoryScorecardRow, FinalReport, ProfileResult


def _minimal_report(profile="Conditional/Watch", revisit=None, insiders=None) -> FinalReport:
    return FinalReport(
        security={"ticker": "TST", "exchange": "NASDAQ", "currency": "USD"},
        knowledge_timestamp="2026-07-16T21:00:00+00:00",
        profile=ProfileResult(profile=profile, raw_score=65.0, band="Mixed"),
        category_scorecard=[
            CategoryScorecardRow(category="business", max_points=20, awarded_points=12,
                                 score_10=6, confidence=70, coverage=0.8)
        ],
        executive_thesis=[f"sentence {i}" for i in range(7)],
        valuation_scenarios=[{"name": "base", "value": 130.0, "probability": 0.5}],
        important_levels=[{"source": "technical", "kind": "support", "price": 100.0,
                           "distance_percent": -5.0, "distance_atr": -1.0, "confluence": False}],
        insider_activity=insiders or [],
        revisit_date_or_event=revisit,
    )


def test_all_sections_present(tmp_path):
    md = render(_minimal_report(), {}, tmp_path).read_text()
    for header in ["Executive summary", "Research classification", "Category scorecard",
                   "Price scenario ranges", "Important levels", "Insider activity",
                   "Profile fit", "Missing or conflicted data", "Audit appendix"]:
        assert header in md


def test_avoid_without_revisit_raises(tmp_path):
    with pytest.raises(ValueError, match="revisit"):
        render(_minimal_report(profile="Avoid/Wait", revisit=None), {}, tmp_path)


def test_avoid_with_revisit_ok(tmp_path):
    md_path = render(_minimal_report(profile="Avoid/Wait", revisit="Next earnings"), {}, tmp_path)
    assert "Revisit" in md_path.read_text()


def test_insider_filter_1m_threshold(tmp_path):
    insiders = [
        {"name": "small", "total_usd": 999_999.0, "significant": False},
        {"name": "big", "total_usd": 1_000_001.0, "significant": True},
    ]
    md = render(_minimal_report(insiders=insiders), {}, tmp_path).read_text()
    assert "big" in md
    assert "small" not in md.split("## Insider activity")[1].split("##")[0]


def test_report_json_roundtrips(tmp_path):
    render(_minimal_report(), {}, tmp_path)
    data = json.loads((tmp_path / "report.json").read_text())
    restored = FinalReport.model_validate(data)
    assert restored.report_version == "2.0.0"


def test_no_forbidden_language(tmp_path):
    md = render(_minimal_report(), {}, tmp_path).read_text().lower()
    for phrase in ("guaranteed target", "must hold", "certain floor"):
        assert phrase not in md
