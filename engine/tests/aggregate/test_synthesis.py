"""Tests for price synthesis, overrides and contradictions (Task 21)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wbj.aggregate.contradiction import contradictions
from wbj.aggregate.overrides import apply_overrides
from wbj.aggregate.synthesis import confluence_tolerance, synthesize_levels
from wbj.schemas.packet import Packet
from wbj.specialists import business, financial, market, risk, technical, valuation

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "packet" / "NVDA_packet.json"


class _Tech:
    important_levels = {
        "support": [{"center": 95.0, "status": "confirmed", "strength_0_100": 60}],
        "resistance": [{"center": 130.0, "status": "strong", "strength_0_100": 80}],
    }


class _Val:
    scenarios_detail = [
        {"name": "base", "value": 130.4},  # within tolerance of the 130 resistance
        {"name": "bear", "value": 80.0},
    ]


def test_confluence_tolerance_formula():
    # atr=2, price=100 -> max(1.0, 0.75) = 1.0
    assert confluence_tolerance(2.0, 100.0) == pytest.approx(1.0)


def test_synthesis_never_averages_technical_and_intrinsic():
    levels = synthesize_levels(_Tech(), _Val(), price=120.0, atr=4.0)
    prices = [round(e["price"], 1) for e in levels]
    # Both the technical 130.0 and intrinsic 130.4 survive as-is; no 130.2 average.
    assert 130.0 in prices and 130.4 in prices
    assert 130.2 not in prices


def test_confluence_flagged_within_tolerance():
    levels = synthesize_levels(_Tech(), _Val(), price=120.0, atr=4.0)
    intrinsic_base = next(e for e in levels if e["kind"] == "scenario_base")
    assert intrinsic_base["confluence"] is True  # 130.4 within tol of 130.0


def test_contradiction_strong_business_weak_technical():
    notes = contradictions({"business": 18, "financial": 12, "market": 12,
                            "technical": 5, "risk": 10, "valuation": 6})
    assert any("Quality watch" in n for n in notes)


def test_overrides_on_fixture():
    packet = Packet.model_validate(json.loads(_FIXTURE.read_text()))
    outputs = {
        "business": business.run(packet), "financial": financial.run(packet),
        "market": market.run(packet), "technical": technical.run(packet),
        "risk": risk.run(packet), "valuation": valuation.run(packet),
    }
    ovr = apply_overrides(outputs, packet)
    # Several categories have low coverage on this packet -> gate-ineligible.
    assert any(o.action == "GATE_INELIGIBLE" for o in ovr)


def test_build_final_report_on_fixture():
    from wbj.aggregate.assemble import build_final_report

    packet = Packet.model_validate(json.loads(_FIXTURE.read_text()))
    outputs = {
        "business": business.run(packet), "financial": financial.run(packet),
        "market": market.run(packet), "technical": technical.run(packet),
        "risk": risk.run(packet), "valuation": valuation.run(packet),
    }
    report = build_final_report(outputs, packet)
    assert report.report_version == "2.0.0"
    assert 0.0 <= report.profile.raw_score <= 100.0
    assert len(report.executive_thesis) == 7
    assert len(report.category_scorecard) == 6
    assert "packet_hash" in report.audit
