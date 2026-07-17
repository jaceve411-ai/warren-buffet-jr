"""Tests for aggregation gates/overrides (Task 21), including MAIN rows."""

from __future__ import annotations

import pytest

from wbj.aggregate.gates import apply_gates, descriptive_band, evaluate_momentum, raw_total
from wbj.schemas.final_report import Override


def _cats(business=16, financial=11, market=16, technical=17, risk=9, valuation=7):
    return {
        "business": business, "financial": financial, "market": market,
        "technical": technical, "risk": risk, "valuation": valuation,
    }


def test_MAIN_002_raw_total():
    assert raw_total([16, 10.5, 18, 16, 9, 7]) == 76.5


def test_MAIN_003_risk_cap_forces_speculative():
    cats = _cats(business=18, financial=15, market=20, technical=20, risk=4, valuation=10)  # total 87
    res = apply_gates(cats, {"technical": 90}, {k: 1.0 for k in cats})
    assert res.profile == "Speculative"


def test_MAIN_007_low_coverage_blocks_gates():
    cats = _cats(business=18, financial=14, market=18, technical=18, risk=12, valuation=8)
    coverages = {k: 1.0 for k in cats}
    coverages["market"] = 0.65
    res = apply_gates(cats, {"technical": 90}, coverages)
    assert res.profile not in ("Momentum", "Quality", "Value")
    assert any("coverage<0.70" in r for r in res.reasons)


def test_momentum_gate_exact_thresholds_pass():
    # raw 78, tech 17, market 16, bus+fin 28, risk 8, tech_conf 70 -> passes
    cats = _cats(business=17, financial=11, market=16, technical=17, risk=8, valuation=9)
    assert sum(cats.values()) == 78
    passed, reasons = evaluate_momentum(cats, {"technical": 70})
    assert passed and reasons == []
    res = apply_gates(cats, {"technical": 70}, {k: 1.0 for k in cats})
    assert res.profile == "Momentum"


def test_momentum_gate_fails_below_78():
    cats = _cats(business=16, financial=11, market=16, technical=17, risk=8, valuation=9)  # 77
    passed, reasons = evaluate_momentum(cats, {"technical": 70})
    assert not passed
    assert "raw_total<78" in reasons


def test_avoid_when_override_demands():
    cats = _cats()
    ovr = [Override(override_id="OVR-5", condition="x", action="WAIT_AVOID")]
    res = apply_gates(cats, {"technical": 90}, {k: 1.0 for k in cats}, overrides=ovr)
    assert res.profile == "Avoid/Wait"


def test_descriptive_bands():
    assert descriptive_band(92) == "Elite"
    assert descriptive_band(80) == "Strong"
    assert descriptive_band(49) == "Avoid"
