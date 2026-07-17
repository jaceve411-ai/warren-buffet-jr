"""Tests for the financial specialist (Task 14), including the
02_financial_analysis/VALIDATION_TESTS.md rows FIN-T001..T010.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wbj.schemas.packet import Packet
from wbj.specialists import financial as fin
from wbj.specialists.common import SpecialistOutput, core_diagnostic

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "packet" / "NVDA_packet.json"


@pytest.fixture(scope="module")
def nvda_packet() -> Packet:
    return Packet.model_validate(json.loads(_FIXTURE.read_text()))


# --- VALIDATION_TESTS.md rows ------------------------------------------------


def test_FIN_T001_revenue_growth():
    assert fin.yoy_growth(110, 100) == pytest.approx(0.10)


def test_FIN_T002_current_ratio():
    assert fin.current_ratio(150, 100) == pytest.approx(1.5)


def test_FIN_T003_coverage_1_5_no_warning():
    cov = fin.interest_coverage(30, 20)
    assert cov == pytest.approx(1.5)
    assert fin.band_interest_coverage(cov) == "GOOD"  # >=1.5 not BAD


def test_FIN_T004_coverage_below_1_5_solvency_warning():
    cov = fin.interest_coverage(29, 20)
    assert cov == pytest.approx(1.45)
    assert fin.band_interest_coverage(cov) == "BAD"  # triggers solvency warning


def test_FIN_T005_fcf_and_margin():
    f = fin.free_cash_flow(120, 40)  # capex positive magnitude -> subtract
    assert f == pytest.approx(80)
    assert fin.fcf_margin(f, 800) == pytest.approx(0.10)


def test_FIN_T006_loss_negfcf_override(nvda_packet):
    # Build a packet where latest annual has a loss and negative FCF.
    data = json.loads(_FIXTURE.read_text())
    latest = data["fundamentals"]["annual"][0]
    latest["net_income"] = -10.0
    latest["operating_cash_flow"] = -20.0
    latest["capex"] = -5.0  # FCF = -25
    packet = Packet.model_validate(data)
    out = fin.run(packet)
    assert "OVERRIDE_1_CAPITAL_DEPENDENCE" in out.mandatory_overrides


def test_FIN_T007_roic_below_wacc_not_excellent():
    assert fin.band_roic_vs_wacc(0.09, 0.11) == "BAD"


def test_FIN_T008_core27_all_excellent_is_100():
    core = core_diagnostic(["EXCELLENT"] * 27)
    assert core["percent"] == pytest.approx(100.0)
    assert core["maximum_valid_points"] == 54


def test_FIN_T009_negative_equity_not_meaningful():
    assert fin.debt_to_equity(100, -50).is_null


def test_FIN_T010_bank_adapter_suppresses_conventional():
    assert fin.conventional_metrics_applicable("bank") is False
    assert fin.conventional_metrics_applicable("Technology") is True


# --- band edge conventions ---------------------------------------------------


def test_revenue_band_10pct_is_good_not_excellent():
    assert fin.band_revenue_growth(0.10) == "GOOD"
    assert fin.band_revenue_growth(0.1001) == "EXCELLENT"


# --- envelope / run on fixture ----------------------------------------------


def test_run_produces_schema_valid_output(nvda_packet):
    out = fin.run(nvda_packet)
    assert isinstance(out, SpecialistOutput)
    assert out.agent_id == "financial_analysis"
    assert out.version == "2.0.0"
    assert out.category.max_points == 15.0


def test_category_points_reconcile_with_dimensions(nvda_packet):
    out = fin.run(nvda_packet)
    dim_sum = sum(d.awarded_points for d in out.dimensions)
    assert out.category.awarded_points == pytest.approx(dim_sum, abs=1e-6)


def test_coverage_between_0_and_1(nvda_packet):
    out = fin.run(nvda_packet)
    assert 0.0 <= out.coverage <= 1.0
