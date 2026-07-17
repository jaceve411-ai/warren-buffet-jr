"""Tests for business/market/technical/risk/valuation specialists (Tasks 15-19).

Each specialist must: run schema-valid on the NVDA fixture, reconcile
category points with its dimensions (±1e-6), keep coverage in [0,1], and
honor the specific quantitative rules the plan calls out.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wbj.schemas.packet import Packet
from wbj.specialists import business, market, risk, technical, valuation
from wbj.specialists.common import SpecialistOutput

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "packet" / "NVDA_packet.json"


@pytest.fixture(scope="module")
def nvda_packet() -> Packet:
    return Packet.model_validate(json.loads(_FIXTURE.read_text()))


ALL_SPECIALISTS = [
    (business, 20.0),
    (market, 20.0),
    (technical, 20.0),
    (risk, 15.0),
    (valuation, 10.0),
]


@pytest.mark.parametrize("module,max_points", ALL_SPECIALISTS)
def test_run_schema_valid(nvda_packet, module, max_points):
    out = module.run(nvda_packet)
    assert isinstance(out, SpecialistOutput)
    assert out.version == "2.0.0"
    assert out.category.max_points == max_points


@pytest.mark.parametrize("module,_max", ALL_SPECIALISTS)
def test_category_reconciles_with_dimensions(nvda_packet, module, _max):
    out = module.run(nvda_packet)
    dim_sum = sum(d.awarded_points for d in out.dimensions)
    assert out.category.awarded_points == pytest.approx(dim_sum, abs=1e-6)
    assert out.category.score_10 == pytest.approx(10.0 * dim_sum / out.category.max_points, abs=1e-6)


@pytest.mark.parametrize("module,_max", ALL_SPECIALISTS)
def test_coverage_in_range(nvda_packet, module, _max):
    out = module.run(nvda_packet)
    assert 0.0 <= out.coverage <= 1.0


# --- Market: TimeDecay and >=5 estimate rule --------------------------------


def test_market_time_decay_12_months_is_half():
    assert market.time_decay(12.0) == pytest.approx(0.5)


def test_market_revision_breadth_needs_5_estimates():
    assert market.revision_breadth([{"revision": 1}] * 4).is_null  # <5
    ok = market.revision_breadth([{"revision": 1}, {"revision": -1}, {"revision": 1}, {"revision": 1}, {"revision": 0}])
    assert ok.is_valid and ok.value == pytest.approx(3 / 5)


# --- Technical: trend anchors ------------------------------------------------


def test_technical_trend_anchor_strong_uptrend():
    # ADX>=25, pos52w>=0.80, close>sma50>sma200 -> 9
    assert technical.trend_anchor_score(110, 105, 100, 0.5, 30, 0.9) == 9.0


def test_technical_trend_anchor_downtrend():
    # close<sma50<sma200 and slope < -1 ATR -> 1
    assert technical.trend_anchor_score(90, 95, 100, -2.0, 15, 0.2) == 1.0


# --- Risk: Beneish / Altman / VaR closed forms ------------------------------


def test_beneish_m_score_neutral_inputs():
    # All ratios = 1 (no change) -> deterministic constant.
    m = risk.beneish_m_score(1, 1, 1, 1, 1, 1, 0, 1)
    expected = -4.84 + 0.920 + 0.528 + 0.404 + 0.892 + 0.115 - 0.172 + 0.0 - 0.327
    assert m == pytest.approx(expected)


def test_altman_z_double_prime():
    # WC/TA=0.2, RE/TA=0.3, EBIT/TA=0.1, BE/TL=1.0
    z = risk.altman_z_double_prime(wc=20, re=30, ebit=10, book_equity=100, ta=100, tl=100)
    assert z == pytest.approx(6.56 * 0.2 + 3.26 * 0.3 + 6.72 * 0.1 + 1.05 * 1.0)


def test_piotroski_counts_true_signals():
    assert risk.piotroski_f({"a": True, "b": False, "c": True}) == 2


def test_historical_var_on_synthetic_returns():
    returns = np.linspace(-0.05, 0.05, 101)  # symmetric
    var = risk.historical_var(returns, 0.95)
    assert var == pytest.approx(0.045, abs=1e-3)


def test_max_drawdown():
    prices = np.array([100.0, 120.0, 60.0, 80.0])  # peak 120 -> trough 60 = -50%
    assert risk.max_drawdown(prices) == pytest.approx(-0.5)


# --- Valuation: scenarios sum to 1, reverse DCF present ---------------------


def test_valuation_scenarios_and_reverse_dcf(nvda_packet):
    out = valuation.run(nvda_packet)
    probs = [s["probability"] for s in out.scenarios_detail]
    if probs:
        assert sum(probs) == pytest.approx(1.0)
    assert "implied_growth" in out.reverse_dcf or out.reverse_dcf == {}


def test_business_moat_capped_without_positive_spread(nvda_packet):
    out = business.run(nvda_packet)
    moat_dim = next(d for d in out.dimensions if d.name == "moat")
    assert moat_dim.score_10 <= 6.0  # cap until positive ROIC-WACC spread confirmed
