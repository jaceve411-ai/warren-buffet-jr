"""Tests for wbj.engines.valuation_engine (Task 13).

Closed-form cases where the answer is known analytically, plus the
economic-profit/FCFF reconciliation identity and reverse-DCF round trip.
"""

from __future__ import annotations

import pytest

from wbj.core.nullstates import NullState
from wbj.engines.valuation_engine import (
    dcf_value,
    economic_profit_value,
    ensemble,
    equity_bridge,
    fcff,
    gordon_terminal_value,
    hist_zscore,
    justified_pe,
    margin_of_safety,
    monte_carlo,
    per_share,
    relever_beta,
    reverse_dcf,
    scenarios,
    synthetic_kd,
    unlever_beta,
    wacc,
)


# --- terminal value / DCF ----------------------------------------------------


def test_gordon_terminal_math():
    # FCFF_N=100 growing 2%, WACC 10% -> 100*1.02/0.08 = 1275
    assert gordon_terminal_value(100.0, 0.02, 0.10) == pytest.approx(1275.0)


def test_g_greater_than_wacc_refused():
    result = dcf_value([100.0, 100.0], wacc_value=0.08, terminal_growth=0.10)
    assert result.ev.is_null
    assert result.ev.state == NullState.NOT_MEANINGFUL


def test_terminal_share_warning_above_75pct():
    # Small explicit FCFFs, large terminal -> terminal dominates EV.
    result = dcf_value([1.0], wacc_value=0.10, terminal_growth=0.02)
    assert result.terminal_share > 0.75
    assert any("terminal value" in w for w in result.warnings)


def test_dcf_value_matches_hand_computation():
    result = dcf_value([100.0], wacc_value=0.10, terminal_growth=0.0)
    # pv_explicit = 100/1.1; tv = 100/0.10 = 1000; pv_terminal = 1000/1.1
    assert result.pv_explicit == pytest.approx(100 / 1.1)
    assert result.pv_terminal == pytest.approx(1000 / 1.1)
    assert result.ev.value == pytest.approx(1000.0)


# --- WACC / beta -------------------------------------------------------------


def test_wacc():
    # E=800, D=200, Ke=10%, Kd=5%, tax=25% -> 0.8*.10 + 0.2*.05*.75 = 8.75%
    assert wacc(800, 200, 0.10, 0.05, 0.25) == pytest.approx(0.0875)


def test_unlever_relever_beta_roundtrip():
    bu = unlever_beta(1.5, tax_rate=0.25, de=0.5)
    bl = relever_beta(bu, tax_rate=0.25, target_de=0.5)
    assert bl == pytest.approx(1.5)


def test_synthetic_kd_low_coverage_high_spread():
    # coverage 1.0 -> CCC band spread 0.0455 on top of rf
    assert synthetic_kd(0.04, 1.0) == pytest.approx(0.04 + 0.0455)
    # coverage 10 -> AAA band 0.0069
    assert synthetic_kd(0.04, 10.0) == pytest.approx(0.04 + 0.0069)


# --- equity bridge / per share ----------------------------------------------


def test_equity_bridge_and_per_share():
    eq = equity_bridge(ev=1000.0, cash=50.0, nonop=10.0, debt=200.0, lease_debt_value=30.0)
    assert eq == pytest.approx(830.0)
    assert per_share(eq, 100.0).value == pytest.approx(8.30)


def test_per_share_zero_shares_not_meaningful():
    assert per_share(1000.0, 0.0).is_null


# --- reverse DCF -------------------------------------------------------------


def test_reverse_dcf_recovers_known_growth():
    # A monotonic per-share model in growth; price set at g*=0.08.
    def per_share_of_growth(g: float) -> float:
        f0 = 100.0
        fcffs = [f0 * (1 + g) ** t for t in range(1, 6)]
        return dcf_value(fcffs, wacc_value=0.12, terminal_growth=0.02).ev.value / 10.0

    price = per_share_of_growth(0.08)
    implied = reverse_dcf(price, per_share_of_growth)
    assert implied.value == pytest.approx(0.08, abs=1e-4)


# --- scenarios / Monte Carlo -------------------------------------------------


def test_scenario_probabilities_must_sum_to_1():
    specs = [
        {"name": "bear", "probability": 0.3, "v": 50.0},
        {"name": "base", "probability": 0.5, "v": 100.0},
        {"name": "bull", "probability": 0.3, "v": 150.0},  # sums to 1.1
    ]
    with pytest.raises(ValueError, match="sum to 1"):
        scenarios(specs, lambda s: s["v"])


def test_scenario_weighted_value():
    specs = [
        {"name": "bear", "probability": 0.25, "v": 50.0},
        {"name": "base", "probability": 0.50, "v": 100.0},
        {"name": "bull", "probability": 0.25, "v": 150.0},
    ]
    result = scenarios(specs, lambda s: s["v"])
    assert result.weighted == pytest.approx(100.0)


def test_monte_carlo_deterministic_given_seed():
    params = {"growth": (0.0, 0.05, 0.10), "margin": (0.1, 0.2, 0.3), "wacc": (0.08, 0.10, 0.12)}
    fn = lambda g, m, w: 1000.0 * (1 + g) * m / w
    a = monte_carlo(fn, params, n=500, seed=42)
    b = monte_carlo(fn, params, n=500, seed=42)
    assert a.median == b.median and a.p10 == b.p10 and a.p90 == b.p90
    c = monte_carlo(fn, params, n=500, seed=43)
    assert c.median != a.median


# --- economic profit reconciliation -----------------------------------------


def test_economic_profit_reconciles_with_fcff():
    # No-growth perpetuity: NOPAT=100, WACC=10%, IC0=500.
    # FCFF DCF EV = 100/0.10 = 1000; EP = 100 - 0.10*500 = 50 -> 500 + 50/0.10 = 1000.
    ev_fcff = dcf_value([100.0], wacc_value=0.10, terminal_growth=0.0).ev.value
    ev_ep = economic_profit_value(ic0=500.0, economic_profits=[50.0], wacc_value=0.10)
    assert abs(ev_ep - ev_fcff) / ev_fcff < 0.01


# --- misc cross-checks -------------------------------------------------------


def test_justified_pe_refuses_nonpositive_roe():
    assert justified_pe(g=0.03, roe=-0.1, ke=0.10).is_null


def test_hist_zscore_robust():
    z = hist_zscore(20.0, [10.0, 11.0, 9.0, 10.0, 200.0])  # outlier ignored via MAD
    assert z.is_valid


def test_margin_of_safety_sign():
    assert margin_of_safety(100.0, 80.0) == pytest.approx(0.2)
    assert margin_of_safety(100.0, 120.0) == pytest.approx(-0.2)


def test_ensemble_weighted_value():
    result = ensemble([(100.0, 0.5), (120.0, 0.5)])
    assert result.value == pytest.approx(110.0)
    assert result.dispersion == pytest.approx(10.0)


def test_fcff_formula():
    # EBIT=200, tax=25%, D&A=30, capex=40, dNWC=10 -> 150+30-40-10 = 130
    assert fcff(200.0, 0.25, 30.0, 40.0, 10.0) == pytest.approx(130.0)
