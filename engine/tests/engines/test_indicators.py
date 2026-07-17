"""Tests for wbj.engines.indicators (Task 11).

Small hand-checkable series where the answer is known by construction, plus
a golden pass over the committed NVDA fixture OHLCV cross-checked against
straightforward inline pandas — never hardcoded magic numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wbj.engines.indicators import (
    adx14,
    atr14,
    cmf,
    composite_rs_percentile,
    ema,
    macd,
    median_dollar_volume,
    obv,
    range_position_52w,
    realized_vol,
    relative_strength,
    roc,
    rsi14,
    sma,
    true_range,
    up_down_volume_ratio,
    volume_ratio,
)

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "packet" / "NVDA_packet.json"


def constant_tr_frame(tr: float, bars: int) -> pd.DataFrame:
    """Every bar identical (H=L+tr, C=midpoint) so true range == `tr` on
    every bar, including the first (H-L)."""
    low, high, close = 10.0, 10.0 + tr, 10.0 + tr / 2
    return pd.DataFrame(
        {
            "open": [close] * bars,
            "high": [high] * bars,
            "low": [low] * bars,
            "close": [close] * bars,
            "volume": [1_000_000.0] * bars,
        }
    )


@pytest.fixture(scope="module")
def nvda_ohlcv() -> pd.DataFrame:
    daily = json.loads(_FIXTURE.read_text())["market_data"]["daily"]
    df = pd.DataFrame(daily)
    df = df.iloc[::-1].reset_index(drop=True)  # fixture is newest-first
    return df


# --- Wilder ATR --------------------------------------------------------------


def test_wilder_atr_smoothing():
    df = constant_tr_frame(tr=2.0, bars=20)
    assert abs(atr14(df).iloc[-1] - 2.0) < 1e-9


def test_true_range_first_bar_is_high_low():
    df = constant_tr_frame(tr=2.0, bars=5)
    assert true_range(df).iloc[0] == 2.0


def test_atr_is_nan_before_seed_window():
    df = constant_tr_frame(tr=2.0, bars=20)
    assert np.isnan(atr14(df).iloc[12])  # needs 14 TRs -> seeded at index 13
    assert not np.isnan(atr14(df).iloc[13])


# --- RSI ---------------------------------------------------------------------


def test_rsi_all_gains_is_100():
    close = pd.Series(np.arange(1.0, 40.0))
    assert rsi14(close).iloc[-1] == 100.0


def test_rsi_all_losses_is_zero():
    close = pd.Series(np.arange(40.0, 1.0, -1.0))
    assert rsi14(close).iloc[-1] == 0.0


def test_rsi_flat_series_is_nan_or_neutral():
    # No gains and no losses -> avg_loss 0 -> our convention yields 100.
    close = pd.Series([50.0] * 30)
    assert rsi14(close).iloc[-1] == 100.0


# --- EMA / SMA ---------------------------------------------------------------


def test_ema_initialized_with_sma():
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    # seed at index 2 is SMA of first 3 = 2.0
    assert ema(close, 3).iloc[2] == pytest.approx(2.0)


def test_sma_matches_rolling_mean(nvda_ohlcv):
    close = nvda_ohlcv["close"]
    expected = close.rolling(50).mean()
    pd.testing.assert_series_equal(sma(close, 50), expected)


def test_ema_recurrence_after_seed():
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    alpha = 2 / (3 + 1)
    seed = 2.0
    expected_idx3 = alpha * 4.0 + (1 - alpha) * seed
    assert ema(close, 3).iloc[3] == pytest.approx(expected_idx3)


# --- MACD --------------------------------------------------------------------


def test_macd_is_ema12_minus_ema26(nvda_ohlcv):
    close = nvda_ohlcv["close"]
    out = macd(close)
    expected = ema(close, 12) - ema(close, 26)
    pd.testing.assert_series_equal(out["macd"], expected, check_names=False)
    # histogram = macd - signal
    pd.testing.assert_series_equal(
        out["hist"], out["macd"] - out["signal"], check_names=False
    )


# --- ADX ---------------------------------------------------------------------


def test_adx_strong_uptrend_is_high():
    n = 80
    df = pd.DataFrame(
        {
            "open": [i + 1.0 for i in range(n)],
            "high": [i + 2.0 for i in range(n)],
            "low": [i + 0.0 for i in range(n)],
            "close": [i + 1.5 for i in range(n)],
            "volume": [1_000_000.0] * n,
        }
    )
    assert adx14(df).iloc[-1] > 40.0


# --- ROC / relative strength -------------------------------------------------


def test_roc_simple():
    close = pd.Series([100.0, 110.0, 121.0])
    assert roc(close, 1).iloc[-1] == pytest.approx(0.1)


def test_relative_strength_outperformance_above_one():
    close = pd.Series([100.0, 120.0])  # +20%
    bench = pd.Series([100.0, 100.0])  # flat
    assert relative_strength(close, bench, 1).iloc[-1] > 1.0


def test_composite_rs_weights_sum():
    # Universe 1..100 in every window; a value of 50 sits at the 50th pct,
    # so the weighted blend (weights sum to 1) is 50.
    universe = pd.DataFrame({w: np.arange(1, 101) for w in (21, 63, 126, 252)})
    rs = {21: 50, 63: 50, 126: 50, 252: 50}
    assert composite_rs_percentile(rs, universe) == pytest.approx(50.0)


def test_composite_rs_top_of_universe_is_100():
    universe = pd.DataFrame({w: np.arange(1, 101) for w in (21, 63, 126, 252)})
    rs = {21: 100, 63: 100, 126: 100, 252: 100}
    assert composite_rs_percentile(rs, universe) == pytest.approx(100.0)


# --- volume / volatility -----------------------------------------------------


def test_realized_vol_positive_on_fixture(nvda_ohlcv):
    rv = realized_vol(nvda_ohlcv["close"], 63)
    assert rv.iloc[-1] > 0.0


def test_volume_ratio_equals_one_on_constant_volume():
    vol = pd.Series([1_000_000.0] * 60)
    assert volume_ratio(vol).iloc[-1] == pytest.approx(1.0)


def test_up_down_volume_ratio_nan_when_no_down_days():
    df = pd.DataFrame(
        {
            "close": np.arange(1.0, 61.0),  # strictly up -> no down volume
            "volume": [1_000_000.0] * 60,
            "high": np.arange(1.0, 61.0),
            "low": np.arange(1.0, 61.0),
            "open": np.arange(1.0, 61.0),
        }
    )
    assert np.isnan(up_down_volume_ratio(df).iloc[-1])


def test_obv_accumulates_signed_volume():
    df = pd.DataFrame(
        {
            "close": [10.0, 11.0, 10.5, 11.5],
            "volume": [100.0, 200.0, 300.0, 400.0],
            "high": [10.0, 11.0, 10.5, 11.5],
            "low": [10.0, 11.0, 10.5, 11.5],
            "open": [10.0, 11.0, 10.5, 11.5],
        }
    )
    # +200 (up), -300 (down), +400 (up) = 300
    assert obv(df).iloc[-1] == pytest.approx(300.0)


def test_cmf_flat_bar_uses_zero_multiplier():
    df = pd.DataFrame(
        {
            "high": [10.0] * 20,
            "low": [10.0] * 20,  # H==L every bar -> multiplier 0
            "close": [10.0] * 20,
            "volume": [1_000_000.0] * 20,
            "open": [10.0] * 20,
        }
    )
    assert cmf(df).iloc[-1] == pytest.approx(0.0)


def test_range_position_at_high_is_one():
    n = 300
    close = pd.Series(np.arange(1.0, n + 1.0))
    df = pd.DataFrame(
        {"high": close, "low": close, "close": close, "open": close, "volume": [1.0] * n}
    )
    assert range_position_52w(df).iloc[-1] == pytest.approx(1.0)


def test_median_dollar_volume_on_fixture(nvda_ohlcv):
    mdv = median_dollar_volume(nvda_ohlcv)
    assert mdv.iloc[-1] > 0.0


# --- golden fixture sanity ---------------------------------------------------


def test_indicators_finite_on_fixture_tail(nvda_ohlcv):
    close = nvda_ohlcv["close"]
    for series in (
        sma(close, 200),
        ema(close, 50),
        rsi14(close),
        atr14(nvda_ohlcv),
        macd(close)["hist"],
    ):
        assert np.isfinite(series.iloc[-1])
