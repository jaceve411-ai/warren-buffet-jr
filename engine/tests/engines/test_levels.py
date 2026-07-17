"""Tests for wbj.engines.levels_engine (Task 12).

Synthetic OHLCV constructed so pivots, zones, touches, strength, breakouts
and gaps have known answers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wbj.engines.levels_engine import (
    breakout_confirmed,
    classify,
    cluster_zones,
    compute_levels,
    earnings_gaps,
    find_pivots,
    strength,
    zone_tolerance,
)
from wbj.schemas.levels import Touch, Zone

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "packet" / "NVDA_packet.json"


def _zone(ztype="resistance", **kw) -> Zone:
    base = dict(zone_id="z", type=ztype, lower=99.0, center=100.0, upper=101.0)
    base.update(kw)
    return Zone(**base)


# --- pivots ------------------------------------------------------------------


def test_symmetric_pivot_k3_detects_local_max():
    highs = [10, 11, 12, 13, 14, 15, 16, 20, 16, 15, 14, 13, 12, 11, 10]
    lows = [h - 1 for h in highs]
    df = pd.DataFrame({"high": highs, "low": lows, "close": highs, "open": highs, "volume": [1] * 15})
    pivots = find_pivots(df, k=3)
    highs_at = [p["index"] for p in pivots if p["kind"] == "high"]
    assert 7 in highs_at  # the value 20 at index 7 is the local max


def test_last_k_bars_never_pivot():
    highs = list(range(20))
    df = pd.DataFrame({"high": highs, "low": highs, "close": highs, "open": highs, "volume": [1] * 20})
    pivots = find_pivots(df, k=3)
    assert all(p["index"] < 17 for p in pivots)


# --- tolerance / zones -------------------------------------------------------


def test_zone_tolerance_formula():
    # atr=2, price=100 -> max(0.5*2, 0.0075*100) = max(1.0, 0.75) = 1.0
    assert zone_tolerance(2.0, 100.0) == pytest.approx(1.0)


def test_cluster_groups_nearby_highs():
    pivots = [
        {"index": 10, "price": 100.0, "kind": "high"},
        {"index": 20, "price": 100.5, "kind": "high"},
        {"index": 30, "price": 130.0, "kind": "high"},
    ]
    zones = cluster_zones(pivots, atr=2.0, price=120.0, last_index=40)
    # 100 and 100.5 cluster (within tolerance); 130 is separate.
    centers = sorted(round(z.center, 1) for z in zones)
    assert len(zones) == 2
    assert centers[1] == pytest.approx(130.0)


# --- strength / classify -----------------------------------------------------


def test_strength_formula_exact():
    touches = [
        Touch(date="a", pivot_price=100.0, rejection_atr=1.0, volume_ratio=1.5, age_sessions=0),
        Touch(date="b", pivot_price=100.0, rejection_atr=1.0, volume_ratio=1.5, age_sessions=0),
    ]
    z = _zone(touches=touches, timeframe="daily", confluence_count=0)
    # 30*min(2/4,1)=15 + 20*min(1/2,1)=10 + 15*min(1.5/1.5,1)=15
    # + 15*exp(0)=15 + 5 (daily) + 0 = 60
    assert strength(z) == pytest.approx(60.0)


def test_classify_candidate_confirmed_strong():
    t = lambda age, rej=1.0, vr=1.5: Touch(
        date="d", pivot_price=100.0, rejection_atr=rej, volume_ratio=vr, age_sessions=age
    )
    assert classify(_zone(touches=[t(0)])) == "candidate"
    assert classify(_zone(touches=[t(0), t(6, rej=0.6, vr=1.0)])) == "confirmed"
    assert classify(_zone(touches=[t(0), t(6), t(12)])) == "strong"


def test_classify_preserves_broken():
    assert classify(_zone(status="broken")) == "broken"


# --- breakouts ---------------------------------------------------------------


def test_breakout_requires_volume_and_close():
    n = 60
    close = [100.0] * n
    vol = [1_000_000.0] * n
    # Break at index 55: two closes above buffer with high volume.
    close[55] = 110.0
    close[56] = 111.0
    vol[55] = 2_000_000.0
    df = pd.DataFrame(
        {"high": close, "low": close, "close": close, "open": close, "volume": vol}
    )
    zone = _zone(ztype="resistance", lower=99.0, center=100.0, upper=101.0)
    assert breakout_confirmed(df, zone, atr=2.0) is True

    # Same price break but no volume expansion -> not confirmed.
    df_lowvol = df.copy()
    df_lowvol.loc[55, "volume"] = 1_000_000.0
    df_lowvol.loc[56, "volume"] = 1_000_000.0
    assert breakout_confirmed(df_lowvol, zone, atr=2.0) is False


# --- gaps --------------------------------------------------------------------


def test_gap_material_threshold():
    # atr=1, prior close=100: threshold = max(1.0, 0.03*100)=3.0
    close = [100.0] * 10
    open_ = [100.0] * 10
    open_[5] = 102.0  # 2% gap -> 2.0 < 3.0 not material
    df = pd.DataFrame(
        {"high": close, "low": close, "close": close, "open": open_, "volume": [1] * 10}
    )
    assert earnings_gaps(df, [5], atr=1.0)[0].material is False

    open_[5] = 103.1  # 3.1% gap -> 3.1 >= 3.0 material
    df2 = pd.DataFrame(
        {"high": close, "low": close, "close": close, "open": open_, "volume": [1] * 10}
    )
    assert earnings_gaps(df2, [5], atr=1.0)[0].material is True


# --- orchestration on fixture -----------------------------------------------


def test_compute_levels_on_fixture():
    daily = json.loads(_FIXTURE.read_text())["market_data"]["daily"]
    df = pd.DataFrame(daily).iloc[::-1].reset_index(drop=True)
    out = compute_levels(df)
    assert len(out.support) <= 3 and len(out.resistance) <= 3
    assert "sma200" in out.moving_averages
    assert np.isfinite(out.avwaps["anchor_start"])
