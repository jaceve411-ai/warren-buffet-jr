"""Task 12 — important-levels engine.

Deterministic support/resistance detection per `Cerebro/special_sauces/
IMPORTANT_LEVELS_ENGINE.md` and TECH-022…040. It never draws a level because
it "looks important": pivots are symmetric and confirmation-delayed, zones
are clustered by an ATR/price tolerance, and strength/status come from
explicit formulas over independent touches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wbj.engines.indicators import atr14, sma
from wbj.schemas.levels import Gap, LevelsOutput, Touch, Zone

_LN2 = np.log(2.0)
_HALFLIFE = 126.0


# --- swing detection --------------------------------------------------------


def find_pivots(df: pd.DataFrame, k: int) -> list[dict]:
    """TECH-PIV-022 — symmetric pivots. `High_t` is a pivot high iff it is the
    max of `High[t-k : t+k]`; low analog. A pivot is only known after `k`
    future bars close, so the last `k` bars never yield a pivot."""
    highs, lows = df["high"].to_numpy(), df["low"].to_numpy()
    n = len(df)
    pivots: list[dict] = []
    for i in range(k, n - k):
        window_hi = highs[i - k : i + k + 1]
        window_lo = lows[i - k : i + k + 1]
        if highs[i] == window_hi.max():
            pivots.append({"index": i, "price": float(highs[i]), "kind": "high"})
        if lows[i] == window_lo.min():
            pivots.append({"index": i, "price": float(lows[i]), "kind": "low"})
    return pivots


def zone_tolerance(atr_at_pivot: float, pivot_price: float) -> float:
    """TECH-ZTOL-024 — max(0.50·ATR14, 0.0075·pivot_price)."""
    return max(0.50 * atr_at_pivot, 0.0075 * pivot_price)


def _recency_weight(age_sessions: float) -> float:
    return float(np.exp(-_LN2 * age_sessions / _HALFLIFE))


def _weighted_median(values: list[float], weights: list[float]) -> float:
    order = np.argsort(values)
    v = np.asarray(values)[order]
    w = np.asarray(weights)[order]
    cum = np.cumsum(w)
    cutoff = cum[-1] / 2.0
    idx = int(np.searchsorted(cum, cutoff))
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def cluster_zones(
    pivots: list[dict], atr: float, price: float, last_index: int
) -> list[Zone]:
    """TECH-ZONE-025 — cluster same-type pivots whose tolerance intervals
    overlap into zones with a recency-weighted center and half-width."""
    zones: list[Zone] = []
    for kind, ztype in (("high", "resistance"), ("low", "support")):
        group = sorted(
            (p for p in pivots if p["kind"] == kind), key=lambda p: p["price"]
        )
        if not group:
            continue
        cluster: list[dict] = []

        def flush(cluster: list[dict]) -> None:
            if not cluster:
                return
            prices = [p["price"] for p in cluster]
            ages = [last_index - p["index"] for p in cluster]
            weights = [_recency_weight(a) for a in ages]
            tols = [zone_tolerance(atr, p["price"]) for p in cluster]
            center = _weighted_median(prices, weights)
            half = _weighted_median(tols, weights)
            zones.append(
                Zone(
                    zone_id=f"{ztype}_{center:.2f}",
                    type=ztype,  # type: ignore[arg-type]
                    lower=center - half,
                    center=center,
                    upper=center + half,
                )
            )

        for p in group:
            tol = zone_tolerance(atr, p["price"])
            if cluster and p["price"] - cluster[-1]["price"] <= 2 * tol:
                cluster.append(p)
            else:
                flush(cluster)
                cluster = [p]
        flush(cluster)
    return zones


# --- touches / strength / classification ------------------------------------


def count_touches(zone: Zone, df: pd.DataFrame, atr_series: pd.Series) -> list[Touch]:
    """TECH-NEFF-026/REJ-027 — independent touches inside the zone with a
    valid ≥0.5-ATR rejection within 3 sessions; daily touches must be ≥5
    sessions apart."""
    highs, lows = df["high"].to_numpy(), df["low"].to_numpy()
    vol = df["volume"].to_numpy()
    vol_med = df["volume"].rolling(50).median().to_numpy()
    n = len(df)
    last_index = n - 1
    touches: list[Touch] = []
    last_touch_idx = -10_000

    for i in range(n - 3):
        if i - last_touch_idx < 5:
            continue
        atr_i = atr_series.iloc[i]
        if not np.isfinite(atr_i) or atr_i == 0:
            continue
        if zone.type == "resistance":
            if not (zone.lower <= highs[i] <= zone.upper):
                continue
            reaction = (zone.center - lows[i + 1 : i + 4].min()) / atr_i
        else:
            if not (zone.lower <= lows[i] <= zone.upper):
                continue
            reaction = (highs[i + 1 : i + 4].max() - zone.center) / atr_i
        if reaction < 0.5:
            continue
        vr = vol[i] / vol_med[i] if np.isfinite(vol_med[i]) and vol_med[i] else 1.0
        touches.append(
            Touch(
                date=str(df.index[i]) if df.index.dtype == object else str(i),
                pivot_price=float(highs[i] if zone.type == "resistance" else lows[i]),
                rejection_atr=float(reaction),
                volume_ratio=float(vr),
                age_sessions=int(last_index - i),
            )
        )
        last_touch_idx = i
    return touches


def strength(zone: Zone) -> float:
    """TECH-LSTR-028 — level strength score (0-100)."""
    touches = zone.touches
    if not touches:
        return 0.0
    n_eff = sum(_recency_weight(t.age_sessions) for t in touches)
    median_reaction = float(np.median([t.rejection_atr for t in touches]))
    median_volume = float(np.median([t.volume_ratio for t in touches]))
    age_latest = min(t.age_sessions for t in touches)
    timeframe_pts = 10.0 if zone.timeframe == "weekly" else 5.0

    score = (
        30.0 * min(n_eff / 4.0, 1.0)
        + 20.0 * min(median_reaction / 2.0, 1.0)
        + 15.0 * min(median_volume / 1.5, 1.0)
        + 15.0 * _recency_weight(age_latest)
        + timeframe_pts
        + 10.0 * min(zone.confluence_count / 3.0, 1.0)
    )
    return min(score, 100.0)


def classify(zone: Zone) -> str:
    """TECH-ROLE / label rules — status from touch evidence.

    Pre-set `broken`/`role_reversed` statuses are preserved; otherwise:
    Candidate (1 touch), Confirmed (≥2), Strong (≥3 OR 2 touches with median
    reaction ≥1 ATR and any volume ratio ≥1.5)."""
    if zone.status in ("broken", "role_reversed"):
        return zone.status
    touches = zone.touches
    n = len(touches)
    if n == 0:
        return "candidate"
    if n >= 3:
        return "strong"
    if n == 2:
        median_reaction = float(np.median([t.rejection_atr for t in touches]))
        strong_2 = median_reaction >= 1.0 and any(t.volume_ratio >= 1.5 for t in touches)
        return "strong" if strong_2 else "confirmed"
    return "candidate"


# --- breakouts --------------------------------------------------------------


def breakout_confirmed(df: pd.DataFrame, zone: Zone, atr: float) -> bool:
    """TECH-BCONF-031 — buffer pass AND volume ≥1.5× median AND (two closes
    above OR one close then 3 sessions with no close back inside)."""
    close = df["close"].to_numpy()
    vol = df["volume"].to_numpy()
    vol_med = df["volume"].rolling(50).median().to_numpy()
    buffer = zone.upper + 0.25 * atr
    n = len(df)

    for i in range(n):
        if close[i] <= buffer:
            continue
        if not (np.isfinite(vol_med[i]) and vol_med[i] and vol[i] / vol_med[i] >= 1.5):
            continue
        # Two consecutive closes above buffer.
        if i + 1 < n and close[i + 1] > buffer:
            return True
        # One close then 3 sessions with no close back inside the zone.
        follow = close[i + 1 : i + 4]
        if len(follow) >= 1 and not np.any(follow < zone.lower):
            if np.all(follow > zone.lower):
                return True
    return False


# --- AVWAP / volume profile / gaps ------------------------------------------


def avwap(df: pd.DataFrame, anchor_index: int) -> float:
    """TECH-AVWAP-034 — anchored VWAP from `anchor_index` to the last bar.
    TypicalPrice = (H+L+C)/3."""
    seg = df.iloc[anchor_index:]
    tp = (seg["high"] + seg["low"] + seg["close"]) / 3.0
    vol = seg["volume"]
    denom = vol.sum()
    if denom == 0:
        return float("nan")
    return float((tp * vol).sum() / denom)


def volume_profile(df: pd.DataFrame, atr: float) -> dict:
    """TECH-VP-035 — approximate volume profile. bin_width = max(0.5·ATR,
    0.5%·price). Returns POC, and HVN/LVN price bins."""
    price = float(df["close"].iloc[-1])
    bin_width = max(0.5 * atr, 0.005 * price)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    bins = (tp / bin_width).round().astype(int)
    grouped = df["volume"].groupby(bins).sum()
    poc_bin = grouped.idxmax()
    p75, p25 = grouped.quantile(0.75), grouped.quantile(0.25)
    hvn = [float(b * bin_width) for b, v in grouped.items() if v > p75]
    lvn = [float(b * bin_width) for b, v in grouped.items() if v < p25]
    return {"poc": float(poc_bin * bin_width), "hvn": hvn, "lvn": lvn, "bin_width": bin_width}


def earnings_gaps(df: pd.DataFrame, earnings_indices: list[int], atr: float) -> list[Gap]:
    """TECH-GAP-020/GHOLD-021 — earnings gaps; material when |open−prior close|
    ≥ max(1.0·ATR, 0.03·prior close). Tracks day-1/5/20 hold ratios."""
    gaps: list[Gap] = []
    close = df["close"].to_numpy()
    open_ = df["open"].to_numpy()
    n = len(df)
    for idx in earnings_indices:
        if idx <= 0 or idx >= n:
            continue
        prior_close = close[idx - 1]
        gap = open_[idx] - prior_close
        threshold = max(1.0 * atr, 0.03 * prior_close)
        material = abs(gap) >= threshold
        gap_pct = gap / prior_close if prior_close else 0.0

        def hold(k: int) -> float | None:
            if idx + k >= n or gap == 0:
                return None
            return float((close[idx + k] - prior_close) / gap)

        gaps.append(
            Gap(
                date=str(idx),
                gap_percent=float(gap_pct),
                material=bool(material),
                day1_hold=hold(1),
                day5_hold=hold(5),
                day20_hold=hold(20),
            )
        )
    return gaps


# --- ranking / orchestration ------------------------------------------------


def rank_levels(zones: list[Zone]) -> list[Zone]:
    """TECH — rank = 0.45·strength + 0.25·recency + 0.20·confluence +
    0.10·liquidity. Returns zones sorted strongest-first."""

    def score(z: Zone) -> float:
        recency = _recency_weight(min((t.age_sessions for t in z.touches), default=252))
        confluence = min(z.confluence_count / 3.0, 1.0)
        return (
            0.45 * (z.strength_0_100 / 100.0)
            + 0.25 * recency
            + 0.20 * confluence
            + 0.10 * z.liquidity_confidence
        )

    return sorted(zones, key=score, reverse=True)


def compute_levels(
    df_daily: pd.DataFrame, earnings_indices: list[int] | None = None
) -> LevelsOutput:
    """Orchestrate the daily technical zone engine into a `LevelsOutput`:
    nearest 3 support + 3 resistance zones, key moving averages, an AVWAP
    anchored at the series start, and material earnings gaps."""
    df = df_daily.reset_index(drop=True)
    atr_series = atr14(df)
    atr = float(atr_series.iloc[-1])
    price = float(df["close"].iloc[-1])
    last_index = len(df) - 1

    pivots = find_pivots(df, k=3)
    zones = cluster_zones(pivots, atr, price, last_index)
    for z in zones:
        z.touches = count_touches(z, df, atr_series)
        z.strength_0_100 = strength(z)
        z.status = classify(z)  # type: ignore[assignment]
        z.distance_percent = (z.center - price) / price * 100.0
        z.distance_atr = (z.center - price) / atr if atr else None

    resistance = sorted(
        (z for z in zones if z.center >= price), key=lambda z: z.center
    )[:3]
    support = sorted(
        (z for z in zones if z.center < price), key=lambda z: -z.center
    )[:3]

    mas = {}
    for n in (50, 200):
        val = sma(df["close"], n).iloc[-1]
        if np.isfinite(val):
            mas[f"sma{n}"] = float(val)

    avwaps = {"anchor_start": avwap(df, 0)}
    gaps = earnings_gaps(df, earnings_indices or [], atr)

    return LevelsOutput(
        support=support,
        resistance=resistance,
        moving_averages=mas,
        avwaps=avwaps,
        gaps=gaps,
    )
