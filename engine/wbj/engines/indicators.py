"""Task 11 — technical indicator library.

Pure-math implementations of the TECH-* formulas in
`Cerebro/04_technical_momentum/FORMULAS.md`. Every function takes a pandas
`Series` (adjusted close/volume) or a `DataFrame` of adjusted OHLCV columns
(`open`, `high`, `low`, `close`, `volume`) and returns a `Series` aligned to
the input index — except the two explicitly-scalar reductions
(`composite_rs_percentile`).

Sign conventions and smoothing follow the registry verbatim:
- ATR/RSI/ADX use **Wilder** smoothing: `X_t = ((n-1)*X_{t-1} + x_t)/n`,
  seeded with the simple mean of the first `n` observations (TECH-ATR-006,
  TECH-RSI-007, TECH-DMI-009).
- EMA uses `alpha = 2/(n+1)`, seeded with the SMA of the first `n` closes
  (TECH-EMA-003).
Values before an indicator has enough observations to seed are `NaN`, never
imputed to 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_LN2 = np.log(2.0)


# --- Wilder smoothing core --------------------------------------------------


def _wilder(series: pd.Series, n: int) -> pd.Series:
    """Wilder-smooth `series`: seed with the mean of the first `n` valid
    observations, then `X_t = ((n-1)*X_{t-1} + x_t)/n`.

    Leading NaNs in `series` are skipped when locating the seed window, so
    this composes correctly on already-lagged inputs (e.g. true range, DX).
    """
    out = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.dropna()
    if len(valid) < n:
        return out

    seed = valid.iloc[:n].mean()
    labels = valid.index
    out.loc[labels[n - 1]] = seed
    prev = seed
    for i in range(n, len(valid)):
        prev = (prev * (n - 1) + valid.iloc[i]) / n
        out.loc[labels[i]] = prev
    return out


# --- moving averages --------------------------------------------------------


def sma(close: pd.Series, n: int) -> pd.Series:
    """TECH-SMA-002 — simple moving average; NaN until `n` observations."""
    return close.rolling(n).mean()


def ema(close: pd.Series, n: int) -> pd.Series:
    """TECH-EMA-003 — EMA with `alpha=2/(n+1)`, seeded with the SMA of the
    first `n` (non-NaN) closes."""
    alpha = 2.0 / (n + 1)
    out = pd.Series(np.nan, index=close.index, dtype=float)
    valid = close.dropna()
    if len(valid) < n:
        return out

    seed = valid.iloc[:n].mean()
    labels = valid.index
    out.loc[labels[n - 1]] = seed
    prev = seed
    for i in range(n, len(valid)):
        prev = alpha * valid.iloc[i] + (1 - alpha) * prev
        out.loc[labels[i]] = prev
    return out


# --- true range / ATR -------------------------------------------------------


def true_range(df: pd.DataFrame) -> pd.Series:
    """TECH-TR-005 — max(H-L, |H-prevC|, |L-prevC|). First bar has no prior
    close, so its true range is simply H-L."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return tr


def atr14(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """TECH-ATR-006 — Wilder ATR of true range, seed = mean of first `n` TRs."""
    return _wilder(true_range(df), n)


# --- RSI ---------------------------------------------------------------------


def rsi14(close: pd.Series, n: int = 14) -> pd.Series:
    """TECH-RSI-007 — Wilder RSI. A zero average loss yields RSI=100."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = pd.Series(np.nan, index=close.index, dtype=float)
    avg_loss = pd.Series(np.nan, index=close.index, dtype=float)
    if len(close) < n + 1:
        return pd.Series(np.nan, index=close.index, dtype=float)

    # First delta (index 0) is NaN, so the seed window is deltas 1..n.
    ag = gain.iloc[1 : n + 1].mean()
    al = loss.iloc[1 : n + 1].mean()
    avg_gain.iloc[n] = ag
    avg_loss.iloc[n] = al
    for i in range(n + 1, len(close)):
        ag = (ag * (n - 1) + gain.iloc[i]) / n
        al = (al * (n - 1) + loss.iloc[i]) / n
        avg_gain.iloc[i] = ag
        avg_loss.iloc[i] = al

    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # Zero average loss -> RSI 100 (avoid inf/NaN); keep pre-seed rows NaN.
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    rsi = rsi.mask(avg_loss.isna())
    return rsi


# --- MACD / ADX / ROC -------------------------------------------------------


def macd(close: pd.Series) -> dict[str, pd.Series]:
    """TECH-MACD-008 — 12/26/9 MACD. Returns macd/signal/hist Series."""
    macd_line = ema(close, 12) - ema(close, 26)
    signal = ema(macd_line, 9)
    return {"macd": macd_line, "signal": signal, "hist": macd_line - signal}


def adx14(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """TECH-DMI-009 — Wilder ADX14 (trend strength, direction-agnostic)."""
    high, low = df["high"], df["low"]
    up_move = high.diff()
    down_move = low.shift(1) - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )
    plus_dm.iloc[0] = 0.0
    minus_dm.iloc[0] = 0.0

    atr = _wilder(true_range(df), n)
    plus_di = 100.0 * _wilder(plus_dm, n) / atr
    minus_di = 100.0 * _wilder(minus_dm, n) / atr

    di_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    dx = dx.where(di_sum != 0.0)
    return _wilder(dx, n)


def roc(close: pd.Series, n: int) -> pd.Series:
    """TECH-ROC-010 — rate of change: Close_t / Close_{t-n} - 1."""
    return close / close.shift(n) - 1.0


# --- relative strength ------------------------------------------------------


def relative_strength(close: pd.Series, bench: pd.Series, n: int) -> pd.Series:
    """Ratio of the security's `n`-day price relative to the benchmark's.

    `(close_t/close_{t-n}) / (bench_t/bench_{t-n})` — above 1 means the
    security out-returned the benchmark over the window (TECH-RS-011 family,
    expressed as a ratio per the Task 11 interface)."""
    stock_rel = close / close.shift(n)
    bench_rel = bench / bench.shift(n)
    return stock_rel / bench_rel


def composite_rs_percentile(rs_by_window: dict[int, float], universe: pd.DataFrame) -> float:
    """TECH-RSC-013 — weighted blend of the security's percentile rank within
    a point-in-time `universe` of RS values across four windows.

    `0.35*Pct(RS21) + 0.25*Pct(RS63) + 0.25*Pct(RS126) + 0.15*Pct(RS252)`,
    each percentile expressed 0-100. `universe` must have one column per
    window key (21/63/126/252)."""
    weights = {21: 0.35, 63: 0.25, 126: 0.25, 252: 0.15}
    total = 0.0
    for window, weight in weights.items():
        col = universe[window].dropna()
        pct = float((col <= rs_by_window[window]).mean()) * 100.0
        total += weight * pct
    return total


# --- volatility / volume ----------------------------------------------------


def realized_vol(close: pd.Series, n: int) -> pd.Series:
    """TECH-VOL-018 — annualized realized vol: rolling std of log returns × √252."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(n).std(ddof=1) * np.sqrt(252.0)


def volume_ratio(volume: pd.Series, n: int = 50) -> pd.Series:
    """TECH-VR-014 — volume vs its trailing `n`-session median."""
    return volume / volume.rolling(n).median()


def up_down_volume_ratio(df: pd.DataFrame, n: int = 50) -> pd.Series:
    """TECH-UDV-015 — Σ(up-close volume) / Σ(down-close volume) over `n`.

    A zero down-volume denominator is not meaningful and yields NaN."""
    change = df["close"].diff()
    up_vol = df["volume"].where(change > 0, 0.0)
    down_vol = df["volume"].where(change < 0, 0.0)
    ratio = up_vol.rolling(n).sum() / down_vol.rolling(n).sum()
    return ratio.replace([np.inf, -np.inf], np.nan)


def obv(df: pd.DataFrame) -> pd.Series:
    """TECH-OBV-016 — on-balance volume (use slope/divergence, not level)."""
    sign = np.sign(df["close"].diff()).fillna(0.0)
    return (sign * df["volume"]).cumsum()


def cmf(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """TECH-CMF-017 — Chaikin money flow. H==L bars contribute a 0 multiplier."""
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]
    hl = high - low
    mult = ((2.0 * close - high - low) / hl).where(hl != 0.0, 0.0)
    mfv = mult * volume
    return mfv.rolling(n).sum() / volume.rolling(n).sum()


def range_position_52w(df: pd.DataFrame, n: int = 252) -> pd.Series:
    """TECH-52W-036 — (Close - Nd low) / (Nd high - Nd low). Zero range -> NaN."""
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rng = high_n - low_n
    pos = (df["close"] - low_n) / rng
    return pos.where(rng != 0.0)


def median_dollar_volume(df: pd.DataFrame, n: int = 63) -> pd.Series:
    """TECH-LIQ-040 — rolling median of Close×Volume over `n` sessions."""
    return (df["close"] * df["volume"]).rolling(n).median()
