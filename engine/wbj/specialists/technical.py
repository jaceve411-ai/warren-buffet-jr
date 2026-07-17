"""Technical & Momentum specialist (20 pts) — Task 17.

Dimensions: Primary trend 4 / Relative strength 4 / Volume demand 3 /
Earnings-gap behavior 3 / Breakout & base quality 3 / Sector breadth &
volatility 3. Trend uses the verbatim SMA/ADX/52w anchors from
DECISION_RULES.md; relative strength is NOT_SCORABLE without a benchmark
series (on the prohibited-imputation list). Consumes the indicator and
levels engines (Tasks 11-12).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wbj.core.nullstates import NullState, Value
from wbj.engines.indicators import (
    adx14,
    atr14,
    cmf,
    range_position_52w,
    sma,
    up_down_volume_ratio,
)
from wbj.engines.levels_engine import compute_levels
from wbj.schemas.packet import Packet
from wbj.specialists.common import (
    Dimension,
    JudgmentRequest,
    MetricRow,
    SpecialistOutput,
    category_from_dimensions,
)

AGENT_ID = "technical_momentum"
MAX_POINTS = 20.0


class TechnicalOutput(SpecialistOutput):
    market_state: dict = {}
    indicators: dict = {}
    important_levels: dict = {}


def trend_anchor_score(
    close: float,
    sma50: float,
    sma200: float,
    sma200_slope_atr: float,
    adx: float,
    pos52w: float,
) -> float:
    """DECISION_RULES.md primary-trend anchors (0-10)."""
    if adx >= 25 and pos52w >= 0.80 and close > sma50 > sma200:
        return 9.0
    if close > sma50 > sma200 and sma200_slope_atr > 0:
        return 8.0
    if close > sma200 and not (close > sma50 > sma200):
        return 6.0
    if abs(close - sma200) <= 1.0 and abs(sma200_slope_atr) <= 0.25:
        return 5.0
    if close < sma50 < sma200 and sma200_slope_atr < -1.0:
        return 1.0
    if close < sma200:
        return 3.0
    return 4.0


def _packet_df(packet: Packet) -> pd.DataFrame:
    daily = [row.model_dump() for row in packet.market_data.daily]
    df = pd.DataFrame(daily)
    if df.empty:
        return df
    return df.iloc[::-1].reset_index(drop=True)  # packet is newest-first


def run(packet: Packet, overlay: dict | None = None) -> TechnicalOutput:
    df = _packet_df(packet)
    metrics: list[MetricRow] = []
    flags: list[str] = []
    requests: list[JudgmentRequest] = []
    indicators: dict = {}

    n = len(df)
    has_200 = n >= 200

    # --- Primary trend -------------------------------------------------------
    trend_score = 0.0
    if n >= 50:
        close = float(df["close"].iloc[-1])
        sma50 = float(sma(df["close"], 50).iloc[-1])
        atr = float(atr14(df).iloc[-1])
        sma200_series = sma(df["close"], 200)
        if has_200 and np.isfinite(sma200_series.iloc[-1]):
            sma200 = float(sma200_series.iloc[-1])
            slope = (sma200 - float(sma200_series.iloc[-51])) / atr if atr else 0.0
            adx = float(adx14(df).iloc[-1])
            pos = float(range_position_52w(df).iloc[-1])
            pos = pos if np.isfinite(pos) else 0.5
            trend_score = trend_anchor_score(close, sma50, sma200, slope, adx, pos)
            indicators.update({"sma50": sma50, "sma200": sma200, "adx": adx, "pos52w": pos})
            metrics.append(MetricRow(metric_id="TECH-trend-anchor", value=Value.of(trend_score, unit="score"),
                                     formula="TECH-anchors@2.0.0", score=trend_score, evidence_class="C",
                                     source="indicators", confidence=80.0))
        else:
            trend_score = min(6.0, 4.0)  # no valid SMA200 -> capped 6
            flags.append("TREND_CAP_NO_SMA200")

    # --- Relative strength: NOT_SCORABLE without a benchmark series ----------
    metrics.append(MetricRow(metric_id="TECH-rs-percentile",
                             value=Value.null(NullState.NOT_SCORABLE, unit="pct",
                                              warnings=["no benchmark OHLCV in packet"]),
                             formula="TECH-RSC-013@2.0.0", score=None, evidence_class="C",
                             source="indicators", confidence=0.0))
    rs_score = 0.0

    # --- Volume demand -------------------------------------------------------
    vol_score = 5.0  # capped 5 if volume weak/missing
    if "volume" in df.columns and n >= 50:
        ud = float(up_down_volume_ratio(df).iloc[-1]) if np.isfinite(up_down_volume_ratio(df).iloc[-1]) else np.nan
        cm = float(cmf(df).iloc[-1])
        indicators.update({"up_down_vol": ud, "cmf": cm})
        if np.isfinite(ud):
            if ud > 1.2 and cm > 0.10:
                vol_score = 8.0
            elif cm < -0.10:
                vol_score = 3.0
            else:
                vol_score = 5.0
        metrics.append(MetricRow(metric_id="TECH-CMF", value=Value.of(cm, unit="ratio"),
                                 formula="TECH-CMF-017@2.0.0", score=vol_score, evidence_class="C",
                                 source="indicators", confidence=70.0))

    # --- Earnings-gap behavior: needs >=4 valid events -----------------------
    gap_score = 0.0
    levels = compute_levels(df) if n >= 252 else None
    # Breakout & base and sector breadth are judgment / benchmark dependent.
    requests.append(JudgmentRequest(request_id="TECH-base-quality", agent_id=AGENT_ID,
                                    metric_id="TECH-base", question="Assess base/breakout quality.",
                                    schema_hint="{base_quality: 0-10}"))
    breakout_score = 0.0
    breadth_score = 0.0

    dims = [
        Dimension(name="primary_trend", max_points=4.0, score_10=trend_score, awarded_points=4.0 * trend_score / 10.0),
        Dimension(name="relative_strength", max_points=4.0, score_10=rs_score, awarded_points=4.0 * rs_score / 10.0),
        Dimension(name="volume_demand", max_points=3.0, score_10=vol_score, awarded_points=3.0 * vol_score / 10.0),
        Dimension(name="earnings_gap_behavior", max_points=3.0, score_10=gap_score, awarded_points=3.0 * gap_score / 10.0),
        Dimension(name="breakout_base_quality", max_points=3.0, score_10=breakout_score,
                  awarded_points=3.0 * breakout_score / 10.0),
        Dimension(name="sector_breadth_volatility", max_points=3.0, score_10=breadth_score,
                  awarded_points=3.0 * breadth_score / 10.0),
    ]

    scored = [m for m in metrics if m.score is not None]
    coverage = len(scored) / 6.0
    confidence = round(min(75.0, 25.0 + coverage * 50.0), 1)
    category = category_from_dimensions(dims, MAX_POINTS, confidence)

    return TechnicalOutput(
        agent_id=AGENT_ID,
        security={"ticker": packet.security.ticker, "exchange": packet.security.exchange,
                  "currency": packet.security.reporting_currency},
        knowledge_timestamp=packet.analysis.knowledge_timestamp,
        category=category, coverage=coverage, dimensions=dims, metrics=metrics,
        mandatory_flags=flags, judgment_requests=requests,
        source_lineage=[f"packet:{packet.packet_hash[:12]}"],
        indicators=indicators,
        important_levels=levels.model_dump() if levels else {},
    )
