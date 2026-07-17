"""Risk & Resilience specialist (15 pts, higher = safer) — Task 18.

Dimensions: Financing 3 / Concentration 3 / Execution & earnings quality 3 /
Regulatory-macro 2 / Valuation-compression 2 / Volatility & drawdown 2.
Includes closed-form Beneish M-score, Altman Z″ and Piotroski F, historical
VaR/CVaR, and drawdown. Beta / downside beta are on the prohibited-imputation
list and never proxied. Reads the investor profile for position-fit notes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wbj.core.nullstates import NullState, Value
from wbj.schemas.packet import Packet
from wbj.specialists.common import (
    Dimension,
    JudgmentRequest,
    MetricRow,
    SpecialistOutput,
    category_from_dimensions,
)

AGENT_ID = "risk_analysis"
MAX_POINTS = 15.0
SOLVENCY_WARNING_TEXT = "Operating earnings do not provide a comfortable interest buffer."


class RiskOutput(SpecialistOutput):
    profile_fit: dict = {}
    distress_scores: dict = {}


# --- closed-form distress models --------------------------------------------


def beneish_m_score(dsri, gmi, aqi, sgi, depi, sgai, tata, lvgi) -> float:
    """8-variable Beneish M-score; > −1.78 flags possible manipulation."""
    return (
        -4.84
        + 0.920 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.679 * tata
        - 0.327 * lvgi
    )


def altman_z_double_prime(wc, re, ebit, book_equity, ta, tl) -> float:
    """Altman Z″ for non-manufacturers = 6.56·WC/TA + 3.26·RE/TA +
    6.72·EBIT/TA + 1.05·BE/TL."""
    return 6.56 * wc / ta + 3.26 * re / ta + 6.72 * ebit / ta + 1.05 * book_equity / tl


def piotroski_f(signals: dict[str, bool]) -> int:
    """Piotroski F-score (0-9): count of satisfied binary signals."""
    return sum(1 for v in signals.values() if v)


# --- market risk -------------------------------------------------------------


def historical_var(returns: np.ndarray, level: float = 0.95) -> float:
    """Historical VaR: the loss at the (1−level) quantile (positive number)."""
    q = np.percentile(returns, (1.0 - level) * 100.0)
    return float(-q)


def historical_cvar(returns: np.ndarray, level: float = 0.95) -> float:
    """Historical CVaR: mean loss beyond the VaR threshold."""
    q = np.percentile(returns, (1.0 - level) * 100.0)
    tail = returns[returns <= q]
    return float(-tail.mean()) if tail.size else 0.0


def max_drawdown(prices: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown (negative fraction)."""
    running_max = np.maximum.accumulate(prices)
    drawdowns = prices / running_max - 1.0
    return float(drawdowns.min())


def _load_profile() -> str:
    for base in (Path.cwd(), Path(__file__).resolve().parents[3]):
        p = base / "Perfil Inversionista" / "Victor Gonzalez.md"
        if p.exists():
            return p.read_text(encoding="utf-8", errors="ignore")
    return ""


def run(packet: Packet, overlay: dict | None = None) -> RiskOutput:
    metrics: list[MetricRow] = []
    flags: list[str] = []
    requests: list[JudgmentRequest] = []
    distress: dict = {}

    daily = [row.model_dump() for row in packet.market_data.daily]
    df = pd.DataFrame(daily)

    # --- Volatility & drawdown dimension -------------------------------------
    vol_dd_score = 0.0
    if not df.empty:
        prices = df["close"].to_numpy()[::-1]  # oldest-first
        log_ret = np.diff(np.log(prices))
        ann_vol = float(np.std(log_ret, ddof=1) * np.sqrt(252))
        mdd = max_drawdown(prices)
        var95 = historical_var(log_ret, 0.95)
        distress.update({"annualized_vol": ann_vol, "max_drawdown": mdd, "var95_1d": var95})
        # Anchor: 3y maxDD better than -30% good; -30..-60 mid; worse than -60 bad.
        if mdd > -0.30:
            vol_dd_score = 8.0
        elif mdd > -0.60:
            vol_dd_score = 5.0
        else:
            vol_dd_score = 2.0
        metrics.append(MetricRow(metric_id="RSK-maxdd", value=Value.of(mdd, unit="ratio"),
                                 formula="RSK-drawdown@2.0.0", score=vol_dd_score, evidence_class="C",
                                 source="packet.market_data", confidence=75.0))

    # --- Beta / downside beta: prohibited to proxy ---------------------------
    metrics.append(MetricRow(metric_id="RSK-beta",
                             value=Value.null(NullState.NOT_SCORABLE, unit="ratio",
                                              warnings=["no benchmark series; beta never proxied"]),
                             formula="RSK-beta@2.0.0", score=None, evidence_class="C",
                             source="packet.market_data", confidence=0.0))

    # --- Financing / concentration / execution: judgment or NOT_SCORABLE -----
    requests.append(JudgmentRequest(request_id="RSK-thesis-killers", agent_id=AGENT_ID,
                                    metric_id="RSK-thesis-killers",
                                    question="List ≥3 risk thesis-killers with early-warning metrics.",
                                    schema_hint="{killers: [{risk, early_warning_metric}]}"))
    financing_score = 0.0
    concentration_score = 0.0
    execution_score = 0.0
    regmacro_score = 0.0
    valcompression_score = 0.0

    # --- Profile fit ---------------------------------------------------------
    profile_text = _load_profile()
    profile_fit = {
        "profile_loaded": bool(profile_text),
        "capital_usd": 25000,
        "max_position_pct_range": [30, 60],
        "horizon_years": [3, 5],
        "note": "Position sizing vs 30-60% cap requires the aggregate volatility/beta read.",
    }

    dims = [
        Dimension(name="financing", max_points=3.0, score_10=financing_score, awarded_points=3.0 * financing_score / 10.0),
        Dimension(name="concentration", max_points=3.0, score_10=concentration_score,
                  awarded_points=3.0 * concentration_score / 10.0),
        Dimension(name="execution_earnings_quality", max_points=3.0, score_10=execution_score,
                  awarded_points=3.0 * execution_score / 10.0),
        Dimension(name="regulatory_macro", max_points=2.0, score_10=regmacro_score,
                  awarded_points=2.0 * regmacro_score / 10.0),
        Dimension(name="valuation_compression", max_points=2.0, score_10=valcompression_score,
                  awarded_points=2.0 * valcompression_score / 10.0),
        Dimension(name="volatility_drawdown", max_points=2.0, score_10=vol_dd_score,
                  awarded_points=2.0 * vol_dd_score / 10.0),
    ]

    scored = [m for m in metrics if m.score is not None]
    coverage = len(scored) / 6.0
    confidence = round(min(70.0, 25.0 + coverage * 50.0), 1)
    category = category_from_dimensions(dims, MAX_POINTS, confidence)

    return RiskOutput(
        agent_id=AGENT_ID,
        security={"ticker": packet.security.ticker, "exchange": packet.security.exchange,
                  "currency": packet.security.reporting_currency},
        knowledge_timestamp=packet.analysis.knowledge_timestamp,
        category=category, coverage=coverage, dimensions=dims, metrics=metrics,
        mandatory_flags=flags, judgment_requests=requests,
        source_lineage=[f"packet:{packet.packet_hash[:12]}"],
        profile_fit=profile_fit, distress_scores=distress,
    )
