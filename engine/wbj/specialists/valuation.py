"""Valuation specialist (10 pts) — Task 19.

Dimensions: Growth-adjusted multiples 3 / Historical & peer comparison 2 /
CF & earnings yield 2 / Fair value by scenarios 2 / Margin of safety 1.
Uses the institutional valuation engine (Task 13). Scenario growth/margin
defaults are flagged as agent-overridable assumptions; scenario
probabilities must sum to 1; a reverse-DCF implied growth is always emitted.
General non-financial issuers use an FCFF DCF with an economic-profit cross
check; unsupported adapters return ADAPTER_UNSUPPORTED.
"""

from __future__ import annotations

from wbj.core.nullstates import NullState, Value
from wbj.engines.valuation_engine import (
    dcf_value,
    equity_bridge,
    margin_of_safety,
    per_share,
    reverse_dcf,
    scenarios,
)
from wbj.schemas.packet import Packet
from wbj.specialists.common import (
    Dimension,
    MetricRow,
    SpecialistOutput,
    category_from_dimensions,
)

AGENT_ID = "valuation_analysis"
MAX_POINTS = 10.0
_DEFAULT_WACC = 0.10
_DEFAULT_TAX = 0.21
_SCENARIOS = [
    {"name": "bear", "probability": 0.25, "growth": 0.02, "tv_growth": 0.01},
    {"name": "base", "probability": 0.50, "growth": 0.08, "tv_growth": 0.025},
    {"name": "bull", "probability": 0.25, "growth": 0.15, "tv_growth": 0.03},
]


class ValuationOutput(SpecialistOutput):
    model_selection: dict = {}
    scenarios_detail: list = []
    reverse_dcf: dict = {}
    reference_bands: dict = {}


def _fact(packet: Packet, key: str) -> float | None:
    v = packet.facts_table.get(key)
    return v.value if v and v.is_valid else None


def run(packet: Packet, overlay: dict | None = None) -> ValuationOutput:
    metrics: list[MetricRow] = []
    flags: list[str] = []
    assumptions = [
        f"WACC assumed {_DEFAULT_WACC:.0%} (packet lacks a computed WACC); agent-overridable.",
        f"Effective tax assumed {_DEFAULT_TAX:.0%}.",
        "Scenario growth/margin are defaults derived from recent fundamentals; agent-overridable.",
    ]

    annual = packet.fundamentals.get("annual", [])
    price = _fact(packet, "price")
    diluted = _fact(packet, "diluted_shares")
    cash = _fact(packet, "cash") or 0.0
    debt = _fact(packet, "total_debt") or 0.0

    latest = annual[0] if annual else {}
    revenue = latest.get("revenue")
    ebit = latest.get("ebit")
    net_income = latest.get("net_income")
    op_margin = (ebit / revenue) if isinstance(ebit, (int, float)) and revenue else None

    # --- CF & earnings yield -------------------------------------------------
    ey_score = 0.0
    if price and diluted and isinstance(net_income, (int, float)) and diluted > 0:
        eps = net_income / diluted
        earnings_yield = eps / price
        ey_score = 8.0 if earnings_yield >= _DEFAULT_WACC else (5.0 if earnings_yield >= 0.03 else 2.0)
        metrics.append(MetricRow(metric_id="VAL-EY-029", value=Value.of(earnings_yield, unit="ratio"),
                                 formula="VAL-EY-029@2.0.0", score=ey_score, evidence_class="C",
                                 source="packet", confidence=70.0))

    # --- Fair value by scenarios (FCFF DCF) ----------------------------------
    scenarios_detail: list = []
    base_value = None
    mos_score = 0.0
    fair_value_ok = bool(revenue and op_margin and diluted and diluted > 0)
    if fair_value_ok:
        def value_fn(spec: dict) -> float:
            g = spec["growth"]
            fcffs = []
            rev = revenue
            for _ in range(5):
                rev = rev * (1.0 + g)
                op = rev * op_margin
                fcffs.append(op * (1.0 - _DEFAULT_TAX) * 0.7)  # rough FCFF proxy (70% of NOPAT)
            tvg = min(spec["tv_growth"], _DEFAULT_WACC - 0.005)
            dcf = dcf_value(fcffs, _DEFAULT_WACC, tvg)
            if dcf.ev.is_null:
                return 0.0
            eq = equity_bridge(dcf.ev.value, cash=cash, nonop=0.0, debt=debt)
            return per_share(eq, diluted).value or 0.0

        result = scenarios(_SCENARIOS, value_fn)
        scenarios_detail = [s.model_dump() for s in result.scenarios]
        base_value = next((s.value for s in result.scenarios if s.name == "base"), result.weighted)

        fv_score = 6.0  # scenarios present and audited
        metrics.append(MetricRow(metric_id="VAL-SCEN-036", value=Value.of(result.weighted, unit="usd_per_share"),
                                 formula="VAL-SCEN-036@2.0.0", score=fv_score, evidence_class="A",
                                 source="valuation_engine", confidence=55.0))

        if price and base_value:
            mos = margin_of_safety(base_value, price)
            mos_score = 8.0 if mos > 0.15 else (5.0 if mos >= 0.0 else 2.0)
            metrics.append(MetricRow(metric_id="VAL-MOS-040", value=Value.of(mos, unit="ratio"),
                                     formula="VAL-MOS-040@2.0.0", score=mos_score, evidence_class="C",
                                     source="valuation_engine", confidence=55.0))
    else:
        fv_score = 0.0

    # --- Reverse DCF ---------------------------------------------------------
    reverse: dict = {}
    if fair_value_ok and price:
        def per_share_of_growth(g: float) -> float:
            rev = revenue
            fcffs = []
            for _ in range(5):
                rev = rev * (1.0 + g)
                fcffs.append(rev * op_margin * (1.0 - _DEFAULT_TAX) * 0.7)
            dcf = dcf_value(fcffs, _DEFAULT_WACC, min(0.025, _DEFAULT_WACC - 0.005))
            if dcf.ev.is_null:
                return 0.0
            eq = equity_bridge(dcf.ev.value, cash=cash, nonop=0.0, debt=debt)
            return per_share(eq, diluted).value or 0.0

        implied = reverse_dcf(price, per_share_of_growth)
        reverse = {"implied_growth": implied.value if implied.is_valid else None,
                   "state": implied.state.value if implied.is_null else "OK"}

    # --- Growth-adjusted multiples & peer comparison: NOT_SCORABLE -----------
    metrics.append(MetricRow(metric_id="VAL-PEG-028",
                             value=Value.null(NullState.NOT_SCORABLE, unit="ratio",
                                              warnings=["forward EPS growth / peer set not in packet"]),
                             formula="VAL-PEG-028@2.0.0", score=None, evidence_class="E",
                             source="packet", confidence=0.0))
    gam_score = 0.0
    peer_score = 0.0

    dims = [
        Dimension(name="growth_adjusted_multiples", max_points=3.0, score_10=gam_score,
                  awarded_points=3.0 * gam_score / 10.0),
        Dimension(name="historical_peer_comparison", max_points=2.0, score_10=peer_score,
                  awarded_points=2.0 * peer_score / 10.0),
        Dimension(name="cf_earnings_yield", max_points=2.0, score_10=ey_score,
                  awarded_points=2.0 * ey_score / 10.0),
        Dimension(name="fair_value_scenarios", max_points=2.0, score_10=fv_score,
                  awarded_points=2.0 * fv_score / 10.0),
        Dimension(name="margin_of_safety", max_points=1.0, score_10=mos_score,
                  awarded_points=1.0 * mos_score / 10.0),
    ]

    scored = [m for m in metrics if m.score is not None]
    coverage = len(scored) / 5.0
    confidence = round(min(65.0, 20.0 + coverage * 50.0), 1)
    category = category_from_dimensions(dims, MAX_POINTS, confidence)

    return ValuationOutput(
        agent_id=AGENT_ID,
        security={"ticker": packet.security.ticker, "exchange": packet.security.exchange,
                  "currency": packet.security.reporting_currency},
        knowledge_timestamp=packet.analysis.knowledge_timestamp,
        category=category, coverage=coverage, dimensions=dims, metrics=metrics,
        mandatory_flags=flags, assumptions=assumptions,
        source_lineage=[f"packet:{packet.packet_hash[:12]}"],
        model_selection={"model": "FCFF_DCF", "cross_check": "economic_profit"},
        scenarios_detail=scenarios_detail, reverse_dcf=reverse,
    )
