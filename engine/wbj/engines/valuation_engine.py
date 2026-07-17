"""Task 13 — institutional valuation engine.

Transcribes the VAL-* formula registry (`Cerebro/06_valuation_analysis/
FORMULAS.md`) and the method in `Cerebro/special_sauces/
INSTITUTIONAL_VALUATION_ENGINE.md`: normalization, ROIC/economic-profit,
discount rate, FCFF/FCFE DCF, cross-checks, reverse DCF, scenarios, Monte
Carlo and the reliability-weighted ensemble.

Every function uses the exact registered definition and sign convention.
Functions refuse to produce a number where the registry says the result is
`NOT_MEANINGFUL` (terminal growth >= WACC, non-positive diluted shares,
ROE<=0 for a justified P/E) rather than returning a misleading value.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import brentq

from wbj.core.nullstates import NullState, Value
from wbj.schemas.valuation import (
    DcfResult,
    EnsembleResult,
    MonteCarloResult,
    ScenarioResult,
    ScenarioValue,
)

# Damodaran synthetic-rating spread table (large non-financial firms, cost of
# debt lookup by interest coverage). Source: A. Damodaran, "Ratings, Interest
# Coverage Ratios and Default Spread", NYU Stern (Jan-2024 dataset). Ordered
# high-coverage -> low; each row is (coverage_lower_bound, default_spread).
_SYNTHETIC_SPREADS: list[tuple[float, float]] = [
    (8.50, 0.0069),
    (6.50, 0.0082),
    (5.50, 0.0103),
    (4.25, 0.0114),
    (3.00, 0.0129),
    (2.50, 0.0159),
    (2.25, 0.0193),
    (2.00, 0.0215),
    (1.75, 0.0264),
    (1.50, 0.0318),
    (1.25, 0.0364),
    (0.80, 0.0455),
    (0.65, 0.0650),
    (0.20, 0.0940),
    (float("-inf"), 0.1233),
]


# --- normalization ----------------------------------------------------------


def normalized_ebit(
    reported: float,
    unusual_gains: float = 0.0,
    nonrecurring_charges: float = 0.0,
    misclassified: float = 0.0,
) -> float:
    """VAL-NORM-001 — reported EBIT less unusual gains, plus non-recurring
    charges added back and reclassification adjustments."""
    return reported - unusual_gains + nonrecurring_charges + misclassified


def rd_capitalize(rd_history: list[float], life: int) -> dict[str, float]:
    """VAL-RD-002/003 — capitalize R&D. `rd_history[j]` is R&D spent `j`
    years ago (index 0 = current year). Returns the R&D asset, this year's
    amortization, and the adjustment to add to EBIT (current R&D expensed
    back in, minus amortization)."""
    cohorts = rd_history[:life]
    asset = sum(rd * (1.0 - j / life) for j, rd in enumerate(cohorts))
    # Amortization of prior cohorts still on the books (exclude current year).
    amortization = sum(rd / life for rd in cohorts[1:])
    adjusted_ebit_delta = rd_history[0] - amortization if rd_history else 0.0
    return {
        "asset": asset,
        "amortization": amortization,
        "adjusted_ebit_delta": adjusted_ebit_delta,
    }


def lease_debt(commitments: list[float], pretax_kd: float) -> float:
    """VAL-LEASE-004 — PV of future operating-lease commitments discounted at
    the pre-tax cost of debt. `commitments[t]` is the payment in year t+1."""
    return sum(cf / (1.0 + pretax_kd) ** (t + 1) for t, cf in enumerate(commitments))


# --- ROIC / economic profit -------------------------------------------------


def nopat(norm_ebit: float, tax_rate: float) -> float:
    """NOPAT = normalized EBIT × (1 − tax rate)."""
    return norm_ebit * (1.0 - tax_rate)


def invested_capital(
    debt: float,
    equity: float,
    excess_cash: float = 0.0,
    debt_like: float = 0.0,
    operating_assets: float | None = None,
    operating_liabilities: float | None = None,
) -> Value:
    """Financing-view invested capital = Debt + Equity − Excess cash +
    Debt-like claims, reconciled to the operating view when provided (warns
    if the two views differ by more than 5%)."""
    financing = debt + equity - excess_cash + debt_like
    warnings: list[str] = []
    if operating_assets is not None and operating_liabilities is not None:
        operating = operating_assets - operating_liabilities
        denom = abs(financing) if financing != 0 else 1.0
        if abs(financing - operating) / denom > 0.05:
            warnings.append(
                f"invested capital views diverge >5%: financing={financing!r} "
                f"operating={operating!r}"
            )
    return Value.of(financing, unit="usd", warnings=warnings)


def roic(nopat_value: float, invested_capital_value: float) -> Value:
    """ROIC = NOPAT / invested capital; NOT_MEANINGFUL on zero capital."""
    if invested_capital_value == 0:
        return Value.null(NullState.NOT_MEANINGFUL, unit="ratio")
    return Value.of(nopat_value / invested_capital_value, unit="ratio")


def spread(roic_value: float, wacc_value: float) -> float:
    """ROIC value-creation spread = ROIC − WACC."""
    return roic_value - wacc_value


def eva(roic_value: float, wacc_value: float, ic_beginning: float) -> float:
    """Economic profit = (ROIC − WACC) × beginning invested capital."""
    return (roic_value - wacc_value) * ic_beginning


def incremental_roic(delta_nopat: float, delta_ic: float) -> Value:
    """Incremental ROIC = ΔNOPAT / ΔInvested capital."""
    if delta_ic == 0:
        return Value.null(NullState.NOT_MEANINGFUL, unit="ratio")
    return Value.of(delta_nopat / delta_ic, unit="ratio")


def fundamental_growth(reinvestment_rate: float, roic_value: float) -> float:
    """Sustainable growth = reinvestment rate × ROIC."""
    return reinvestment_rate * roic_value


# --- discount rate ----------------------------------------------------------


def unlever_beta(beta: float, tax_rate: float, de: float) -> float:
    """VAL-UBETA-009 — β_u = β / (1 + (1−tax)·D/E)."""
    return beta / (1.0 + (1.0 - tax_rate) * de)


def relever_beta(bu: float, tax_rate: float, target_de: float) -> float:
    """VAL-LBETA-010 — β_L = β_u · (1 + (1−tax)·D/E_target)."""
    return bu * (1.0 + (1.0 - tax_rate) * target_de)


def cost_of_equity(rf: float, beta: float, erp: float, crp: float = 0.0) -> float:
    """VAL-KE-008 — Ke = Rf + β·ERP + CRP."""
    return rf + beta * erp + crp


def synthetic_spread(interest_coverage: float) -> float:
    """Default spread for an interest-coverage ratio, per the synthetic
    rating table (Damodaran, Jan-2024)."""
    for lower, spread_value in _SYNTHETIC_SPREADS:
        if interest_coverage >= lower:
            return spread_value
    return _SYNTHETIC_SPREADS[-1][1]


def synthetic_kd(rf: float, interest_coverage: float) -> float:
    """VAL-KD-011 — pre-tax cost of debt = Rf + synthetic default spread."""
    return rf + synthetic_spread(interest_coverage)


def wacc(e: float, d: float, ke: float, kd: float, tax: float) -> float:
    """VAL-WACC-007 — E/(D+E)·Ke + D/(D+E)·Kd·(1−tax)."""
    total = e + d
    if total == 0:
        raise ValueError("wacc: zero capital base")
    return e / total * ke + d / total * kd * (1.0 - tax)


def wacc_sensitivity(w: float, bp: int = 100) -> dict[str, float]:
    """Base WACC plus ±`bp` basis points."""
    delta = bp / 10_000.0
    return {"base": w, "up": w + delta, "down": w - delta}


# --- FCFF DCF ---------------------------------------------------------------


def fcff(ebit: float, tax: float, dna: float, capex: float, dnwc: float) -> float:
    """VAL-FCFF-005 — EBIT·(1−tax) + D&A − Capex − ΔNWC."""
    return ebit * (1.0 - tax) + dna - capex - dnwc


def gordon_terminal_value(fcff_n: float, g: float, wacc_value: float) -> float:
    """VAL-TVG-012 — FCFF_N·(1+g)/(WACC−g). Requires g < WACC."""
    if g >= wacc_value:
        raise ValueError("gordon_terminal_value requires g < WACC")
    return fcff_n * (1.0 + g) / (wacc_value - g)


def dcf_value(fcffs: list[float], wacc_value: float, terminal_growth: float) -> DcfResult:
    """VAL-EV-014 — enterprise value from explicit FCFFs + Gordon terminal.

    Refuses (`ev = NOT_MEANINGFUL`) when terminal growth >= WACC; warns when
    the terminal value's share of EV exceeds 75%."""
    if terminal_growth >= wacc_value:
        return DcfResult(
            ev=Value.null(
                NullState.NOT_MEANINGFUL,
                unit="usd",
                warnings=["terminal growth >= WACC; DCF refused"],
            ),
            warnings=["terminal growth >= WACC; DCF refused"],
        )

    pv_explicit = sum(f / (1.0 + wacc_value) ** t for t, f in enumerate(fcffs, start=1))
    n = len(fcffs)
    tv = gordon_terminal_value(fcffs[-1], terminal_growth, wacc_value)
    pv_terminal = tv / (1.0 + wacc_value) ** n
    ev = pv_explicit + pv_terminal
    terminal_share = pv_terminal / ev if ev != 0 else None

    warnings: list[str] = []
    if terminal_share is not None and terminal_share > 0.75:
        warnings.append(
            f"terminal value is {terminal_share:.0%} of EV (>75%); high sensitivity"
        )

    return DcfResult(
        ev=Value.of(ev, unit="usd", warnings=warnings),
        pv_explicit=pv_explicit,
        pv_terminal=pv_terminal,
        terminal_share=terminal_share,
        warnings=warnings,
    )


def equity_bridge(
    ev: float,
    cash: float,
    nonop: float,
    debt: float,
    lease_debt_value: float = 0.0,
    preferred: float = 0.0,
    minority: float = 0.0,
    pension: float = 0.0,
) -> float:
    """VAL-EQ-015 — EV + Cash + Non-operating − Debt − Lease − Preferred −
    Minority − Pension/other debt-like claims."""
    return ev + cash + nonop - debt - lease_debt_value - preferred - minority - pension


def per_share(equity: float, diluted: float) -> Value:
    """VAL-PS-016 — equity value / fully-diluted shares."""
    if diluted <= 0:
        return Value.null(NullState.NOT_MEANINGFUL, unit="usd_per_share")
    return Value.of(equity / diluted, unit="usd_per_share")


# --- cross-checks -----------------------------------------------------------


def fcfe_value(fcfes: list[float], cost_equity: float, g: float) -> float:
    """VAL-FCFEV-018 — PV of explicit FCFE + terminal FCFE perpetuity."""
    if g >= cost_equity:
        raise ValueError("fcfe_value requires g < cost of equity")
    pv = sum(f / (1.0 + cost_equity) ** t for t, f in enumerate(fcfes, start=1))
    terminal = fcfes[-1] * (1.0 + g) / (cost_equity - g)
    pv += terminal / (1.0 + cost_equity) ** len(fcfes)
    return pv


def economic_profit_value(
    ic0: float, economic_profits: list[float], wacc_value: float, terminal_growth: float = 0.0
) -> float:
    """VAL-EVAEV-021 — IC0 + PV(economic profits) + terminal EP perpetuity.

    Reconciles to the FCFF DCF under consistent assumptions."""
    if terminal_growth >= wacc_value:
        raise ValueError("economic_profit_value requires terminal growth < WACC")
    pv = sum(ep / (1.0 + wacc_value) ** t for t, ep in enumerate(economic_profits, start=1))
    terminal = economic_profits[-1] * (1.0 + terminal_growth) / (wacc_value - terminal_growth)
    pv += terminal / (1.0 + wacc_value) ** len(economic_profits)
    return ic0 + pv


def residual_income_value(
    book0: float, residual_incomes: list[float], cost_equity: float, terminal_growth: float = 0.0
) -> float:
    """VAL-RIV-023 — Book equity_0 + PV(residual income) + terminal RI."""
    if terminal_growth >= cost_equity:
        raise ValueError("residual_income_value requires terminal growth < cost of equity")
    pv = sum(ri / (1.0 + cost_equity) ** t for t, ri in enumerate(residual_incomes, start=1))
    terminal = residual_incomes[-1] * (1.0 + terminal_growth) / (cost_equity - terminal_growth)
    pv += terminal / (1.0 + cost_equity) ** len(residual_incomes)
    return book0 + pv


def justified_pe(g: float, roe: float, ke: float) -> Value:
    """VAL-JPE-032 — (1 − g/ROE)/(Ke − g); NOT_MEANINGFUL if ROE<=0 or g>=Ke."""
    if roe <= 0 or g >= ke:
        return Value.null(NullState.NOT_MEANINGFUL, unit="ratio")
    return Value.of((1.0 - g / roe) / (ke - g), unit="ratio")


def justified_ev_sales(after_tax_margin: float, g: float, roic_value: float, wacc_value: float) -> Value:
    """VAL-JEVS-033 — after-tax operating margin·(1 − g/ROIC)/(WACC − g)."""
    if roic_value == 0 or g >= wacc_value:
        return Value.null(NullState.NOT_MEANINGFUL, unit="ratio")
    return Value.of(after_tax_margin * (1.0 - g / roic_value) / (wacc_value - g), unit="ratio")


def hist_zscore(current: float, history: list[float]) -> Value:
    """VAL-ZHIST-035 — (current − median)/(1.4826·MAD); robust to outliers."""
    arr = np.asarray(history, dtype=float)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    scale = 1.4826 * mad
    if scale == 0:
        return Value.null(NullState.NOT_MEANINGFUL, unit="zscore")
    return Value.of((current - median) / scale, unit="zscore")


# --- reverse DCF / scenarios / Monte Carlo / ensemble -----------------------


def reverse_dcf(
    price: float,
    per_share_of_growth: Callable[[float], float],
    bracket: tuple[float, float] = (-0.90, 1.00),
) -> Value:
    """VAL-RDCF-027 — implied growth such that modeled per-share value = price.

    `per_share_of_growth(g)` maps a growth assumption to a modeled per-share
    value. Solved with Brent's method over `bracket`. Returns the implied
    assumption (NOT a target); NOT_MEANINGFUL if the price is unreachable in
    the bracket."""

    def f(g: float) -> float:
        return per_share_of_growth(g) - price

    lo, hi = bracket
    try:
        implied = brentq(f, lo, hi, xtol=1e-8)
    except ValueError:
        return Value.null(NullState.NOT_MEANINGFUL, unit="ratio")
    return Value.of(implied, unit="ratio")


def scenarios(
    scenario_specs: list[dict], value_fn: Callable[[dict], float]
) -> ScenarioResult:
    """VAL-SCEN-036 — probability-weighted value across scenarios.

    Each spec is `{name, probability, ...drivers}`; `value_fn(spec)` returns
    that scenario's value. Probabilities must sum to 1.0."""
    total_p = sum(s["probability"] for s in scenario_specs)
    if abs(total_p - 1.0) > 1e-9:
        raise ValueError(f"scenario probabilities must sum to 1.0, got {total_p}")
    values = [
        ScenarioValue(
            name=s.get("name", f"scenario_{i}"),
            probability=s["probability"],
            value=value_fn(s),
        )
        for i, s in enumerate(scenario_specs)
    ]
    weighted = sum(v.probability * v.value for v in values)
    return ScenarioResult(scenarios=values, weighted=weighted)


def monte_carlo(
    value_fn: Callable[[float, float, float], float],
    params: dict[str, tuple[float, float, float]],
    n: int = 2000,
    seed: int = 0,
) -> MonteCarloResult:
    """VAL-MC-037 — Monte Carlo valuation over triangular driver draws.

    `params` gives `(low, mode, high)` triangular bounds for keys `growth`,
    `margin`, `wacc`. Reproducible for a given `seed`."""
    rng = np.random.default_rng(seed)
    draws = {
        key: rng.triangular(low, mode, high, size=n)
        for key, (low, mode, high) in params.items()
    }
    vals = np.array(
        [
            value_fn(draws["growth"][i], draws["margin"][i], draws["wacc"][i])
            for i in range(n)
        ]
    )
    return MonteCarloResult(
        p10=float(np.percentile(vals, 10)),
        p25=float(np.percentile(vals, 25)),
        median=float(np.percentile(vals, 50)),
        p75=float(np.percentile(vals, 75)),
        p90=float(np.percentile(vals, 90)),
        seed=seed,
        trials=n,
    )


def ensemble(model_values: list[tuple[float, float]]) -> EnsembleResult:
    """VAL-ENSEMBLE-044 — reliability-weighted value + weighted dispersion.

    `model_values` is a list of `(value, reliability_weight)`."""
    weights = np.array([w for _, w in model_values], dtype=float)
    values = np.array([v for v, _ in model_values], dtype=float)
    if weights.sum() == 0:
        raise ValueError("ensemble: total reliability weight is zero")
    value = float(np.average(values, weights=weights))
    dispersion = float(np.sqrt(np.average((values - value) ** 2, weights=weights)))
    return EnsembleResult(value=value, dispersion=dispersion)


def margin_of_safety(value: float, price: float) -> float:
    """VAL-MOS-040 — (Estimated value − Price)/Value; negative means overpriced."""
    if value == 0:
        raise ValueError("margin_of_safety: zero estimated value")
    return (value - price) / value
