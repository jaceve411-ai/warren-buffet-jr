"""Task 10 — the packet builder.

Assembles the analysis `Packet` handed to the six Cerebro specialists from
the raw provider payloads (FMP, EDGAR, FinnHub, FRED). This is where the
shared validation pipeline lands in code: canonical field naming,
FMP<->EDGAR source-hierarchy reconciliation (`wbj.packet.reconcile`),
per-data-type staleness classification (`wbj.packet.staleness`), the hard
data-quality gates, and a content hash that pins the exact inputs a report
was computed from.

Design (Task 10 brief):
- Providers are passed in as an already-constructed `Providers` bundle so
  this layer stays network-free and unit-testable with fake providers
  (see `engine/tests/fixtures/packet/make_packet_fixture.py`).
- `now` is an explicit parameter (the frozen analysis clock, Phase 0 of
  `Cerebro/00_main_agent/ORCHESTRATION.md`) — never `datetime.now()` — so
  packets and their staleness table are reproducible.
- Fundamentals are emitted under canonical snake_case names; raw provider
  keys (`netIncome`, `weightedAverageShsOutDil`, ...) never leak through.
- Hard rejects raise `PacketRejected`: missing reporting currency, no
  market-data timestamp at all, fewer than 252 daily sessions, or no
  diluted share count available from any source.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from wbj.core.nullstates import NullState, Value
from wbj.packet.reconcile import reconcile
from wbj.packet.staleness import staleness_state
from wbj.schemas.packet import AnalysisMeta, MarketData, OHLCVRow, Packet, Security

_MIN_DAILY_SESSIONS = 252


class PacketRejected(Exception):
    """Raised when assembled inputs fail a hard data-quality gate.

    The message names the failing gate (currency / timestamp / daily
    sessions / diluted share) so callers and tests can distinguish them.
    """


@dataclass
class Providers:
    """The four data providers a packet build draws on.

    Real `wbj.providers.*` instances in production; fakes that read fixture
    JSON in tests — only the public method surface matters here.
    """

    fmp: Any
    edgar: Any
    finnhub: Any
    fred: Any


# --- small payload helpers --------------------------------------------------


def _rows(payload: Any) -> list[dict]:
    """Normalize a provider payload to a list of dict rows."""
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _edgar_fact_latest(companyfacts: dict | None, taxonomy: str, tag: str) -> float | None:
    """Latest (max `end` date) numeric value for an XBRL tag, or None.

    Scans every unit list under `facts[taxonomy][tag]` and returns the value
    of the entry with the newest `end` date. Missing taxonomy/tag/unit or a
    malformed payload yields None (EDGAR simply didn't report it).
    """
    if not isinstance(companyfacts, dict):
        return None
    try:
        units = companyfacts["facts"][taxonomy][tag]["units"]
    except (KeyError, TypeError):
        return None

    best: tuple[str, float] | None = None
    for entries in units.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            end = entry.get("end")
            val = entry.get("val")
            if end is None or val is None:
                continue
            if best is None or end > best[0]:
                best = (end, val)
    return best[1] if best else None


def _edgar_total_debt(companyfacts: dict | None) -> float | None:
    """EDGAR total debt = long-term (noncurrent) + current debt.

    Returns whichever components are present summed, or None if neither is.
    """
    ltd = _edgar_fact_latest(companyfacts, "us-gaap", "LongTermDebtNoncurrent")
    current = _edgar_fact_latest(companyfacts, "us-gaap", "DebtCurrent")
    if ltd is None and current is None:
        return None
    return (ltd or 0.0) + (current or 0.0)


def _val(x: float | None, unit: str, source: str) -> Value:
    """Wrap a raw number in a lineage-carrying `Value`, or a MISSING null."""
    if x is None:
        return Value.null(NullState.MISSING, unit=unit, source_name=source)
    return Value.of(x, unit=unit, source_name=source)


def _age_days(now: datetime, iso_date: str) -> int:
    """Whole calendar days between `iso_date` and `now`'s date."""
    return (now.date() - date.fromisoformat(iso_date)).days


def _canonical_fundamentals(
    income: list[dict], balance: list[dict], cashflow: list[dict]
) -> list[dict]:
    """Merge income/balance/cashflow rows (aligned by period end date) into
    canonical snake_case records, one per fiscal period."""
    bal_by_date = {r.get("date"): r for r in balance}
    cf_by_date = {r.get("date"): r for r in cashflow}

    records: list[dict] = []
    for inc in income:
        d = inc.get("date")
        bal = bal_by_date.get(d, {})
        cf = cf_by_date.get(d, {})
        records.append(
            {
                "period_end": d,
                "fiscal_period": inc.get("period"),
                "revenue": inc.get("revenue"),
                "ebit": inc.get("operatingIncome"),
                "net_income": inc.get("netIncome"),
                "diluted_shares": inc.get("weightedAverageShsOutDil"),
                "operating_cash_flow": cf.get("netCashProvidedByOperatingActivities"),
                "capex": cf.get("capitalExpenditure"),
                "cash": bal.get("cashAndCashEquivalents"),
                "total_debt": bal.get("totalDebt"),
            }
        )
    return records


# --- the build --------------------------------------------------------------


def build_packet(ticker: str, providers: Providers, now: datetime) -> Packet:
    """Build the analysis `Packet` for `ticker` as of `now`.

    Raises `PacketRejected` if a hard data-quality gate fails.
    """
    fmp, edgar, finnhub, fred = (
        providers.fmp,
        providers.edgar,
        providers.finnhub,
        providers.fred,
    )

    # --- identity / currency (hard gate: currency) --------------------------
    profiles = _rows(fmp.profile(ticker))
    if not profiles:
        raise PacketRejected("packet rejected: no company profile available")
    profile = profiles[0]
    currency = profile.get("currency")
    if not currency:
        raise PacketRejected("packet rejected: missing reporting currency")

    security = Security(
        ticker=ticker,
        exchange=profile.get("exchangeShortName") or "",
        security_type="ETF" if profile.get("isEtf") else "EQUITY",
        reporting_currency=currency,
        valuation_currency=currency,
    )

    # --- market data (hard gates: timestamp, then >=252 sessions) -----------
    daily = [
        OHLCVRow(
            date=b["date"],
            open=b["open"],
            high=b["high"],
            low=b["low"],
            close=b["close"],
            adj_close=b.get("adjClose", b["close"]),
            volume=b["volume"],
        )
        for b in _rows(fmp.ohlcv_daily(ticker))
    ]

    market_timestamp: str | None = None
    if daily:
        market_timestamp = daily[0].date  # provider returns newest-first
    else:
        quote = finnhub.quote(ticker)
        if isinstance(quote, dict) and quote.get("t"):
            market_timestamp = datetime.fromtimestamp(
                quote["t"], tz=timezone.utc
            ).isoformat()
    if market_timestamp is None:
        raise PacketRejected(
            "packet rejected: no market-data timestamp available (no OHLCV, no quote)"
        )

    if len(daily) < _MIN_DAILY_SESSIONS:
        raise PacketRejected(
            f"packet rejected: only {len(daily)} daily sessions "
            f"(need >= {_MIN_DAILY_SESSIONS})"
        )

    # --- fundamentals (canonical names) -------------------------------------
    income_annual = _rows(fmp.income_annual(ticker))
    balance_annual = _rows(fmp.balance_annual(ticker))
    cashflow_annual = _rows(fmp.cashflow_annual(ticker))
    income_quarterly = _rows(fmp.income_quarterly(ticker))
    balance_quarterly = _rows(fmp.balance_quarterly(ticker))
    cashflow_quarterly = _rows(fmp.cashflow_quarterly(ticker))

    fundamentals = {
        "annual": _canonical_fundamentals(income_annual, balance_annual, cashflow_annual),
        "quarterly": _canonical_fundamentals(
            income_quarterly, balance_quarterly, cashflow_quarterly
        ),
    }

    # --- EDGAR facts for reconciliation -------------------------------------
    cik = edgar.cik_for(ticker)
    companyfacts = edgar.companyfacts(cik) if cik is not None else None

    edgar_revenue = _edgar_fact_latest(companyfacts, "us-gaap", "Revenues")
    edgar_cash = _edgar_fact_latest(
        companyfacts, "us-gaap", "CashAndCashEquivalentsAtCarryingValue"
    )
    edgar_total_debt = _edgar_total_debt(companyfacts)
    edgar_diluted = _edgar_fact_latest(
        companyfacts, "us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"
    )
    if edgar_diluted is None:
        # Fallback: basic outstanding count from the cover page (dei).
        edgar_diluted = _edgar_fact_latest(
            companyfacts, "dei", "EntityCommonStockSharesOutstanding"
        )

    # --- FMP latest-annual facts --------------------------------------------
    inc0 = income_annual[0] if income_annual else {}
    bal0 = balance_annual[0] if balance_annual else {}

    fmp_diluted = inc0.get("weightedAverageShsOutDil")
    if fmp_diluted is None and income_quarterly:
        fmp_diluted = income_quarterly[0].get("weightedAverageShsOutDil")

    # Hard gate: diluted share count must exist somewhere.
    if fmp_diluted is None and edgar_diluted is None:
        raise PacketRejected(
            "packet rejected: no diluted share count from any source"
        )

    # --- facts table (source-hierarchy reconciled) --------------------------
    facts_table: dict[str, Value] = {
        "revenue": reconcile(
            "revenue",
            _val(inc0.get("revenue"), "usd", "FMP"),
            _val(edgar_revenue, "usd", "EDGAR"),
        ),
        "diluted_shares": reconcile(
            "diluted_shares",
            _val(fmp_diluted, "shares", "FMP"),
            _val(edgar_diluted, "shares", "EDGAR"),
        ),
        "cash": reconcile(
            "cash",
            _val(bal0.get("cashAndCashEquivalents"), "usd", "FMP"),
            _val(edgar_cash, "usd", "EDGAR"),
        ),
        "total_debt": reconcile(
            "total_debt",
            _val(bal0.get("totalDebt"), "usd", "FMP"),
            _val(edgar_total_debt, "usd", "EDGAR"),
        ),
        # Market price is an FMP-only quote; EDGAR never carries it.
        "price": _val(profile.get("price"), "usd", "FMP"),
    }

    # --- staleness table ----------------------------------------------------
    staleness: dict[str, str] = {
        "daily_market": staleness_state("daily_market", _age_days(now, daily[0].date))
    }

    q_income_dates = [r["date"] for r in income_quarterly if r.get("date")]
    if q_income_dates:
        staleness["quarterly_fundamentals"] = staleness_state(
            "quarterly_fundamentals", _age_days(now, max(q_income_dates))
        )

    earnings = _rows(fmp.earnings_calendar(ticker))
    printed = [
        e["date"]
        for e in earnings
        if e.get("date")
        and e.get("eps") is not None
        and date.fromisoformat(e["date"]) <= now.date()
    ]
    if printed:
        staleness["consensus"] = staleness_state(
            "consensus", _age_days(now, max(printed))
        )

    holders = _rows(fmp.institutional_holders(ticker))
    holder_dates = [h["dateReported"] for h in holders if h.get("dateReported")]
    if holder_dates:
        staleness["peer_set"] = staleness_state(
            "peer_set", _age_days(now, max(holder_dates))
        )

    # --- supporting blocks --------------------------------------------------
    risk_free = fred.risk_free_rate()
    capital_structure = {
        "diluted_shares": facts_table["diluted_shares"].value,
        "total_debt": facts_table["total_debt"].value,
        "cash": facts_table["cash"].value,
        "market_cap": profile.get("mktCap"),
        "beta": profile.get("beta"),
        "risk_free_rate": risk_free.value if risk_free.is_valid else None,
    }

    estimates = {
        "analyst_estimates": fmp.analyst_estimates(ticker),
        "eps": finnhub.estimates(ticker),
        "revenue": finnhub.revenue_estimates(ticker),
        "earnings_calendar": earnings,
    }

    analysis = AnalysisMeta(
        knowledge_timestamp=now.isoformat(),
        market_timestamp=market_timestamp,
        industry_adapter=profile.get("industry") or profile.get("sector") or "default",
    )

    packet = Packet(
        security=security,
        analysis=analysis,
        fundamentals=fundamentals,
        market_data=MarketData(daily=daily),
        estimates=estimates,
        capital_structure=capital_structure,
        insiders=_rows(fmp.insider_trades(ticker)),
        institutional_holders=holders,
        facts_table=facts_table,
        staleness=staleness,
        packet_hash="",
    )
    packet.packet_hash = _compute_hash(packet)
    return packet


def _compute_hash(packet: Packet) -> str:
    """Deterministic sha256 over the packet's content, excluding the hash
    field itself. Stable for identical inputs; changes if any input does."""
    payload = packet.model_dump(mode="json")
    payload.pop("packet_hash", None)
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
