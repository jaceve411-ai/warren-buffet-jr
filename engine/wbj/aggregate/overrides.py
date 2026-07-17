"""Mandatory overrides (Task 21), per SCORING_AND_GATES.md.

Overrides are derived from the frozen specialist outputs and the packet.
They never mutate a score — they emit `Override` records that the gate layer
and renderer honor.
"""

from __future__ import annotations

from wbj.schemas.final_report import Override
from wbj.schemas.packet import Packet
from wbj.specialists.common import SpecialistOutput

_MIN_COVERAGE = 0.70


def apply_overrides(outputs: dict[str, SpecialistOutput], packet: Packet) -> list[Override]:
    """Return every triggered override for the six category outputs."""
    overrides: list[Override] = []
    fin = outputs.get("financial")
    biz = outputs.get("business")
    risk = outputs.get("risk")
    tech = outputs.get("technical")
    val = outputs.get("valuation")

    # 1. Capital dependence -> cap Avoid/Speculative.
    if fin is not None and "OVERRIDE_1_CAPITAL_DEPENDENCE" in getattr(fin, "mandatory_overrides", []):
        overrides.append(Override(override_id="OVR-1", condition="loss + negative FCF + external financing",
                                  action="CAP_AVOID_SPECULATIVE", note="Company depends on external capital."))

    # 2. ROIC<WACC / value destruction -> no Elite/Quality.
    if biz is not None and ("VALUE_DESTRUCTION" in biz.mandatory_flags):
        overrides.append(Override(override_id="OVR-2", condition="ROIC<WACC",
                                  action="NO_ELITE_QUALITY", note="Returns below cost of capital."))

    # 3. Coverage<1.5x -> warning (solvency).
    if fin is not None and "SOLVENCY_WARNING" in fin.mandatory_flags:
        overrides.append(Override(override_id="OVR-3", condition="interest coverage <1.5x",
                                  action="WARNING", note="Operating earnings do not cover interest comfortably."))

    # 4. Risk <=4/15 -> cap Speculative.
    if risk is not None and risk.category.awarded_points <= 4.0:
        overrides.append(Override(override_id="OVR-4", condition="risk <=4/15",
                                  action="CAP_SPECULATIVE", note="Risk score in the fragile band."))

    # 5. Valuation <=4 AND Technical <=8 -> Wait/Avoid.
    if val is not None and tech is not None and val.category.awarded_points <= 4.0 and tech.category.awarded_points <= 8.0:
        overrides.append(Override(override_id="OVR-5", condition="valuation<=4 and technical<=8",
                                  action="WAIT_AVOID", note="Expensive and technically weak."))

    # 6. Any category coverage <0.70 -> gate-ineligible.
    for name, out in outputs.items():
        if out is not None and out.coverage < _MIN_COVERAGE:
            overrides.append(Override(override_id="OVR-6", condition=f"{name} coverage<0.70",
                                      action="GATE_INELIGIBLE", note=f"{name} coverage {out.coverage:.2f}."))

    # 7. Unresolved facts-table conflict -> suppress per-share.
    conflicted = [k for k, v in packet.facts_table.items() if v.is_null and v.state and v.state.value == "CONFLICTED"]
    if conflicted:
        overrides.append(Override(override_id="OVR-7", condition="facts-table conflict",
                                  action="SUPPRESS_PER_SHARE", note=f"Conflicted facts: {conflicted}."))

    return overrides
