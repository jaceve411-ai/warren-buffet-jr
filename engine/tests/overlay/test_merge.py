"""Tests for the judgment overlay (Task 20)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wbj.core.nullstates import EvidenceClass
from wbj.overlay.merge import collect_requests, merge_overlay
from wbj.schemas.overlay import Judgment
from wbj.schemas.packet import Packet
from wbj.specialists import business
from wbj.specialists.common import compute_output_hash

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "packet" / "NVDA_packet.json"


@pytest.fixture(scope="module")
def business_output():
    packet = Packet.model_validate(json.loads(_FIXTURE.read_text()))
    return business.run(packet)


def test_collect_requests_finds_moat(business_output):
    reqs = collect_requests([business_output])
    assert any(r.request_id == "BUS-moat-classification" for r in reqs)


def test_merge_increases_points_and_changes_hash(business_output):
    before_hash = compute_output_hash(business_output)
    before_points = business_output.category.awarded_points
    before_moat = next(d for d in business_output.dimensions if d.name == "moat").score_10

    judgment = Judgment(
        request_id="BUS-moat-classification",
        answer={"class": "wide", "effects": ["scale", "switching costs"]},
        evidence_class=EvidenceClass.Q,
        source="analyst:VG",
        rationale="Wide moat with two quantitative effects.",
        score_10=9.0,
        target_dimension="moat",
    )
    merged = merge_overlay([business_output], [judgment])[0]

    after_moat = next(d for d in merged.dimensions if d.name == "moat").score_10
    assert after_moat == 9.0
    assert after_moat > before_moat
    assert merged.category.awarded_points > before_points
    assert merged.output_hash != before_hash
    # The answered request is removed.
    assert all(r.request_id != "BUS-moat-classification" for r in merged.judgment_requests)


def test_unknown_request_raises(business_output):
    j = Judgment(request_id="NOPE", answer=1.0, evidence_class=EvidenceClass.Q, source="x")
    with pytest.raises(ValueError, match="unknown judgment"):
        merge_overlay([business_output], [j])


def test_missing_source_rejected(business_output):
    j = Judgment(request_id="BUS-moat-classification", answer=1.0,
                 evidence_class=EvidenceClass.Q, source="")
    with pytest.raises(ValueError, match="missing evidence_class/source"):
        merge_overlay([business_output], [j])
