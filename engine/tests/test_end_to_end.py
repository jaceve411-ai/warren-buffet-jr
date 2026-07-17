"""End-to-end offline pipeline test + golden-report stability (Task 24)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wbj.config import Settings
from wbj.pipeline import run_all

_GOLDEN = Path(__file__).parent / "fixtures" / "golden" / "NVDA_report.json"


def _settings_for(tmp_path: Path) -> Settings:
    return Settings(
        repo_root=tmp_path,
        cache_dir=tmp_path / "cache",
        reports_dir=tmp_path / "Reportes",
    )


def _normalize(report: dict) -> dict:
    """Strip volatile fields (timestamps, hashes) for golden comparison."""
    report = json.loads(json.dumps(report))  # deep copy
    report["knowledge_timestamp"] = None
    report["audit"] = {}
    for row in report.get("category_scorecard", []):
        pass
    return report


def test_analyze_offline_end_to_end(tmp_path):
    settings = _settings_for(tmp_path)
    final = run_all("NVDA", settings, offline=True)

    assert final.report_version == "2.0.0"
    assert 0 <= final.profile.raw_score <= 100

    out = tmp_path / "Reportes" / "NVDA"
    day_dir = next(out.iterdir())
    assert (day_dir / "report.md").exists()
    assert (day_dir / "report.json").exists()
    assert len(list((day_dir / "charts").iterdir())) >= 3


def test_golden_report_stable(tmp_path):
    settings = _settings_for(tmp_path)
    final = run_all("NVDA", settings, offline=True)
    assert _GOLDEN.exists(), "golden report fixture missing; regenerate it"
    golden = json.loads(_GOLDEN.read_text())
    assert _normalize(final.model_dump(mode="json")) == _normalize(golden)


def test_artifacts_written(tmp_path):
    settings = _settings_for(tmp_path)
    run_all("NVDA", settings, offline=True)
    artifacts = tmp_path / "cache" / "NVDA" / "artifacts"
    assert (artifacts / "financial.json").exists()
    assert (artifacts / "judgment_requests.json").exists()
    assert len(list(artifacts.glob("*.json"))) == 7  # 6 specialists + requests
