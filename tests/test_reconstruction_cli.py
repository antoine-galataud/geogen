"""Roofer backend routing, replay and output protection."""

import json

import pytest
from click.testing import CliRunner

from geogen.cli import main
from geogen.models import BuildingGroup
from geogen.reconstruction.cache import (
    artifact,
    building_snapshot,
    content_hash,
    write_json,
)
from geogen.reconstruction.types import ReconstructionError


def test_roofer_rejects_mixed_country_before_provider_calls(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "geogen.cli._download_buildings", lambda *a, **k: pytest.fail("provider called")
    )
    output = tmp_path / "model.osm"
    result = CliRunner().invoke(
        main,
        [
            "1 rue Paris France",
            "10 Downing Street London UK",
            "--roof-source",
            "roofer",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code != 0
    assert "French units only" in result.output
    assert not output.exists()
    report = json.loads(output.with_suffix(".reconstruction.json").read_text())
    assert report["reason_codes"] == ["unsupported_country"]
    assert not report["outputs_published"]


def test_conflicting_outputs_never_overwrite_model(tmp_path):
    path = tmp_path / "model.osm"
    path.write_text("previous model")
    result = CliRunner().invoke(
        main,
        [
            "1 rue Paris",
            "--roof-source",
            "roofer",
            "--output",
            str(path),
            "--json-output",
            str(path),
        ],
    )
    assert result.exit_code != 0
    assert path.read_text() == "previous model"


def test_replay_restores_sources_without_provider_calls(monkeypatch, tmp_path):
    from geogen.bdnb import DEFAULT_BASE_URL
    from geogen.reconstruction.runtime import DockerRuntime

    request = {
        "addresses": ["1 rue Paris"],
        "country": "FR",
        "max_buildings": 10,
        "provider_endpoint_hash": content_hash(DEFAULT_BASE_URL),
    }
    original = tmp_path / "original.laz"
    original.write_bytes(b"checksummed input")
    manifest = write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": "1.0",
            "request": request,
            "buildings": building_snapshot({"FR:group": BuildingGroup("group")}),
            "artifacts": {"lidar_original": artifact(original, tmp_path)},
        },
    )
    monkeypatch.setattr(
        "geogen.cli._download_buildings", lambda *a, **k: pytest.fail("provider called")
    )

    def unavailable(self, *args):
        assert self.offline
        raise ReconstructionError("runtime_image_missing", "Image is not preloaded")

    monkeypatch.setattr(DockerRuntime, "preflight", unavailable)
    result = CliRunner().invoke(
        main,
        [
            "1 rue Paris",
            "--country",
            "FR",
            "--roof-source",
            "roofer",
            "--reconstruction-input",
            str(manifest),
            "--reconstruction-cache",
            str(tmp_path / "cache"),
            "--output",
            str(tmp_path / "model.osm"),
        ],
    )
    assert result.exit_code != 0
    assert "Image is not preloaded" in result.output
    report = json.loads((tmp_path / "model.reconstruction.json").read_text())
    assert report["benchmark_status"] == "not_evaluated_no_independent_references"
    assert "runtime_image_missing" in report["reason_codes"]
    assert not (tmp_path / "model.osm").exists()


def test_failure_report_cannot_overwrite_invalid_input_manifest(tmp_path):
    # The report is written beside the model, here exactly onto the replay input.
    manifest = tmp_path / "model.reconstruction.json"
    manifest.write_text("not valid JSON")
    result = CliRunner().invoke(
        main,
        [
            "1 rue Paris",
            "--roof-source",
            "roofer",
            "--reconstruction-input",
            str(manifest),
            "--output",
            str(tmp_path / "model.osm"),
        ],
    )
    assert result.exit_code != 0
    assert manifest.read_text() == "not valid JSON"
