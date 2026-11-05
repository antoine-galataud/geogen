"""Acceptance policies never relax closed-envelope or export validation."""

import copy
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from geogen.cli import main
from geogen.reconstruction.types import ReconstructionError

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "roofer" / "manifest.json"
RAW = ROOT / "tests" / "fixtures" / "reconstruction" / "roofer-box-courtyard.city.jsonl"


def captured_reconstruction(points, outline, directory, runtime, **kwargs):
    """Use frozen Roofer output, optionally warping one shared roof corner by 2 cm."""
    unit = json.loads(outline.read_text())["features"][0]["properties"]["reconstruction_id"]
    header, *features = [json.loads(line) for line in RAW.read_text().splitlines()]
    feature = next(item for item in features if item["id"] == unit)
    if getattr(runtime, "warp", False) and unit == "box":
        for vertex in feature["vertices"]:
            if vertex == [10000, 10000, 110000]:
                vertex[2] += getattr(runtime, "warp_mm", 20)
    path = directory / "captured.city.jsonl"
    path.write_text(json.dumps(header) + "\n" + json.dumps(feature) + "\n")
    return path


class Validator:
    image = "test-runtime"
    warp = False
    raw_error = None
    adapted_error = None

    def __init__(self):
        self.validated = []

    def validate_cityjson(self, path, directory):
        self.validated.append(path.name)
        # Rebuilt candidates (LoD2.2 or LoD1.2) are .city.json; raw Roofer output is .jsonl.
        error = self.adapted_error if path.name.endswith(".city.json") else self.raw_error
        if error:
            raise ReconstructionError(error, "Rejected by test validator")
        return {"validity": True, "report_path": str(path)}





def test_evidence_review_is_reported_not_relabelled(reconstruction):
    run, runtime, manifest, evidence, reports = reconstruction
    evidence.update(fidelity_status="needs_review", reasons=["roof_residual_threshold_exceeded"])
    scene, units = run()
    assert len(scene.solids) == 2
    for unit in reports[-1]["units"]:
        assert unit["geometry_status"] == "accepted"
        assert unit["fidelity_status"] == "needs_review"
        assert unit["requires_review"]
        assert "evidence_review_permitted" in unit["reason_codes"]
        assert not unit["adaptation"]["changed"]


@pytest.mark.parametrize(
    "reason",
    [
        "roof_fills_courtyard",
        "ground_fills_courtyard",
        "roof_outline_mismatch",
        "ground_outline_mismatch",
        "missing_roof_surfaces",
        "missing_ground_surfaces",
        "contradictory_lidar_vertical_crs",
        "surface_computation_limit",
        "unknown_future_failure",
    ],
)
def test_geometry_and_reference_evidence_failures_remain_blocking(reconstruction, reason):
    run, runtime, manifest, evidence, reports = reconstruction
    evidence.update(fidelity_status="needs_review", reasons=[reason])
    with pytest.raises(ReconstructionError):
        run()
    assert reports[-1]["failed_units"] == ["box", "courtyard"]
    assert all(reason in unit["reason_codes"] for unit in reports[-1]["units"])


def test_adapted_geometry_is_independently_validated(reconstruction):
    run, runtime, manifest, evidence, reports = reconstruction
    runtime.warp = True
    scene, units = run()
    box = reports[-1]["units"][0]
    # The 2 cm warp is absorbed by the fitted roof plane, within the 5 cm default.
    assert box["conformance"]["max_roof_plane_residual_m"] > 0.001
    assert box["adapted_artifact"]["sha256"]
    assert "adapted.city.json" in runtime.validated
    assert scene.solids[0].volume > 1000
    runtime.adapted_error = "validator_invalid"
    with pytest.raises(ReconstructionError):
        run()
    assert reports[-1]["units"][0]["geometry_status"] == "geometry_invalid"
    assert "validator_invalid" in reports[-1]["units"][0]["reason_codes"]


def test_raw_validation_failure_requires_a_validated_rebuild(reconstruction):
    run, runtime, manifest, evidence, reports = reconstruction
    runtime.raw_error = "validator_invalid"
    runtime.warp = True
    # Raw Roofer validity is diagnostic: every unit is rebuilt and validated itself.
    run()
    for unit in reports[-1]["units"]:
        assert unit["raw_validation"]["validity"] is False
        assert unit["independent_validation"]["validity"] is True
        assert unit["geometry_status"] == "accepted"
    assert runtime.validated.count("adapted.city.json") == 2
    runtime.adapted_error = "validator_invalid"
    with pytest.raises(ReconstructionError):
        run()
    assert reports[-1]["failed_units"] == ["box", "courtyard"]


def test_validator_execution_failure_is_not_repaired(reconstruction):
    run, runtime, manifest, evidence, reports = reconstruction
    runtime.warp = True
    runtime.raw_error = "validator_report_missing"
    with pytest.raises(ReconstructionError):
        run()
    assert "adapted.city.json" not in runtime.validated
    assert all("validator_report_missing" in unit["reason_codes"] for unit in reports[-1]["units"])


def test_unmodeled_context_contact_is_a_reported_boundary_assumption(reconstruction):
    run, runtime, manifest, evidence, reports = reconstruction
    context = copy.deepcopy(manifest["units"][0])
    context.update(
        reconstruction_id="context",
        role="context",
        working_wkt="POLYGON ((700010 6600000,700020 6600000,700020 6600010,700010 6600010,700010 6600000))",
    )
    manifest["units"].append(context)
    run()
    box = reports[-1]["units"][0]
    assert box["context_contacts"] == ["context"]
    assert box["requires_review"]
    assert "unmodeled_context_contact_treated_as_exterior_geometry" in box["assumptions"]


def test_cli_exports_reviewed_geometry_without_thermal_inputs(
    reconstruction, monkeypatch, tmp_path
):
    from geogen.reconstruction.runtime import DockerRuntime

    run, runtime, manifest, evidence, reports = reconstruction
    evidence.update(fidelity_status="needs_review", reasons=["insufficient_roof_density"])
    monkeypatch.setattr(DockerRuntime, "preflight", lambda *a: {"image": "test"})
    monkeypatch.setattr(DockerRuntime, "validate_cityjson", lambda *a: {"validity": True})
    output = tmp_path / "model.osm"
    args = [
        "Synthetic reconstruction test, France",
        "--country",
        "FR",
        "--base-url",
        "https://api.bdnb.io/v1/bdnb",
        "--roof-source",
        "roofer",
        "--reconstruction-input",
        str(EXAMPLE),
        "--reconstruction-cache",
        str(tmp_path / "cache"),
        "--output",
        str(output),
    ]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    assert "2 unit(s) require review" in result.output
    report = json.loads(output.with_suffix(".reconstruction.json").read_text())
    assert report["outputs_published"]
    assert report["geometry_policy"] == {"missing_roof_height_m": None}
    assert all(unit["thermal_status"] == "geometry_only" for unit in report["units"])
    original = output.read_bytes()
    # Geometry-level evidence failures stay blocking and never replace published outputs.
    evidence.update(reasons=["roof_outline_mismatch"])
    result = CliRunner().invoke(main, args)
    assert result.exit_code != 0
    assert output.read_bytes() == original
    report = json.loads(output.with_suffix(".reconstruction.json").read_text())
    assert report["outputs_published"] is False


def test_omitted_partial_windows_are_removed_and_reported(monkeypatch, tmp_path):
    from geogen.osm import build_model_from_geometry, osmodel, save_model_from_geometry
    from geogen.reconstruction.thermal import make_thermal_scene
    from geogen.reconstruction.types import GeometryScene
    from tests.test_reconstruction_adaptation import FRAME, prism

    scene = make_thermal_scene(GeometryScene(FRAME, (prism(),)))
    native = osmodel.Surface.setWindowToWallRatio
    monkeypatch.setattr(
        osmodel.Surface, "setWindowToWallRatio", lambda wall, ratio: native(wall, ratio / 2)
    )
    model = build_model_from_geometry(scene, window_to_wall_ratio=0.2)
    assert len(model.getSubSurfaces()) == 0
    properties = model.getBuilding().additionalProperties()
    assert properties.getFeatureAsString("geogen_geometry_hash").get() == scene.geometry_hash
    report = json.loads(properties.getFeatureAsString("geogen_glazing_report").get())
    assert sum(len(item["omitted_windows"]) for item in report.values()) == 4
    assert all(item["achieved_window_area_m2"] == 0 for item in report.values())
    assert save_model_from_geometry(model, tmp_path / "no-windows.osm").exists()


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "0", "301"])
def test_invalid_assumed_height_rejected_before_acquisition(monkeypatch, tmp_path, value):
    monkeypatch.setattr(
        "geogen.cli._download_buildings", lambda *a, **k: pytest.fail("provider called")
    )
    output = tmp_path / "model.osm"
    result = CliRunner().invoke(
        main,
        [
            "1 rue Paris France",
            "--roof-source",
            "roofer",
            "--roofer-missing-roof-height",
            value,
            "--output",
            str(output),
        ],
    )
    assert result.exit_code != 0
    assert not output.exists()
    if output.with_suffix(".reconstruction.json").exists():
        report = json.loads(output.with_suffix(".reconstruction.json").read_text())
        assert report["reason_codes"] == ["configuration_invalid"]


@pytest.mark.parametrize(
    "args",
    [
        ["--roofer-missing-roof-height", "12"],
        ["--reconstruction-input", str(EXAMPLE)],
    ],
)
def test_roofer_options_require_roofer_source(monkeypatch, args):
    monkeypatch.setattr(
        "geogen.cli._download_buildings", lambda *a, **k: pytest.fail("provider called")
    )
    result = CliRunner().invoke(main, ["1 rue Paris France", *args])
    assert result.exit_code != 0
    assert "require --roof-source roofer" in result.output

