"""Per-unit LoD1.2 fallback is reported and never bypasses envelope validity."""

import json

import pytest

from geogen.reconstruction.pipeline import roofer_diagnostics
from geogen.reconstruction.types import ReconstructionError
from tests.test_reconstruction_policy import (
    RAW,
    captured_reconstruction,
)


def lower_detail_reconstruction(*args, **kwargs):
    path = captured_reconstruction(*args, **kwargs)
    runtime = args[3]
    header, feature = [json.loads(line) for line in path.read_text().splitlines()]
    if feature["id"] == "box":
        geometries = feature["CityObjects"]["box-0"]["geometry"]
        case = getattr(runtime, "fallback_case", "nonplanar")
        if case == "extrusion":
            feature["CityObjects"]["box"]["attributes"]["rf_force_lod11"] = True
        elif case == "missing":
            geometries[:] = [g for g in geometries if g["lod"] != "2.2"]
        elif case == "nonplanar":
            high = next(g for g in geometries if g["lod"] == "2.2")
            # Lift one LoD2.2 roof corner by 1 m, far beyond any planarity tolerance.
            feature["vertices"].append([0, 0, 111000])
            roof = high["boundaries"][0][-1][0]
            roof[roof.index(11)] = len(feature["vertices"]) - 1
        else:
            high = next(g for g in geometries if g["lod"] == "2.2")
            # Simulate the reported collapsed Roofer ring without changing LoD1.2.
            ring = high["boundaries"][0][-1][0]
            high["boundaries"][0].append([[ring[0], ring[0], ring[0]]])
            high["semantics"]["values"][0].append(high["semantics"]["values"][0][-1])
        lower = next(g for g in geometries if g["lod"] == "1.2")
        if getattr(runtime, "bad_lower", False):
            lower["boundaries"][0][-1][0] = [0, 0, 0]
        if getattr(runtime, "no_lower", False):
            geometries.remove(lower)
        path.write_text(json.dumps(header) + "\n" + json.dumps(feature) + "\n")
    return path


@pytest.fixture
def lower_detail(reconstruction, monkeypatch):
    monkeypatch.setattr("geogen.reconstruction.roofer.reconstruct", lower_detail_reconstruction)
    return reconstruction


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("extrusion", "fallback_lod1"),
        ("nonplanar", "planarity_tolerance_exceeded"),
        ("missing", "lod_unavailable"),
    ],
)
def test_lower_detail_preserves_other_roofs(lower_detail, case, reason):
    run, runtime, manifest, evidence, reports = lower_detail
    runtime.fallback_case = case
    scene, _ = run()
    box, courtyard = reports[-1]["units"]
    assert reports[-1]["failed_units"] == []
    assert box["selected_lod"] == "1.2"
    assert box["geometry_status"] == "fallback_lod1"
    assert box["requires_review"]
    assert box["lod_fallback"]["reason"]["code"] == reason
    assert box["selected_candidate"] == 1
    assert box["attempted_candidates"][0]["failure"]["code"] == reason
    assert box["attempted_candidates"][0]["selection_reason"] == "rejected"
    assert box["attempted_candidates"][0]["artifact"]["sha256"]
    assert "roofer_lod12_fallback" in box["reason_codes"]
    assert box["independent_validation"]["validity"] is True
    assert "fallback-lod12.city.json" in runtime.validated
    assert all(
        "roofer_lod_fallback:1.2" in face.provenance
        for solid in scene.solids
        if solid.reconstruction_id == "box"
        for face in solid.surfaces
    )
    assert courtyard["selected_lod"] == "2.2"
    assert courtyard["geometry_status"] == "accepted"
    assert not courtyard["adaptation"]["changed"]
    assert "lod_fallback" not in courtyard
    assert len(courtyard["attempted_candidates"]) == 1


@pytest.mark.parametrize("defect", ["bad_lower", "no_lower"])
def test_unusable_lower_detail_still_fails(lower_detail, defect):
    run, runtime, manifest, evidence, reports = lower_detail
    setattr(runtime, defect, True)
    with pytest.raises(ReconstructionError):
        run()
    box = reports[-1]["units"][0]
    assert box["geometry_status"] == "geometry_invalid"
    assert box["attempted_candidates"][1]["selection_reason"] == "rejected"
    assert box["lod_fallback"]["reason"]["code"] == "planarity_tolerance_exceeded"
    assert "selected_lod" not in box


def test_collapsed_roofer_ring_no_longer_forces_lower_detail(lower_detail):
    """Walls are rebuilt from the source outline, so Roofer ring defects are irrelevant."""
    run, runtime, manifest, evidence, reports = lower_detail
    runtime.fallback_case = "degenerate"
    scene, _ = run()
    box = reports[-1]["units"][0]
    assert box["selected_lod"] == "2.2"
    assert box["geometry_status"] == "accepted"
    assert box["conformance"]["skipped_degenerate_roof_surfaces"] == 1
    assert box["independent_validation"]["validity"] is True


def test_lower_detail_requires_its_own_independent_validation(lower_detail, monkeypatch):
    run, runtime, manifest, evidence, reports = lower_detail
    validate = runtime.validate_cityjson

    def reject(path, directory):
        if path.name == "fallback-lod12.city.json":
            raise ReconstructionError("validator_invalid", "Invalid fallback")
        return validate(path, directory)

    monkeypatch.setattr(runtime, "validate_cityjson", reject)
    with pytest.raises(ReconstructionError):
        run()
    assert reports[-1]["units"][0]["failure"]["code"] == "validator_invalid"
    assert reports[-1]["failed_units"] == ["box"]


@pytest.mark.parametrize("code", ["validator_report_missing", "runtime_timeout"])
def test_validator_execution_errors_never_trigger_lower_detail(lower_detail, code):
    run, runtime, manifest, evidence, reports = lower_detail
    runtime.raw_error = code
    with pytest.raises(ReconstructionError):
        run()
    assert "fallback-lod12.city.json" not in runtime.validated
    assert all(unit["failure"]["code"] == code for unit in reports[-1]["units"])


@pytest.mark.parametrize("reason", ["roof_fills_courtyard", "roof_outline_mismatch"])
def test_lower_detail_does_not_bypass_outline_evidence(lower_detail, reason):
    run, runtime, manifest, evidence, reports = lower_detail
    evidence.update(fidelity_status="needs_review", reasons=[reason])
    with pytest.raises(ReconstructionError):
        run()
    box = reports[-1]["units"][0]
    assert box["failure"]["code"] == "needs_review"
    assert reason in box["reason_codes"]


@pytest.mark.parametrize("name", ["rf_success", "rf_force_lod11", "rf_pointcloud_unusable"])
@pytest.mark.parametrize("value", [None, "true", 0, 1])
def test_diagnostic_flags_remain_strict_booleans(tmp_path, name, value):
    header, feature, *_ = [json.loads(line) for line in RAW.read_text().splitlines()]
    feature["CityObjects"]["box"]["attributes"][name] = value
    path = tmp_path / "diagnostics.city.jsonl"
    path.write_text(json.dumps(header) + "\n" + json.dumps(feature) + "\n")
    diagnostics = {}
    with pytest.raises(ReconstructionError) as error:
        roofer_diagnostics(path, "box", diagnostics=diagnostics)
    assert error.value.code == "runtime_diagnostics_missing"
    assert f"{name} must be boolean" in str(error.value)
    assert diagnostics[name] == value


def test_missing_diagnostic_names_are_reported(tmp_path):
    header, feature, *_ = [json.loads(line) for line in RAW.read_text().splitlines()]
    del feature["CityObjects"]["box"]["attributes"]["rf_success"]
    path = tmp_path / "diagnostics.city.jsonl"
    path.write_text(json.dumps(header) + "\n" + json.dumps(feature) + "\n")
    diagnostics = {}
    with pytest.raises(ReconstructionError) as error:
        roofer_diagnostics(path, "box", diagnostics=diagnostics)
    assert error.value.code == "runtime_diagnostics_missing"
    assert "Missing Roofer diagnostics: rf_success" in str(error.value)
    assert diagnostics["rf_extrusion_mode"] == "standard"


@pytest.mark.parametrize(
    ("key", "value", "code"),
    [
        ("rf_success", False, "runtime_failed"),
        ("rf_pointcloud_unusable", True, "runtime_failed"),
        ("rf_force_lod11", "true", "runtime_diagnostics_missing"),
        ("rf_extrusion_mode", "unknown", "runtime_diagnostics_missing"),
        ("rf_extrusion_mode", None, "runtime_diagnostics_missing"),
        ("rf_extrusion_mode", [], "runtime_diagnostics_missing"),
    ],
)
def test_bad_diagnostics_never_trigger_lower_detail(lower_detail, monkeypatch, key, value, code):
    run, runtime, manifest, evidence, reports = lower_detail
    runtime.fallback_case = "extrusion"

    def bad_diagnostics(*args, **kwargs):
        path = lower_detail_reconstruction(*args, **kwargs)
        header, feature = [json.loads(line) for line in path.read_text().splitlines()]
        feature["CityObjects"][feature["id"]]["attributes"][key] = value
        path.write_text(json.dumps(header) + "\n" + json.dumps(feature) + "\n")
        return path

    monkeypatch.setattr("geogen.reconstruction.roofer.reconstruct", bad_diagnostics)
    with pytest.raises(ReconstructionError):
        run()
    assert all(unit["failure"]["code"] == code for unit in reports[-1]["units"])
    for unit in reports[-1]["units"]:
        assert unit["diagnostics"][key] == value
        candidate = unit["attempted_candidates"][0]
        assert candidate["artifact"]["sha256"]
        assert candidate["selection_reason"] == "rejected"
        assert candidate["failure"] == unit["failure"]
        if code == "runtime_diagnostics_missing":
            assert key in unit["failure"]["message"]
            assert repr(value) in unit["failure"]["message"]
    assert "fallback-lod12.city.json" not in runtime.validated


@pytest.mark.parametrize("mode", ["skip", "fail"])
def test_skipped_and_failed_runs_are_not_extrusion_fallbacks(lower_detail, monkeypatch, mode):
    run, runtime, manifest, evidence, reports = lower_detail
    runtime.fallback_case = "extrusion"

    def skipped_reconstruction(*args, **kwargs):
        path = lower_detail_reconstruction(*args, **kwargs)
        header, feature = [json.loads(line) for line in path.read_text().splitlines()]
        # Even success/usable flags and supplied meshes cannot override skip/fail.
        feature["CityObjects"][feature["id"]]["attributes"]["rf_extrusion_mode"] = mode
        path.write_text(json.dumps(header) + "\n" + json.dumps(feature) + "\n")
        return path

    monkeypatch.setattr("geogen.reconstruction.roofer.reconstruct", skipped_reconstruction)
    with pytest.raises(ReconstructionError):
        run()
    for unit in reports[-1]["units"]:
        assert unit["diagnostics"]["rf_extrusion_mode"] == mode
        assert unit["failure"]["code"] == "runtime_failed"
        assert f"rf_extrusion_mode={mode!r}" in unit["failure"]["message"]
        assert "fallback_lod1" not in unit["reason_codes"]
        assert "lod_fallback" not in unit
        assert unit["geometry_status"] == "runtime_failed"
        assert unit["attempted_candidates"][0]["artifact"]["sha256"]
    assert runtime.validated == []
