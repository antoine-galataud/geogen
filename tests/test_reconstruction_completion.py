"""The explicit assumed-height completion keeps strict checks and honest provenance."""

import json
import math

import pytest

from geogen.reconstruction.types import ReconstructionError
from tests.test_reconstruction_policy import captured_reconstruction


def assumed_height_reconstruction(points, outline, directory, runtime, **kwargs):
    properties = json.loads(outline.read_text())["features"][0]["properties"]
    height = properties.get("roof_elevation")
    runtime.reconstruction_calls.append((properties["reconstruction_id"], height))
    path = captured_reconstruction(points, outline, directory, runtime, **kwargs)
    header, feature = [json.loads(line) for line in path.read_text().splitlines()]
    if feature["id"] == "box":
        attributes = feature["CityObjects"]["box"]["attributes"]
        attributes.update(
            rf_extrusion_mode="skip" if height is None else "lod11_fallback",
            rf_pointcloud_unusable=True,
        )
        if height is not None:
            if getattr(runtime, "retry_mode", None):
                attributes["rf_extrusion_mode"] = runtime.retry_mode
            if getattr(runtime, "retry_failed", False):
                attributes["rf_success"] = False
            z = 110 if getattr(runtime, "wrong_height", False) else height
            for vertex in feature["vertices"]:
                if vertex[2] == 110000:
                    vertex[2] = round(z * 1000)
            if getattr(runtime, "invalid_retry", False):
                for geometry in feature["CityObjects"]["box-0"]["geometry"]:
                    geometry["boundaries"][0][-1][0] = [0, 0, 0]
        path.write_text(json.dumps(header) + "\n" + json.dumps(feature) + "\n")
    return path


@pytest.fixture
def completion(reconstruction, monkeypatch):
    reconstruction[1].reconstruction_calls = []
    monkeypatch.setattr("geogen.reconstruction.roofer.reconstruct", assumed_height_reconstruction)
    return reconstruction


def test_assumed_height_is_per_failed_unit_and_cache_is_policy_specific(completion):
    run, runtime, manifest, evidence, reports = completion
    for building in manifest["buildings"]:
        building["data"]["height"] = None
    with pytest.raises(ReconstructionError):
        run()
    scene, _ = run(missing_roof_height=12)
    assumed, original = reports[-1]["units"]
    assert runtime.reconstruction_calls == [("box", None), ("courtyard", None), ("box", 112)]
    assert original["selected_lod"] == "2.2"
    assert "height_fallback" not in original
    assert assumed["geometry_status"] == "fallback_lod1"
    assert assumed["selected_lod"] == "1.2"
    assert assumed["height_fallback"]["height_m"] == 12
    assert assumed["height_fallback"]["ground_elevation_m"] == 100
    assert assumed["height_fallback"]["roof_elevation_m"] == 112
    assert assumed["height_fallback"]["original_diagnostics"]["rf_extrusion_mode"] == "skip"
    assert assumed["fidelity_status"] == "needs_review"  # Even if mocked evidence passes.
    assert assumed["requires_review"]
    assert "user_assumed_roof_height" in assumed["reason_codes"]
    assert assumed["attempted_candidates"][0]["selection_reason"] == "rejected"
    assert assumed["selected_candidate"] == 1
    assert assumed["independent_validation"]["validity"]
    assert "fallback-lod12.city.json" in runtime.validated
    assert scene.solids[0].volume == pytest.approx(1200)
    assert all("user_assumed_roof_height" in face.provenance for face in scene.solids[0].surfaces)
    run(missing_roof_height=12)
    assert len(runtime.reconstruction_calls) == 3
    scene, _ = run(missing_roof_height=14)
    assert runtime.reconstruction_calls[-1] == ("box", 114)
    assert scene.solids[0].volume == pytest.approx(1400)
    with pytest.raises(ReconstructionError):
        run()
    assert reports[-1]["failed_units"] == ["box"]


def test_bdnb_height_is_default_assumed_height(completion):
    run, runtime, manifest, evidence, reports = completion
    scene, _ = run()
    assumed, original = reports[-1]["units"]
    assert runtime.reconstruction_calls == [("box", None), ("box", 110), ("courtyard", None)]
    assert "height_fallback" not in original
    assert assumed["height_fallback"]["source"] == "bdnb"
    assert assumed["height_fallback"]["height_m"] == 10
    assert assumed["height_fallback"]["roof_elevation_m"] == 110
    assert assumed["fidelity_status"] == "needs_review"
    assert "bdnb_assumed_roof_height" in assumed["reason_codes"]
    assert all("bdnb_assumed_roof_height" in face.provenance for face in scene.solids[0].surfaces)
    run(missing_roof_height=12)
    assert runtime.reconstruction_calls[-1] == ("box", 112)
    assert reports[-1]["units"][0]["height_fallback"]["source"] == "user"


@pytest.mark.parametrize("defect", ["wrong_height", "invalid_retry", "retry_failed"])
def test_bad_assumed_height_reconstruction_is_rejected(completion, defect):
    run, runtime, manifest, evidence, reports = completion
    setattr(runtime, defect, True)
    with pytest.raises(ReconstructionError):
        run(missing_roof_height=12)
    assert reports[-1]["failed_units"] == ["box"]
    assert len(reports[-1]["units"][0]["attempted_candidates"]) == 2
    assert reports[-1]["units"][0]["attempted_candidates"][-1]["selection_reason"] == "rejected"


@pytest.mark.parametrize("mode", ["skip", "fail", "standard", "unknown"])
def test_height_retry_requires_roofer_extrusion_mode(completion, mode):
    run, runtime, manifest, evidence, reports = completion
    runtime.retry_mode = mode
    with pytest.raises(ReconstructionError):
        run(missing_roof_height=12)
    assert reports[-1]["failed_units"] == ["box"]


def test_assumed_height_retains_independent_validator_gate(completion, monkeypatch):
    run, runtime, manifest, evidence, reports = completion
    validate = runtime.validate_cityjson

    def reject(path, directory):
        if path.name == "fallback-lod12.city.json":
            raise ReconstructionError("validator_invalid", "Invalid fallback")
        return validate(path, directory)

    monkeypatch.setattr(runtime, "validate_cityjson", reject)
    with pytest.raises(ReconstructionError):
        run(missing_roof_height=12)
    assert reports[-1]["units"][0]["failure"]["code"] == "validator_invalid"


def test_height_does_not_retry_runtime_failures(completion, monkeypatch):
    run, runtime, manifest, evidence, reports = completion

    def failed(*args, **kwargs):
        raise ReconstructionError("runtime_timeout", "Timed out")

    monkeypatch.setattr("geogen.reconstruction.roofer.reconstruct", failed)
    with pytest.raises(ReconstructionError):
        run(missing_roof_height=12)
    assert all(unit["failure"]["code"] == "runtime_timeout" for unit in reports[-1]["units"])
    assert all("height_fallback" not in unit for unit in reports[-1]["units"])


@pytest.mark.parametrize("value", [0, -1, 301, math.nan, math.inf])
def test_invalid_assumed_height_is_rejected(completion, value):
    with pytest.raises(ReconstructionError) as error:
        completion[0](missing_roof_height=value)
    assert error.value.code == "configuration_invalid"
