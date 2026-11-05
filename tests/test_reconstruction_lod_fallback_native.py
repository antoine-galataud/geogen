"""Real val3dity, Roofer and OpenStudio gates for mixed-detail envelopes."""

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import laspy
import openstudio
import pytest

from geogen.reconstruction.cache import artifact
from geogen.reconstruction.runtime import DockerRuntime
from tests.test_reconstruction_integration import _invoke, _manifest
from tests.test_reconstruction_lod_fallback import lower_detail_reconstruction
from tests.test_reconstruction_policy import EXAMPLE

pytestmark = pytest.mark.skipif(
    os.environ.get("GEOGEN_RUNTIME_TESTS") != "1", reason="requires pinned Roofer Docker image"
)


def space_volume(output):
    model = openstudio.osversion.VersionTranslator().loadModel(openstudio.path(str(output))).get()
    return [space.volume() for space in model.getSpaces()]


@pytest.mark.parametrize("case", ["extrusion", "degenerate"])
def test_mixed_detail_exports_and_strict_replay(monkeypatch, tmp_path, offline_runtime, case):
    monkeypatch.setattr(DockerRuntime, "fallback_case", case, raising=False)
    monkeypatch.setattr("geogen.reconstruction.roofer.reconstruct", lower_detail_reconstruction)
    result, report, output, svg, metadata = _invoke(EXAMPLE, tmp_path, name=case)
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    assert report["outputs_published"]
    box, courtyard = report["units"]
    assert box["independent_validation"]["validity"] is True
    assert courtyard["selected_lod"] == "2.2"
    assert courtyard["geometry_status"] == "accepted"
    if case == "degenerate":
        # Walls are rebuilt from the outline, so a collapsed Roofer ring keeps LoD2.2.
        assert box["raw_validation"]["validity"] is False
        assert box["selected_lod"] == "2.2"
        assert box["conformance"]["skipped_degenerate_roof_surfaces"] == 1
        assert sum(space_volume(output)) == pytest.approx(4360)
        return
    assert "1 unit(s) use the LoD1.2 fallback" in result.output
    assert box["selected_lod"] == "1.2"
    assert box["geometry_status"] == "fallback_lod1"
    document = json.loads(
        (Path(report["replay_directory"]) / box["adapted_artifact"]["path"]).read_text()
    )
    assert {
        g["lod"] for obj in document["CityObjects"].values() for g in obj.get("geometry", [])
    } == {"1.2"}
    model = openstudio.osversion.VersionTranslator().loadModel(openstudio.path(str(output))).get()
    assert len(model.getSpaces()) == 2
    assert sum(space.volume() for space in model.getSpaces()) == pytest.approx(4360)
    scene_hash = report["scene_geometry_hash"]
    assert ET.parse(svg).getroot().attrib["data-geometry-hash"] == scene_hash
    assert json.loads(metadata.read_text())["reconstruction"]["scene_geometry_hash"] == scene_hash
    result, replay, idf, _, _ = _invoke(
        EXAMPLE, tmp_path, name=f"{case}-idf", output_format="idf"
    )
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    assert replay["scene_geometry_hash"] == scene_hash
    assert openstudio.IdfFile.load(openstudio.path(str(idf))).is_initialized()
    assert output.exists() and idf.exists()


def test_actual_roofer_extrusion_exports_lod12(tmp_path, offline_runtime):
    manifest, _, _ = _manifest(tmp_path / "input", oversized=True)
    result, report, output, _, _ = _invoke(manifest, tmp_path, name="actual-extrusion")
    assert result.exit_code == 0, f"{result.output}\n{json.dumps(report, indent=2)}"
    unit = report["units"][0]
    assert unit["geometry_status"] == "fallback_lod1"
    assert unit["selected_lod"] == "1.2"
    assert unit["diagnostics"]["rf_force_lod11"] or (
        unit["diagnostics"]["rf_extrusion_mode"] != "standard"
    )
    assert unit["independent_validation"]["validity"] is True
    assert (
        openstudio.osversion.VersionTranslator()
        .loadModel(openstudio.path(str(output)))
        .is_initialized()
    )


def test_actual_roofer_skip_is_reported_without_publishing(tmp_path, offline_runtime):
    manifest, original, _ = _manifest(tmp_path / "input")
    cloud = laspy.read(original)
    cloud.classification[cloud.classification == 6] = 1
    cloud.write(original)
    document = json.loads(manifest.read_text())
    document["artifacts"]["lidar_original"] = artifact(original, manifest.parent)
    for building in document["buildings"]:
        building["data"]["height"] = None
    manifest.write_text(json.dumps(document))
    result, report, output, svg, metadata = _invoke(manifest, tmp_path, name="actual-skip")
    assert result.exit_code != 0
    assert report["outputs_published"] is False
    assert not any(path.exists() for path in (output, svg, metadata))
    assert report["failed_units"] == ["box", "courtyard"]
    for unit in report["units"]:
        assert unit["diagnostics"]["rf_extrusion_mode"] == "skip"
        assert unit["diagnostics"]["rf_success"] is True
        assert unit["diagnostics"]["rf_pointcloud_unusable"] is True
        assert unit["failure"]["code"] == "runtime_failed"
        assert "rf_extrusion_mode='skip'" in unit["failure"]["message"]
        assert unit["geometry_status"] == "runtime_failed"
        assert "fallback_lod1" not in unit["reason_codes"]
        assert "lod_fallback" not in unit
        candidate = unit["attempted_candidates"][0]
        assert candidate["selection_reason"] == "rejected"
        assert candidate["failure"] == unit["failure"]
        raw = Path(report["replay_directory"]) / candidate["artifact"]["path"]
        assert raw.is_file()
        assert candidate["artifact"]["sha256"]
