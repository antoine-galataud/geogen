"""Real Roofer retries with an explicit assumed roof height."""

import json
import os
import xml.etree.ElementTree as ET

import laspy
import openstudio
import pytest
from shapely import from_wkt
from shapely.geometry import Polygon

from geogen.reconstruction.cache import artifact, content_hash
from geogen.reconstruction.roofer import reconstruct
from tests.test_reconstruction_integration import _invoke, _manifest

pytestmark = pytest.mark.skipif(
    os.environ.get("GEOGEN_RUNTIME_TESTS") != "1", reason="requires pinned Roofer Docker image"
)


@pytest.mark.parametrize("invalid_courtyard_lod22", [False, True])
def test_native_completion_preserves_measured_roofs_and_rechecks_policy(
    tmp_path, monkeypatch, offline_runtime, invalid_courtyard_lod22
):
    if invalid_courtyard_lod22:

        def with_degenerate_lod22(*args, **kwargs):
            path = reconstruct(*args, **kwargs)
            header, feature = [json.loads(line) for line in path.read_text().splitlines()]
            if feature["id"] == "courtyard":
                geometry = next(
                    g
                    for obj in feature["CityObjects"].values()
                    for g in obj.get("geometry", [])
                    if g["lod"] == "2.2"
                )
                ring = geometry["boundaries"][0][-1][0]
                geometry["boundaries"][0].append([[ring[0]] * 3])
                geometry["semantics"]["values"][0].append(geometry["semantics"]["values"][0][-1])
                path.write_text(json.dumps(header) + "\n" + json.dumps(feature) + "\n")
            return path

        monkeypatch.setattr("geogen.reconstruction.roofer.reconstruct", with_degenerate_lod22)
    manifest, original, _ = _manifest(tmp_path / "input")
    cloud = laspy.read(original)
    mask = (cloud.classification == 6) & (cloud.x < 700015)
    cloud.classification[mask] = 1
    cloud.write(original)
    document = json.loads(manifest.read_text())
    document["artifacts"]["lidar_original"] = artifact(original, manifest.parent)
    court = document["units"][1]
    outline = from_wkt(court["working_wkt"])
    hole = Polygon(outline.interiors[0]).buffer(0.002, join_style="mitre")
    outline = Polygon(outline.exterior.coords, [hole.exterior.coords])
    court["original_wkt"] = court["working_wkt"] = outline.wkt
    court["geometry_revision"] = content_hash(outline.wkt)
    manifest.write_text(json.dumps(document))
    options = ("--roofer-missing-roof-height", "12")
    result, report, output, svg, metadata = _invoke(
        manifest, tmp_path, name="completion", extra_args=options
    )
    assert result.exit_code == 0, f"{result.output}\n{json.dumps(report, indent=2)}"
    assert report["outputs_published"]
    assert report["geometry_policy"]["missing_roof_height_m"] == 12
    box, court = report["units"]
    assert box["height_fallback"]["original_diagnostics"]["rf_extrusion_mode"] == "skip"
    assert box["diagnostics"]["rf_extrusion_mode"] == "lod11_fallback"
    assert box["diagnostics"]["rf_pointcloud_unusable"] is True
    assert box["selected_lod"] == "1.2"
    assert box["fidelity_status"] == "needs_review"
    assert "user_assumed_roof_height" in box["reason_codes"]
    assert box["metrics"]["gross_volume_m3"] == pytest.approx(1200)
    assert box["independent_validation"]["validity"] is True
    assert "assumed (user or BDNB) roof height" in result.output
    # Walls and ground are rebuilt from the outline, so a collapsed Roofer ring is skipped.
    assert court["selected_lod"] == "2.2"
    assert court["conformance"]["skipped_degenerate_roof_surfaces"] == int(invalid_courtyard_lod22)
    assert "height_fallback" not in court
    assert court["independent_validation"]["validity"] is True
    model = openstudio.osversion.VersionTranslator().loadModel(openstudio.path(str(output))).get()
    assert len(model.getSpaces()) == 2
    assert sum(space.volume() for space in model.getSpaces()) == pytest.approx(4560, abs=1)
    scene_hash = report["scene_geometry_hash"]
    assert ET.parse(svg).getroot().attrib["data-geometry-hash"] == scene_hash
    assert json.loads(metadata.read_text())["reconstruction"]["scene_geometry_hash"] == scene_hash
    reconstruction_calls = sum("--id-attribute" in call["args"] for call in offline_runtime)
    assert reconstruction_calls == 3
    result, replay, idf, _, _ = _invoke(
        manifest, tmp_path, name="completion-idf", output_format="idf", extra_args=options
    )
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    assert replay["scene_geometry_hash"] == scene_hash
    assert openstudio.IdfFile.load(openstudio.path(str(idf))).is_initialized()
    assert sum("--id-attribute" in call["args"] for call in offline_runtime) == reconstruction_calls
    result, default, _, _, _ = _invoke(manifest, tmp_path, name="bdnb-height")
    assert result.exit_code == 0, f"{result.output}\n{json.dumps(default, indent=2)}"
    assert default["units"][0]["height_fallback"]["source"] == "bdnb"
    assert "bdnb_assumed_roof_height" in default["units"][0]["reason_codes"]
    document = json.loads(manifest.read_text())
    for building in document["buildings"]:
        building["data"]["height"] = None
    manifest.write_text(json.dumps(document))
    result, strict, rejected, _, _ = _invoke(manifest, tmp_path, name="no-height")
    assert result.exit_code != 0
    assert not strict["outputs_published"]
    assert not rejected.exists()
    assert strict["failed_units"] == ["box"]
    assert output.exists() and idf.exists()
