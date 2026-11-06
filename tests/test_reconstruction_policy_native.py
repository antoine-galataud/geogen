"""Native validation/export of bounded adaptation using frozen Roofer output."""

import json
import os
import xml.etree.ElementTree as ET

import openstudio
import pytest

from geogen.reconstruction.runtime import DockerRuntime
from tests.test_reconstruction_integration import _invoke
from tests.test_reconstruction_policy import EXAMPLE, captured_reconstruction

pytestmark = pytest.mark.skipif(
    os.environ.get("GEOGEN_RUNTIME_TESTS") != "1", reason="requires pinned Roofer Docker image"
)


def test_warped_roofer_roof_exports_osm_and_idf_with_reported_adaptation(
    monkeypatch, tmp_path, offline_runtime
):
    monkeypatch.setattr(DockerRuntime, "warp", True, raising=False)
    monkeypatch.setattr(DockerRuntime, "warp_mm", 80, raising=False)
    monkeypatch.setattr("geogen.reconstruction.roofer.reconstruct", captured_reconstruction)
    result, report, output, svg, metadata = _invoke(EXAMPLE, tmp_path, name="adapted")
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    box = next(unit for unit in report["units"] if unit["reconstruction_id"] == "box")
    assert box["raw_validation"]["validity"] is False
    assert box["independent_validation"]["validity"] is True
    # The warp is absorbed by the fitted roof plane over the source outline.
    assert box["conformance"]["max_roof_plane_residual_m"] > 0
    omissions = [
        omission
        for space in report["glazing_export"].values()
        for omission in space["omitted_windows"]
    ]
    assert omissions
    assert all(item["reason"] == "requested_windows_not_representable" for item in omissions)
    assert "Requested windows omitted" in result.output
    model = openstudio.osversion.VersionTranslator().loadModel(openstudio.path(str(output))).get()
    assert len(model.getSpaces()) == 2
    assert 4360 < sum(space.volume() for space in model.getSpaces()) < 4370
    scene_hash = report["scene_geometry_hash"]
    assert ET.parse(svg).getroot().attrib["data-geometry-hash"] == scene_hash
    assert json.loads(metadata.read_text())["reconstruction"]["scene_geometry_hash"] == scene_hash
    result, replay, idf, _, _ = _invoke(
        EXAMPLE, tmp_path, name="adapted-idf", output_format="idf"
    )
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    assert replay["scene_geometry_hash"] == scene_hash
    assert openstudio.IdfFile.load(openstudio.path(str(idf))).is_initialized()
    assert output.exists() and idf.exists()


@pytest.mark.parametrize("separation", [0, 0.0005])
def test_consecutive_roofer_vertices_are_repaired_and_revalidated(
    monkeypatch, tmp_path, offline_runtime, separation
):
    def with_short_edge(*args, **kwargs):
        path = captured_reconstruction(*args, **kwargs)
        header, feature = [json.loads(line) for line in path.read_text().splitlines()]
        if feature["id"] == "box":
            # A finer integer grid expresses sub-millimetre duplicate coordinates.
            header["transform"]["scale"] = [0.0001] * 3
            feature["vertices"] = [[10 * v for v in vertex] for vertex in feature["vertices"]]
            geometry = next(
                geometry
                for obj in feature["CityObjects"].values()
                for geometry in obj.get("geometry", [])
                if geometry["lod"] == "2.2"
            )
            roof = next(
                face
                for face, semantic in zip(
                    geometry["boundaries"][0], geometry["semantics"]["values"][0]
                )
                if geometry["semantics"]["surfaces"][semantic]["type"] == "RoofSurface"
            )[0]
            a, b = (feature["vertices"][index] for index in roof[:2])
            length = sum((y - x) ** 2 for x, y in zip(a, b)) ** 0.5
            point = [round(x + (y - x) / length * separation / 0.0001) for x, y in zip(a, b)]
            roof.insert(1, len(feature["vertices"]))
            feature["vertices"].append(point)
            path.write_text(json.dumps(header) + "\n" + json.dumps(feature) + "\n")
        return path

    monkeypatch.setattr("geogen.reconstruction.roofer.reconstruct", with_short_edge)
    result, report, output, svg, metadata = _invoke(EXAMPLE, tmp_path, name="cleaned")
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    box = next(unit for unit in report["units"] if unit["reconstruction_id"] == "box")
    assert box["raw_validation"]["validity"] is False
    raw_report = next(
        call["result"]
        for call in offline_runtime
        if call["args"][0] == "val3dity" and "captured.city.jsonl" in call["args"][1]
    )
    assert "102" in raw_report.stdout
    assert box["independent_validation"]["validity"] is True
    # The roof ring is rebuilt over the source outline, which drops the short edge.
    assert box["conformance"]["method"] == "roof_planes_over_source_footprint"
    assert box["adaptation"]["removed_consecutive_vertex_count"] == 0
    assert box["geometry_status"] == "accepted"
    assert report["outputs_published"]
    model = openstudio.osversion.VersionTranslator().loadModel(openstudio.path(str(output))).get()
    assert sum(space.volume() for space in model.getSpaces()) == pytest.approx(4360)
    assert ET.parse(svg).getroot().attrib["data-geometry-hash"] == report["scene_geometry_hash"]
    assert (
        json.loads(metadata.read_text())["reconstruction"]["scene_geometry_hash"]
        == report["scene_geometry_hash"]
    )
    result, replay, idf, _, _ = _invoke(EXAMPLE, tmp_path, name="cleaned-idf", output_format="idf")
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    assert replay["scene_geometry_hash"] == report["scene_geometry_hash"]
    assert openstudio.IdfFile.load(openstudio.path(str(idf))).is_initialized()
