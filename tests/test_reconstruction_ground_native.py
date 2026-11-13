"""Real Roofer/OSM exports when ground lies beyond the initial 3 m search."""

import json
import os
import xml.etree.ElementTree as ET

import laspy
import numpy as np
import openstudio
import pytest
from shapely import contains_xy
from shapely.ops import unary_union

from geogen.reconstruction.cache import artifact, file_hash, write_json
from tests.test_reconstruction_integration import _invoke, _manifest

pytestmark = pytest.mark.skipif(
    os.environ.get("GEOGEN_RUNTIME_TESTS") != "1", reason="requires pinned Roofer Docker image"
)


@pytest.mark.parametrize("origin_z", [0, 100])
def test_expanded_ground_search_exports_and_rechecks_policy(tmp_path, offline_runtime, origin_z):
    manifest_path, original, shapes = _manifest(tmp_path / "source")
    cloud = laspy.read(original)
    near = unary_union([polygon.buffer(3) for polygon in shapes.values()])
    keep = (np.asarray(cloud.classification) != 2) | ~contains_xy(near, cloud.x, cloud.y)
    cloud.points = cloud.points[keep].copy()
    cloud.write(original)
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["lidar_original"] = artifact(original, manifest_path.parent)
    manifest["frame"]["origin"][2] = origin_z
    write_json(manifest_path, manifest)
    original_hash = file_hash(original)

    result, report, output, svg, metadata = _invoke(manifest_path, tmp_path, name="expanded")
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}\n{report}"
    assert report["outputs_published"]
    for selected in report["units"]:
        ground = report["ground_references"][selected["reconstruction_id"]]
        assert ground["search_radius_m"] == 6
        assert ground["elevation_m"] == 100
        assert ground["attempts"][0]["ground_point_count"] == 0
        assert ground["ground_point_count"] >= 3
        assert selected["requires_review"]
        assert "ground_search_expanded" in selected["reason_codes"]
        assert selected["independent_validation"]["validity"]
        assert selected["geometry_status"] == "accepted"
    model = openstudio.osversion.VersionTranslator().loadModel(openstudio.path(str(output))).get()
    assert len(model.getSpaces()) == 2
    assert sum(space.volume() for space in model.getSpaces()) == pytest.approx(4360)
    ground_z = [
        point.z()
        for surface in model.getSurfaces()
        if surface.outsideBoundaryCondition() == "Ground"
        for point in surface.vertices()
    ]
    assert all(z + model.getSite().elevation() == pytest.approx(100) for z in ground_z)
    assert ground_z
    geometry_hash = report["scene_geometry_hash"]
    assert ET.parse(svg).getroot().attrib["data-geometry-hash"] == geometry_hash
    assert (
        json.loads(metadata.read_text())["reconstruction"]["scene_geometry_hash"] == geometry_hash
    )

    result, replay, idf, _, _ = _invoke(manifest_path, tmp_path, name="replay", output_format="idf")
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    assert replay["scene_geometry_hash"] == geometry_hash
    assert openstudio.IdfFile.load(openstudio.path(str(idf))).is_initialized()
    assert file_hash(original) == original_hash
