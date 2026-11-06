"""Independent solid validation and native exports for a non-Delaunay courtyard."""

import os
import xml.etree.ElementTree as ET

import openstudio
import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

from geogen.osm import build_model_from_geometry, save_model_from_geometry
from geogen.reconstruction.adapt import adapt_scene, write_scene_cityjson
from geogen.reconstruction.cityjson import import_cityjson
from geogen.reconstruction.runtime import DockerRuntime
from geogen.reconstruction.thermal import make_thermal_scene, validate_thermal_scene
from geogen.svg import save_svg_from_geometry
from tests.test_reconstruction_triangulation import CONCAVE, OUTER, courtyard_scene

pytestmark = pytest.mark.skipif(
    os.environ.get("GEOGEN_RUNTIME_TESTS") != "1", reason="requires pinned val3dity Docker image"
)


def test_concave_courtyard_survives_containment_partition_and_exports(tmp_path, offline_runtime):
    scene, audit = adapt_scene(courtyard_scene())
    assert not audit["changed"]
    raw = write_scene_cityjson(scene, tmp_path / "courtyard.city.json", lod="1.2")
    assert DockerRuntime(offline=True).validate_cityjson(raw, tmp_path)["validity"] is True
    scene = import_cityjson(raw, scene.frame, lod="1.2")
    thermal = make_thermal_scene(scene)
    validate_thermal_scene(thermal)
    model = build_model_from_geometry(thermal, window_to_wall_ratio=0)
    osm = save_model_from_geometry(model, tmp_path / "model.osm")
    idf = save_model_from_geometry(model, tmp_path / "model.idf", output_format="idf")
    loaded = openstudio.osversion.VersionTranslator().loadModel(openstudio.path(str(osm))).get()
    assert openstudio.IdfFile.load(openstudio.path(str(idf))).is_initialized()
    assert len(loaded.getSpaces()) == 2
    expected = (Polygon(OUTER).area - Polygon(CONCAVE).area + 0.25) * 6
    assert sum(space.volume() for space in loaded.getSpaces()) == pytest.approx(expected)
    roofs = [
        Polygon([(p.x(), p.y()) for p in surface.vertices()])
        for surface in loaded.getSurfaces()
        if surface.surfaceType() == "RoofCeiling"
        and surface.outsideBoundaryCondition() == "Outdoors"
    ]
    roofprint = unary_union(roofs)
    expected_roof = Polygon(OUTER, (CONCAVE,)).union(
        Polygon(((-1, 0), (-0.5, 0), (-0.5, 0.5), (-1, 0.5)))
    )
    assert roofprint.symmetric_difference(expected_roof).area == pytest.approx(0, abs=1e-8)
    svg = save_svg_from_geometry(thermal, tmp_path / "model.svg")
    assert ET.parse(svg).getroot().attrib["data-geometry-hash"] == thermal.geometry_hash
