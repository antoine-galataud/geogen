"""Roofer roofs are rebuilt over the exact source footprint without slivers."""

import math

import pytest
from shapely.geometry import Polygon

from geogen.reconstruction.conform import MIN_FEATURE, conform_scene
from geogen.reconstruction.types import (
    GeometryFrame,
    GeometryScene,
    ReconstructionError,
    Solid,
    Surface,
)
from geogen.reconstruction.validate import node_surfaces, surface_pieces, validate_scene

FRAME = GeometryFrame("EPSG:2154", "NGF-IGN69", (0.0, 0.0, 0.0))
FOOTPRINT = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


def roof(face_id, corners, z):
    return Surface(
        face_id,
        tuple((x, y, z(x, y) if callable(z) else z) for x, y in corners),
        semantic="RoofSurface",
        source_object_id="roofer",
        source_face_id=face_id,
    )


def roofer_scene(*roofs):
    return GeometryScene(FRAME, (Solid("unit-0", "unit", tuple(roofs)),))


def rectangle(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def millimetre_slivers():
    """Roofer parts that overlap, leave gaps and end millimetres apart."""
    return roofer_scene(
        roof("low-south", rectangle(0, -0.01, 5.004, 5.0), 10.0),
        roof("low-north", rectangle(0.003, 5.006, 5.004, 10.01), 10.0),
        roof("high", rectangle(4.998, 0.004, 10.02, 10.0), 12.0),
    )


def vertices_2d(scene):
    return {p[:2] for solid in scene.solids for s in solid.surfaces for p in s.exterior}


def test_millimetre_slivers_are_collapsed_and_footprint_is_exact():
    scene, audit = conform_scene(millimetre_slivers(), FOOTPRINT, 0.0)
    assert audit["merged_nodes"] + audit["moved_nodes"] > 0
    assert audit["min_feature_m"] == MIN_FEATURE
    points = sorted(vertices_2d(scene))
    assert all(
        math.dist(a, b) >= MIN_FEATURE for i, a in enumerate(points) for b in points[i + 1 :]
    )
    assert set(FOOTPRINT.exterior.coords[:-1]) <= set(points)
    solid = validate_scene(scene).solids[0]
    assert solid.volume == pytest.approx(5 * 10 * 10 + 5 * 10 * 12, abs=0.5)
    ground = next(s for s in scene.solids[0].surfaces if s.semantic == "GroundSurface")
    assert Polygon([p[:2] for p in ground.exterior]).equals(FOOTPRINT)


def test_conformed_solid_is_enclosed_for_openstudio():
    openstudio = pytest.importorskip("openstudio")
    scene, _ = conform_scene(millimetre_slivers(), FOOTPRINT, 0.0)
    pieces = node_surfaces(
        [tile for s in validate_scene(scene).solids[0].surfaces for tile in surface_pieces(s)]
    )
    surfaces = openstudio.Surface3dVector()
    for index, piece in enumerate(pieces):
        ring = openstudio.Point3dVector([openstudio.Point3d(*p) for p in piece.exterior])
        surfaces.append(openstudio.Surface3d(ring, piece.face_id, index))
    assert openstudio.Polyhedron(surfaces).isEnclosedVolume()


def test_collapsed_roof_rings_are_skipped():
    scene, audit = conform_scene(
        roofer_scene(
            roof("roof", rectangle(0, 0, 10, 10), 10.0),
            roof("collapsed", [(5, 5), (5, 5), (5, 5)], 10.0),
        ),
        FOOTPRINT,
        0.0,
    )
    assert audit["skipped_degenerate_roof_surfaces"] == 1
    assert validate_scene(scene).solids[0].volume == pytest.approx(1000)


def test_tolerance_bounds_roof_planarity_and_level_merging():
    def warped(x, y):
        return 10.0 + (0.03 if (x, y) == (10, 10) else 0.0)

    warped_scene = roofer_scene(roof("roof", rectangle(0, 0, 10, 10), warped))
    with pytest.raises(ReconstructionError) as error:
        conform_scene(warped_scene, FOOTPRINT, 0.0, tolerance=0.001)
    assert error.value.code == "planarity_tolerance_exceeded"
    _, audit = conform_scene(warped_scene, FOOTPRINT, 0.0, tolerance=0.05)
    assert audit["step_tolerance_m"] == 0.05

    steps = roofer_scene(
        roof("a", rectangle(0, 0, 5, 10), 10.0), roof("b", rectangle(5, 0, 10, 10), 10.03)
    )
    _, merged = conform_scene(steps, FOOTPRINT, 0.0, tolerance=0.05)
    _, stepped = conform_scene(steps, FOOTPRINT, 0.0, tolerance=0.01)
    assert merged["step_wall_surfaces"] == 0
    assert stepped["step_wall_surfaces"] == 1

