"""Concavities must be decomposed consistently for containment and export."""

from dataclasses import replace

import pytest
import shapely
from shapely.errors import GEOSException, UnsupportedGEOSVersionError
from shapely.geometry import GeometryCollection, Polygon
from shapely.ops import triangulate, unary_union

from geogen.reconstruction.adapt import adapt_scene, scene_geometry_hash
from geogen.reconstruction.thermal import make_thermal_scene
from geogen.reconstruction.types import GeometryScene, ReconstructionError
from geogen.reconstruction.validate import (
    _SolidIndex,
    face_plane,
    face_triangles,
    project,
    surface_pieces,
    validate_scene,
)
from tests.test_reconstruction_adaptation import FRAME, prism

# Unconstrained Delaunay loses 0.3849 m² from this valid five-vertex concavity.
CONCAVE = ((0.22, 5.45), (-3.6, 2.44), (0.7, -2.13), (1.36, -6.44), (1.24, -4.49))
OUTER = ((-10, -10), (10, -10), (10, 10), (-10, 10))


def courtyard_scene():
    court = prism(name="courtyard", outer=OUTER, holes=(CONCAVE[::-1],))
    pavilion = prism(name="pavilion", outer=((-1, 0), (-0.5, 0), (-0.5, 0.5), (-1, 0.5)))
    return GeometryScene(FRAME, (court, pavilion))


@pytest.mark.parametrize("courtyard", [False, True])
@pytest.mark.parametrize("scale", [1, 10, 100])
def test_shared_triangulation_covers_concavities_using_exact_source_vertices(courtyard, scale):
    solid = prism(outer=OUTER, holes=(CONCAVE[::-1],)) if courtyard else prism(outer=CONCAVE)
    source = solid.surfaces[1]
    face = replace(
        source,
        exterior=tuple(tuple(scale * v for v in p) for p in source.exterior),
        interiors=tuple(tuple(tuple(scale * v for v in p) for p in r) for r in source.interiors),
    )
    plane = face_plane(face)
    polygon = plane[4]
    old = [part for part in triangulate(polygon) if polygon.covers(part)]
    assert polygon.symmetric_difference(unary_union(old)).area > 0.1 * scale**2
    triangles = face_triangles(face)
    tiles = [Polygon([project(p, plane) for p in ring]) for ring in triangles]
    assert all(len(ring) == 3 for ring in triangles)
    assert all(tile.exterior.is_ccw for tile in tiles)
    assert polygon.symmetric_difference(unary_union(tiles)).area == pytest.approx(0, abs=1e-10)
    assert sum(tile.area for tile in tiles) == pytest.approx(polygon.area, rel=1e-14)
    assert {p for ring in triangles for p in ring} == {
        p for ring in (face.exterior, *face.interiors) for p in ring
    }
    pieces = surface_pieces(face)
    if courtyard:
        assert [p.exterior for p in pieces] == triangles
        assert all(not p.interiors for p in pieces)
        assert all(p.semantic == face.semantic and p.provenance == face.provenance for p in pieces)
    else:
        assert pieces == (face,)  # Preserve planar, hole-free export faces.


def test_courtyard_containment_and_thermal_scene_keep_the_accepted_envelope():
    raw = courtyard_scene()
    checked, audit = adapt_scene(raw)
    assert not audit["changed"]
    assert scene_geometry_hash(checked) == scene_geometry_hash(raw)
    court = checked.solids[0]
    index = _SolidIndex(court.surfaces)
    assert index.contains((-0.75, 0.25, 3)) is False
    assert index.contains((8, 8, 3)) is True
    assert index.contains((11, 8, 3)) is False
    assert index.contains((10, 8, 3)) is None
    thermal = make_thermal_scene(checked)
    expected = (Polygon(OUTER).area - Polygon(CONCAVE).area + 0.25) * 6
    assert sum(space.volume for space in thermal.spaces) == pytest.approx(expected)
    assert any(face.interiors for space in thermal.spaces for face in space.surfaces)
    # Correct containment must still reject an overlapping solid inside the building.
    overlapping = prism(name="overlap", outer=((7, 7), (8, 7), (8, 8), (7, 8)))
    with pytest.raises(ReconstructionError):
        validate_scene(GeometryScene(FRAME, (court, overlapping)))


def test_warped_concavity_uses_the_same_fallback_without_flattening():
    raw = GeometryScene(FRAME, (prism(outer=CONCAVE, warp=0.02),))
    output, audit = adapt_scene(raw)
    assert audit["changed"]
    assert audit["triangulated_face_count"] == 1
    assert audit["max_vertex_movement_m"] == 0
    assert {p for face in output.solids[0].surfaces for p in face.exterior} == {
        p for face in raw.solids[0].surfaces for p in face.exterior
    }


@pytest.mark.parametrize("problem", ["gap", "overlap", "new_vertex"])
@pytest.mark.parametrize("method", [face_triangles, surface_pieces])
def test_invalid_constrained_output_remains_blocking(monkeypatch, problem, method):
    face = prism(outer=OUTER, holes=(CONCAVE[::-1],)).surfaces[1]
    constrained = shapely.constrained_delaunay_triangles

    def invalid(polygon):
        parts = list(constrained(polygon).geoms)
        if problem == "gap":
            parts.pop()
        elif problem == "overlap":
            parts.append(parts[0])
        else:
            a, b, c, _ = parts.pop().exterior.coords
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            parts.extend([Polygon((a, mid, c)), Polygon((mid, b, c))])
        return GeometryCollection(parts)

    monkeypatch.setattr(shapely, "constrained_delaunay_triangles", invalid)
    with pytest.raises(ReconstructionError) as error:
        method(face)
    assert error.value.code == "unsupported_geometry"


@pytest.mark.parametrize("failure", [GEOSException, UnsupportedGEOSVersionError])
def test_constrained_backend_errors_are_reported(monkeypatch, failure):
    def failed(polygon):
        raise failure("Backend failure")

    monkeypatch.setattr(shapely, "constrained_delaunay_triangles", failed)
    with pytest.raises(ReconstructionError, match="Backend failure") as error:
        face_triangles(prism(outer=CONCAVE).surfaces[1])
    assert error.value.code == "unsupported_geometry"


def test_older_shapely_retains_simple_faces_and_explains_missing_capability(monkeypatch):
    monkeypatch.delattr(shapely, "constrained_delaunay_triangles")
    assert len(face_triangles(prism().surfaces[1])) == 2
    with pytest.raises(ReconstructionError, match="Shapely >= 2.1"):
        face_triangles(prism(outer=CONCAVE).surfaces[1])


def test_valid_unconstrained_output_keeps_the_existing_route(monkeypatch):
    def unexpected(polygon):
        raise AssertionError("No fallback is necessary for this face")

    monkeypatch.setattr(shapely, "constrained_delaunay_triangles", unexpected)
    face = prism().surfaces[1]
    assert len(face_triangles(face)) == 2
    assert surface_pieces(face) == (face,)
