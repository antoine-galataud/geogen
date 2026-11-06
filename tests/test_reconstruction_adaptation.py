"""Explicit geometry adaptation and independent-validation artifact contracts."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import replace

import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

from geogen.reconstruction.adapt import adapt_scene, scene_geometry_hash, write_scene_cityjson
from geogen.reconstruction.cityjson import import_cityjson
from geogen.reconstruction.types import (
    GeometryFrame,
    GeometryScene,
    ReconstructionError,
    Solid,
    Surface,
)
from geogen.reconstruction.validate import (
    TOLERANCE,
    face_plane,
    plane_deviation,
    surface_pieces,
    validate_scene,
)

FRAME = GeometryFrame("EPSG:2154", "IGN69", (652000.123, 6862000.456, 35.789))
OUTER = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
HOLE = ((3.0, 3.0), (3.0, 7.0), (7.0, 7.0), (7.0, 3.0))


def prism(*, outer=OUTER, holes=(), warp=0.0, name="building", offset=0.0):
    def ring3(ring, roof):
        return tuple(
            (x + offset, y, 6.0 + (warp if (x, y) == outer[2] else 0.0) if roof else 0.0)
            for x, y in ring
        )

    surfaces = [
        Surface(
            f"{name}:ground", ring3(outer[::-1], False),
            tuple(ring3(hole[::-1], False) for hole in holes), "GroundSurface",
        ),
        Surface(
            f"{name}:roof", ring3(outer, True),
            tuple(ring3(hole, True) for hole in holes), "RoofSurface",
        ),
    ]
    for ri, ring in enumerate((outer, *holes)):
        for i, (a, b) in enumerate(zip(ring, (*ring[1:], ring[0]))):
            surfaces.append(
                Surface(
                    f"{name}:wall:{ri}:{i}",
                    (*ring3((a, b), False), *ring3((b, a), True)),
                    semantic="WallSurface",
                )
            )
    return Solid(
        name, name,
        tuple(
            replace(
                surface,
                source_object_id=name,
                source_face_id=f"raw:{surface.face_id}",
                provenance=("CityJSON LoD 2.2", "native:capture"),
            )
            for surface in surfaces
        ),
    )


def points(scene):
    return {
        point
        for solid in scene.solids
        for surface in solid.surfaces
        for ring in (surface.exterior, *surface.interiors)
        for point in ring
    }


def test_bounded_warp_preserves_vertices_semantics_provenance_and_raw_input():
    raw = GeometryScene(FRAME, (prism(warp=0.04),))
    before = copy.deepcopy(raw)
    output, ledger = adapt_scene(raw)
    assert raw == before
    assert raw.solids[0].volume is None
    assert points(output) == points(raw)
    assert output.solids[0].volume > 600.0
    assert output.solids[0].validation["status"] == "passed"
    assert all(plane_deviation(surface) <= TOLERANCE for surface in output.solids[0].surfaces)
    roofs = [surface for surface in output.solids[0].surfaces if surface.semantic == "RoofSurface"]
    assert len(roofs) == 2
    assert all(len(surface.exterior) == 3 for surface in roofs)
    assert all(surface.source_face_id == surface.face_id for surface in roofs)
    assert all("source_face:raw:building:roof" in surface.provenance for surface in roofs)
    assert all(surface.provenance[:2] == ("CityJSON LoD 2.2", "native:capture") for surface in roofs)
    assert [s for s in output.solids[0].surfaces if s.semantic != "RoofSurface"] == [
        s for s in raw.solids[0].surfaces if s.semantic != "RoofSurface"
    ]
    assert ledger["triangulated_face_count"] == 1
    assert ledger["input_triangulated_face_count"] == 1
    assert ledger["triangles_created"] == 2
    assert ledger["changed"] is True
    assert ledger["moved_vertex_count"] == ledger["max_vertex_movement_m"] == 0
    roof_record = next(item for item in ledger["face_mappings"] if item["semantic"] == "RoofSurface")
    assert roof_record["source_face_id"] == "raw:building:roof"
    assert roof_record["source_provenance"] == ["CityJSON LoD 2.2", "native:capture"]
    assert roof_record["output_face_ids"] == [surface.face_id for surface in roofs]
    assert roof_record["source_rings"] == [[list(point) for point in raw.solids[0].surfaces[1].exterior]]
    assert 0.001 < ledger["max_original_deviation_m"] < 0.05
    json.dumps(ledger, allow_nan=False)


def test_explicit_deviation_bound_and_unchanged_validator_defaults():
    raw = GeometryScene(FRAME, (prism(warp=0.04),))
    with pytest.raises(ReconstructionError) as error:
        adapt_scene(raw, max_deviation=0.01)
    assert error.value.code == "planarity_tolerance_exceeded"
    output, _ = adapt_scene(raw, max_deviation=0.03)
    assert output.solids[0].volume > 0
    for call in (face_plane, surface_pieces):
        with pytest.raises(ReconstructionError, match="0.001 m"):
            call(raw.solids[0].surfaces[1])
        assert call(raw.solids[0].surfaces[1], planarity_tolerance=0.03)
    with pytest.raises(ReconstructionError, match="0.001 m"):
        validate_scene(raw)


@pytest.mark.parametrize(
    "limit", [-1, -0.001, math.nan, math.inf, -math.inf, True, None, "0.05", 10**400]
)
def test_nonfinite_negative_or_nonnumeric_limits_rejected(limit):
    with pytest.raises(ReconstructionError) as error:
        adapt_scene(GeometryScene(FRAME, (prism(),)), max_deviation=limit)
    assert error.value.code == "adaptation_invalid"


def test_planar_identity_hash_and_repeat_consistency():
    raw = GeometryScene(FRAME, (prism(),))
    output, ledger = adapt_scene(raw, max_deviation=0)
    assert output == validate_scene(raw)
    assert ledger["before_geometry_hash"] == ledger["after_geometry_hash"] == scene_geometry_hash(raw)
    assert ledger["changed"] is False
    assert ledger["input_face_count"] == ledger["output_face_count"] == 6
    assert ledger["triangulated_face_count"] == 0
    assert ledger["moved_vertex_count"] == 0
    assert ledger["source_vertex_count"] == ledger["output_vertex_count"] == 8
    assert all(item["source"] == item["output"] for item in ledger["vertex_correspondence"])
    repeated, repeated_ledger = adapt_scene(raw, max_deviation=0)
    assert repeated == output
    assert repeated_ledger == ledger
    assert validate_scene(output) == output


def test_adapted_warp_is_idempotent_with_geometry_hash_excluding_audit_metadata():
    raw = GeometryScene(FRAME, (prism(warp=0.04),))
    output, ledger = adapt_scene(raw)
    assert ledger["before_geometry_hash"] != ledger["after_geometry_hash"]
    assert adapt_scene(raw) == (output, ledger)
    repeated, repeated_ledger = adapt_scene(output)
    assert repeated == output == validate_scene(output)
    assert repeated_ledger["before_geometry_hash"] == repeated_ledger["after_geometry_hash"]
    assert repeated_ledger["after_geometry_hash"] == ledger["after_geometry_hash"]
    assert repeated_ledger["triangulated_face_count"] == 0
    assert repeated_ledger["changed"] is False


@pytest.mark.parametrize(
    "outer,holes",
    [
        (OUTER, (HOLE,)),
        (((0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (4.0, 4.0), (4.0, 10.0), (0.0, 10.0)), ()),
    ],
)
def test_warped_concave_and_holed_faces_retain_exact_footprint(outer, holes):
    raw = GeometryScene(FRAME, (prism(outer=outer, holes=holes, warp=0.04),))
    output, ledger = adapt_scene(raw)
    roofs = [s for s in output.solids[0].surfaces if s.semantic == "RoofSurface"]
    expected = Polygon(outer, holes)
    triangles = [Polygon([point[:2] for point in surface.exterior]) for surface in roofs]
    assert expected.symmetric_difference(unary_union(triangles)).area < 1e-10
    assert sum(part.area for part in triangles) == pytest.approx(expected.area)
    assert points(output) == points(raw)
    assert ledger["triangulated_face_count"] == 1
    assert validate_scene(output) == output


def test_planar_holes_are_not_unnecessarily_decomposed():
    raw = GeometryScene(FRAME, (prism(holes=(HOLE,)),))
    output, ledger = adapt_scene(raw)
    assert sum(len(surface.interiors) for surface in output.solids[0].surfaces) == 2
    assert ledger["triangulated_face_count"] == 0
    assert output.solids[0].volume == pytest.approx(504.0)


@pytest.mark.parametrize("holes", [(), (HOLE,)])
def test_adapted_roofs_pass_the_existing_strict_osm_exporter(tmp_path, holes):
    from geogen.osm import build_model_from_geometry, save_model_from_geometry
    from geogen.reconstruction.thermal import make_thermal_scene

    scene, _ = adapt_scene(GeometryScene(FRAME, (prism(holes=holes, warp=0.04),)))
    thermal = make_thermal_scene(scene)
    model = build_model_from_geometry(thermal)
    assert sum(space.volume() for space in model.getSpaces()) == pytest.approx(
        scene.solids[0].volume
    )
    assert save_model_from_geometry(model, tmp_path / "adapted.osm").is_file()


def test_vertex_equivalence_movement_has_exact_source_output_correspondence():
    solid = prism()
    roof = solid.surfaces[1]
    roof = replace(roof, exterior=((0.0005, 0.0, 6.0), *roof.exterior[1:]))
    solid = replace(solid, surfaces=(solid.surfaces[0], roof, *solid.surfaces[2:]))
    raw = GeometryScene(FRAME, (solid,))
    output, ledger = adapt_scene(raw)
    movement, = [item for item in ledger["vertex_correspondence"] if item["movement_m"]]
    assert movement["source"] == [0.0005, 0.0, 6.0]
    assert movement["output"] == [0.0, 0.0, 6.0]
    assert movement["source_occurrences"] == [
        {"face_id": roof.face_id, "ring_index": 0, "vertex_index": 0}
    ]
    assert len(movement["output_occurrences"]) == 3
    assert ledger["moved_vertex_count"] == 1
    assert ledger["max_vertex_movement_m"] == pytest.approx(0.0005)
    assert ledger["before_geometry_hash"] != ledger["after_geometry_hash"]
    assert ledger["changed"] is True
    assert ledger["validation_normalization"][0]["changed_face_ids"] == [roof.face_id]
    assert output.solids[0].volume == pytest.approx(600.0)
    assert validate_scene(output) == output


def test_normalization_induced_warp_is_reported_and_made_canonically_planar():
    solid = prism()
    wall = solid.surfaces[3]
    assert (10.0, 10.0, 6.0) in wall.exterior
    wall = replace(
        wall,
        exterior=tuple(
            (x, y, z - 0.0004) if (x, y, z) == (10.0, 10.0, 6.0) else (x, y, z)
            for x, y, z in wall.exterior
        ),
    )
    solid = replace(solid, surfaces=(*solid.surfaces[:3], wall, *solid.surfaces[4:]))
    output, ledger = adapt_scene(GeometryScene(FRAME, (solid,)))
    assert ledger["max_original_deviation_m"] <= TOLERANCE
    assert ledger["input_triangulated_face_count"] == 0
    assert ledger["normalization_triangulated_face_count"] == 1
    assert ledger["moved_vertex_count"] == 1
    assert ledger["max_vertex_movement_m"] == pytest.approx(0.0004)
    assert all(plane_deviation(surface) <= TOLERANCE for surface in output.solids[0].surfaces)
    assert validate_scene(output) == output
    repeated, repeated_ledger = adapt_scene(output)
    assert repeated == output
    assert repeated_ledger["moved_vertex_count"] == 0
    assert repeated_ledger["triangulated_face_count"] == 0


def test_edge_noding_is_reported_without_inventing_or_moving_vertices():
    solid = prism()
    roof = solid.surfaces[1]
    roof = replace(roof, exterior=(roof.exterior[0], (5.0, 0.0, 6.0), *roof.exterior[1:]))
    solid = replace(solid, surfaces=(solid.surfaces[0], roof, *solid.surfaces[2:]))
    raw = GeometryScene(FRAME, (solid,))
    output, ledger = adapt_scene(raw)
    assert ledger["validation_normalization"][0]["noded_vertex_occurrences_added"] == 1
    assert ledger["changed"] is True
    assert ledger["moved_vertex_count"] == 0
    assert points(raw) == points(output)
    assert ledger["before_geometry_hash"] != ledger["after_geometry_hash"]


@pytest.mark.parametrize("mutation", ["open", "reversed", "duplicate", "interior-shell", "overlapping"])
def test_adaptation_keeps_topological_gates_hard(mutation):
    solid = prism(warp=0.04)
    solids = (solid,)
    if mutation == "open":
        solids = (replace(solid, surfaces=solid.surfaces[:-1]),)
    elif mutation == "reversed":
        solids = (
            replace(
                solid,
                surfaces=tuple(replace(s, exterior=s.exterior[::-1]) for s in solid.surfaces),
            ),
        )
    elif mutation == "duplicate":
        solids = (replace(solid, surfaces=(*solid.surfaces, solid.surfaces[0])),)
    elif mutation == "interior-shell":
        solids = (replace(solid, interior_shells=(solid.surfaces,)),)
    elif mutation == "overlapping":
        solids = (solid, prism(name="other", offset=2.0))
    with pytest.raises(ReconstructionError):
        adapt_scene(GeometryScene(FRAME, solids))


def test_self_intersecting_closed_triangle_shell_is_not_repaired():
    solid = prism()
    triangles = []
    for face in solid.surfaces:
        vertices = tuple((5.0, 5.0, -2.0) if p == (10.0, 10.0, 6.0) else p for p in face.exterior)
        for i in (1, 2):
            triangles.append(
                replace(
                    face, face_id=f"{face.face_id}:{i}",
                    exterior=(vertices[0], vertices[i], vertices[i + 1]),
                )
            )
    with pytest.raises(ReconstructionError, match="intersection"):
        adapt_scene(GeometryScene(FRAME, (replace(solid, surfaces=tuple(triangles)),)))


def test_generated_triangle_identifiers_do_not_collide_with_original_faces():
    solid = prism(warp=0.04)
    ground = replace(solid.surfaces[0], face_id=f"{solid.surfaces[1].face_id}:adapt:triangle:0")
    solid = replace(solid, surfaces=(ground, *solid.surfaces[1:]))
    output, ledger = adapt_scene(GeometryScene(FRAME, (solid,)))
    identifiers = [surface.face_id for surface in output.solids[0].surfaces]
    assert len(set(identifiers)) == len(identifiers)
    assert ledger["triangles_created"] == 2


def test_cityjson_local_float_transform_and_semantic_roundtrip(tmp_path):
    raw = GeometryScene(FRAME, (prism(holes=(HOLE,), warp=0.0412345),))
    scene, _ = adapt_scene(raw)
    path = write_scene_cityjson(scene, tmp_path / "canonical.city.json")
    assert path == tmp_path / "canonical.city.json"
    doc = json.loads(path.read_text())
    assert doc["version"] == "1.1"
    assert doc["transform"] == {"scale": [1, 1, 1], "translate": list(FRAME.origin)}
    assert doc["metadata"]["referenceSystem"] == FRAME.projected_crs
    assert {tuple(point) for point in doc["vertices"]} == points(scene)
    assert max(abs(v) for point in doc["vertices"] for v in point) < 100
    assert any(point[2] == 6.0412345 for point in doc["vertices"])
    imported = validate_scene(import_cityjson(path, FRAME))
    assert {solid.reconstruction_id for solid in imported.solids} == {"building"}
    assert imported.solids[0].volume == pytest.approx(scene.solids[0].volume, abs=1e-7)
    for actual, expected in zip(sorted(points(imported)), sorted(points(scene))):
        assert actual == pytest.approx(expected, abs=1e-8)
    assert [s.semantic for s in imported.solids[0].surfaces] == [
        s.semantic for s in scene.solids[0].surfaces
    ]


def test_cityjson_preserves_selected_building_parts_and_multisolid_nesting(tmp_path):
    first = replace(prism(name="part"), reconstruction_id="root")
    second = replace(prism(name="other", offset=20), reconstruction_id="root")
    second = replace(
        second, surfaces=tuple(replace(s, source_object_id="part") for s in second.surfaces)
    )
    third = replace(prism(name="other-part", offset=40), reconstruction_id="root")
    scene, _ = adapt_scene(GeometryScene(FRAME, (first, second, third)))
    path = write_scene_cityjson(scene, tmp_path / "parts.city.json")
    doc = json.loads(path.read_text())
    assert doc["CityObjects"]["root"]["children"] == ["part", "other-part"]
    assert "geometry" not in doc["CityObjects"]["root"]
    assert doc["CityObjects"]["part"]["parents"] == ["root"]
    assert doc["CityObjects"]["part"]["geometry"][0]["type"] == "MultiSolid"
    assert doc["CityObjects"]["other-part"]["geometry"][0]["type"] == "Solid"
    imported = validate_scene(import_cityjson(path, FRAME))
    assert len(imported.solids) == 3
    assert {solid.reconstruction_id for solid in imported.solids} == {"root"}
    assert [solid.volume for solid in imported.solids] == pytest.approx([600.0] * 3)


def test_serializer_does_not_silently_normalize_or_export_nonplanar_input(tmp_path):
    with pytest.raises(ReconstructionError):
        write_scene_cityjson(GeometryScene(FRAME, (prism(warp=0.00002),)), tmp_path / "warped.json")
    solid = prism()
    roof = solid.surfaces[1]
    roof = replace(roof, exterior=((0.0005, 0.0, 6.0), *roof.exterior[1:]))
    solid = replace(solid, surfaces=(solid.surfaces[0], roof, *solid.surfaces[2:]))
    with pytest.raises(ReconstructionError, match="canonical"):
        write_scene_cityjson(
            GeometryScene(FRAME, (solid,)),
            tmp_path / "normalized.json",
        )
