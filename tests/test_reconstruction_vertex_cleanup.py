"""Bounded cleanup for Roofer/val3dity CONSECUTIVE_POINTS_SAME (102)."""

import copy
import json
from dataclasses import replace

import pytest

from geogen.reconstruction.adapt import adapt_scene, write_scene_cityjson
from geogen.reconstruction.cityjson import import_cityjson
from geogen.reconstruction.types import GeometryScene, ReconstructionError
from geogen.reconstruction.validate import node_surfaces, validate_scene
from tests.test_reconstruction_adaptation import FRAME, HOLE, prism


def redundant_roof(solid, *, distance=0, hole=False, seam=False):
    roof = solid.surfaces[1]
    ring = roof.interiors[0] if hole else roof.exterior
    a, b = ring[:2]
    point = tuple(x + distance * (y - x) / 10 for x, y in zip(a, b))
    redundant = (*ring, point) if seam else (a, point, *ring[1:])
    roof = (
        replace(roof, interiors=(redundant, *roof.interiors[1:]))
        if hole
        else replace(roof, exterior=redundant)
    )
    return replace(solid, surfaces=(solid.surfaces[0], roof, *solid.surfaces[2:]))


@pytest.mark.parametrize("distance", [0, 0.0005, 0.001])
@pytest.mark.parametrize("hole,seam", [(False, False), (False, True), (True, False), (True, True)])
def test_cleanup_retains_complete_envelope_and_audit(distance, hole, seam, tmp_path):
    solid = redundant_roof(
        prism(holes=(HOLE,) if hole else ()), distance=distance, hole=hole, seam=seam
    )
    raw = GeometryScene(FRAME, (solid,))
    original = copy.deepcopy(raw)
    cleaned, ledger = adapt_scene(raw)
    assert raw == original
    assert ledger["changed"]
    assert ledger["removed_consecutive_vertex_count"] == 1
    removed = ledger["removed_consecutive_vertices"][0]
    assert removed["face_id"] == solid.surfaces[1].face_id
    assert removed["ring_index"] == int(hole)
    assert ledger["max_vertex_movement_m"] <= 0.001
    assert cleaned.solids[0].volume == pytest.approx(504 if hole else 600)
    assert len(cleaned.solids[0].surfaces) == len(solid.surfaces)
    assert sum(len(s.interiors) for s in cleaned.solids[0].surfaces) == (2 if hole else 0)
    source = next(item for item in ledger["face_mappings"] if item["semantic"] == "RoofSurface")
    assert source["source_rings"] == [
        [list(point) for point in ring]
        for ring in (solid.surfaces[1].exterior, *solid.surfaces[1].interiors)
    ]
    repeated, audit = adapt_scene(cleaned)
    assert repeated == cleaned == validate_scene(cleaned)
    assert not audit["changed"]
    assert audit["removed_consecutive_vertex_count"] == 0
    path = write_scene_cityjson(cleaned, tmp_path / "cleaned.json")
    restored = validate_scene(import_cityjson(path, FRAME))
    assert restored.solids[0].volume == pytest.approx(cleaned.solids[0].volume)
    json.dumps(ledger, allow_nan=False)


def test_cleanup_is_shared_across_faces_and_retains_vertex_correspondence():
    solid = redundant_roof(prism(), distance=0.0005)
    wall = solid.surfaces[2]
    # The adjoining wall traverses the same split edge in the opposite direction.
    wall = replace(wall, exterior=(*wall.exterior[:-1], (0.0005, 0, 6), wall.exterior[-1]))
    solid = replace(solid, surfaces=(*solid.surfaces[:2], wall, *solid.surfaces[3:]))
    cleaned, audit = adapt_scene(GeometryScene(FRAME, (solid,)))
    assert audit["removed_consecutive_vertex_count"] == 2
    (moved,) = [item for item in audit["vertex_correspondence"] if item["movement_m"] > 0]
    assert moved["source"] == [0.0005, 0, 6]
    assert moved["output"] == [0, 0, 6]
    assert len(moved["source_occurrences"]) == 2
    assert len(moved["output_occurrences"]) == 3
    assert cleaned.solids[0].volume == pytest.approx(600)


def test_snap_bound_is_not_increased_by_face_warping_tolerance():
    raw = GeometryScene(FRAME, (redundant_roof(prism(), distance=0.0011),))
    cleaned, audit = adapt_scene(raw, max_deviation=1)
    assert audit["removed_consecutive_vertex_count"] == 0
    assert audit["moved_vertex_count"] == 0
    assert (0.0011, 0, 6) in cleaned.solids[0].surfaces[1].exterior


@pytest.mark.parametrize("mutation", ["triangle", "hole", "pinch", "open"])
def test_cleanup_does_not_discard_faces_holes_or_invalid_topology(mutation):
    solid = redundant_roof(prism())
    roof = solid.surfaces[1]
    if mutation == "triangle":
        roof = replace(roof, exterior=((0, 0, 6), (0.0005, 0, 6), (10, 10, 6)))
    elif mutation == "hole":
        roof = replace(roof, interiors=(((3, 3, 6), (3, 3.0005, 6), (3.0005, 3, 6)),))
    elif mutation == "pinch":
        ring = roof.exterior
        roof = replace(roof, exterior=(*ring[:4], ring[0], *ring[4:]))
    solid = replace(solid, surfaces=(solid.surfaces[0], roof, *solid.surfaces[2:]))
    if mutation == "open":
        solid = replace(solid, surfaces=solid.surfaces[:-1])
    with pytest.raises(ReconstructionError):
        adapt_scene(GeometryScene(FRAME, (solid,)))


def test_cleanup_and_triangulation_both_reported_without_relaxing_warp_limit():
    raw = GeometryScene(FRAME, (redundant_roof(prism(warp=0.04)),))
    with pytest.raises(ReconstructionError) as error:
        adapt_scene(raw, max_deviation=0.001)
    assert error.value.code == "planarity_tolerance_exceeded"
    output, audit = adapt_scene(raw)
    assert audit["removed_consecutive_vertex_count"] == 1
    assert audit["triangulated_face_count"] == 1
    assert output.solids[0].volume > 600


def test_default_vertex_noding_remains_strict():
    with pytest.raises(ReconstructionError) as error:
        node_surfaces(redundant_roof(prism(), distance=0.0005).surfaces)
    assert error.value.code == "vertex_equivalence_collapsed_edge"
