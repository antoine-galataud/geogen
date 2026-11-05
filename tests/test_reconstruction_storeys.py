"""Storey partitions must conserve the validated exterior, including its roof."""

from dataclasses import replace

import pytest
from shapely.errors import GEOSException
from shapely.geometry import Polygon
from shapely.ops import unary_union

from geogen.reconstruction import thermal as thermal_module
from geogen.reconstruction.thermal import (
    make_thermal_scene,
    thermal_geometry_hash,
    validate_thermal_scene,
)
from geogen.reconstruction.types import GeometryScene, ReconstructionError
from geogen.reconstruction.validate import (
    TOLERANCE,
    dot,
    face_plane,
    project,
    sub,
    validate_scene,
)
from tests.test_reconstruction_geometry import FRAME, prism, profile_solid


def assert_conserved(solid, thermal):
    source = validate_scene(GeometryScene(FRAME, (solid,))).solids[0]
    children = [space for space in thermal.spaces if space.solid_id == solid.solid_id]
    assert sum(space.volume for space in children) == pytest.approx(source.volume)
    for surface in source.surfaces:
        plane = face_plane(surface)
        pieces = [
            s
            for space in children
            for s in space.surfaces
            if s.source_face_id == surface.source_face_id
        ]
        assert pieces
        assert all(
            abs(dot(sub(point, plane[0]), plane[3])) <= TOLERANCE
            for piece in pieces
            for ring in (piece.exterior, *piece.interiors)
            for point in ring
        )
        polygons = [
            Polygon(
                [project(p, plane) for p in s.exterior],
                [[project(p, plane) for p in r] for r in s.interiors],
            )
            for s in pieces
        ]
        assert unary_union(polygons).symmetric_difference(plane[4]).area <= TOLERANCE
        assert sum(p.area for p in polygons) == pytest.approx(plane[4].area)
        assert all(piece.semantic == surface.semantic for piece in pieces)
    assert validate_thermal_scene(thermal) is thermal


def test_storey_cuts_share_containment_index_and_exclude_courtyard(monkeypatch):
    solid = prism(holes=(((3.0, 3.0), (3.0, 7.0), (7.0, 7.0), (7.0, 3.0)),))
    native_index = thermal_module._SolidIndex
    queries, indexes = [], []

    class ObservedIndex:
        def __init__(self, surfaces):
            self.index = native_index(surfaces)
            indexes.append(self.index)

        def contains(self, point):
            queries.append(point)
            return self.index.contains(point)

    monkeypatch.setattr(thermal_module, "_SolidIndex", ObservedIndex)
    parts = thermal_module._partition(solid, [2.0, 4.0])
    assert len(indexes) == 1
    assert len(queries) == 4  # Building cross-section and courtyard at both elevations.
    caps = [surface for part in parts for surface in part.surfaces if surface.boundary == "surface"]
    assert len(caps) == 4
    assert all(face_plane(cap)[4].area == pytest.approx(84) for cap in caps)
    assert all(len(cap.interiors) == 1 for cap in caps)


@pytest.mark.parametrize(
    "solid,cuts,volumes",
    [
        (prism(), [2.0, 4.0], [200.0, 200.0, 200.0]),
        (
            profile_solid(((0.0, 6.0), (5.0, 9.0), (10.0, 6.0))),
            [2.0, 4.0],
            [200.0, 200.0, 350.0],
        ),
        (
            prism(holes=(((3.0, 3.0), (3.0, 7.0), (7.0, 7.0), (7.0, 3.0)),)),
            [2.0, 4.0],
            [168.0, 168.0, 168.0],
        ),
    ],
    ids=["flat", "gable", "courtyard"],
)
def test_storeys_conserve_volume_and_every_source_face(solid, cuts, volumes):
    report = {}
    thermal = make_thermal_scene(
        GeometryScene(FRAME, (solid,)),
        levels={solid.solid_id: cuts},
        partition_report=report,
    )
    assert [space.volume for space in thermal.spaces] == pytest.approx(volumes)
    assert [space.storey_number for space in thermal.spaces] == [1, 2, 3]
    assert [space.nominal_z for space in thermal.spaces] == [0.0, *cuts]
    assert all(space.conditioned is None for space in thermal.spaces)
    assert report == {
        solid.solid_id: {
            "status": "partitioned",
            "requested_levels_m": cuts,
            "space_ids": [space.space_id for space in thermal.spaces],
        }
    }
    assert_conserved(solid, thermal)
    interfaces = {}
    for space in thermal.spaces:
        for surface in space.surfaces:
            if surface.boundary == "surface":
                interfaces.setdefault(surface.interface_id, []).append((space, surface))
    assert len(interfaces) == len(cuts)
    for pairs in interfaces.values():
        assert len(pairs) == 2
        (a, fa), (b, fb) = pairs
        assert fa.adjacent_space_id == b.space_id
        assert fb.adjacent_space_id == a.space_id
        assert {fa.semantic, fb.semantic} == {"GroundSurface", "RoofSurface"}
        assert dot(face_plane(fa)[3], face_plane(fb)[3]) == pytest.approx(-1)
        assert len(fa.interiors) == len(solid.surfaces[0].interiors)
    source_roofs = {s.face_id for s in solid.surfaces if s.semantic == "RoofSurface"}
    assert not any(
        surface.source_face_id in source_roofs
        for space in thermal.spaces[:-1]
        for surface in space.surfaces
    )


def test_cut_through_gable_preserves_sloping_roof_geometry():
    solid = profile_solid(((0.0, 4.0), (5.0, 7.0), (10.0, 4.0)))
    thermal = make_thermal_scene(GeometryScene(FRAME, (solid,)), levels={"profile": [3.0, 5.0]})
    assert len(thermal.spaces) == 3
    assert_conserved(solid, thermal)
    assert any(
        s.semantic == "RoofSurface" and s.boundary == "outdoors" for s in thermal.spaces[1].surfaces
    )
    assert max(p[2] for space in thermal.spaces for s in space.surfaces for p in s.exterior) == 7.0


def test_sloping_courtyard_roof_keeps_its_hole_and_source_planes():
    solid = prism(holes=(((3.0, 3.0), (3.0, 7.0), (7.0, 7.0), (7.0, 3.0)),))

    def slope(ring):
        return tuple((x, y, 6.0 + 0.2 * x + 0.1 * y if z == 6.0 else z) for x, y, z in ring)

    solid = replace(
        solid,
        surfaces=tuple(
            replace(
                surface,
                exterior=slope(surface.exterior),
                interiors=tuple(slope(ring) for ring in surface.interiors),
            )
            for surface in solid.surfaces
        ),
    )
    thermal = make_thermal_scene(
        GeometryScene(FRAME, (solid,)), levels={"building": [2.0, 4.0, 7.5]}
    )
    assert len(thermal.spaces) == 4
    assert_conserved(solid, thermal)
    assert sum(space.volume for space in thermal.spaces) == pytest.approx(630.0)


def test_legacy_and_explicit_single_storey_share_geometry_hash():
    solid = prism(offset=(0.0, 0.0, -2.0))
    scene = GeometryScene(FRAME, (solid,))
    report = {}
    legacy = make_thermal_scene(scene, partition_report=report)
    assert legacy.spaces[0].storey_number is None
    assert legacy.spaces[0].nominal_z is None
    assert report["building"]["requested_levels_m"] is None
    assert report["building"]["status"] == "single_volume"
    single = make_thermal_scene(scene, levels={"building": []}, partition_report=report)
    assert single.spaces[0].storey_number == 1
    assert single.spaces[0].nominal_z == -2.0
    assert single.geometry_hash == legacy.geometry_hash
    assert report["building"] == {
        "status": "single_volume",
        "requested_levels_m": [],
        "space_ids": ["building"],
    }
    partitioned = make_thermal_scene(scene, levels={"building": [0.0, 2.0]})
    assert [space.nominal_z for space in partitioned.spaces] == [-2.0, 0.0, 2.0]
    assert_conserved(solid, partitioned)


def test_adjacent_buildings_with_different_levels_keep_reciprocal_party_walls():
    a = prism("a")
    b = prism("b", offset=(10.0, 0.0, 0.0))
    thermal = make_thermal_scene(GeometryScene(FRAME, (a, b)), levels={"a": [3.0], "b": [2.0, 4.0]})
    assert len(thermal.spaces) == 5
    assert_conserved(a, thermal)
    assert_conserved(b, thermal)
    walls = [
        surface
        for space in thermal.spaces
        for surface in space.surfaces
        if surface.boundary == "surface" and surface.semantic == "WallSurface"
    ]
    assert len(walls) == 8
    assert sum(face_plane(s)[4].area for s in walls) == pytest.approx(120.0)
    by_id = {space.space_id: space for space in thermal.spaces}
    for space in thermal.spaces:
        for surface in space.surfaces:
            if surface in walls:
                assert by_id[surface.adjacent_space_id].solid_id != space.solid_id


def test_levels_are_scoped_to_exact_solid_not_shared_reconstruction_id():
    a = replace(prism("a"), reconstruction_id="building")
    b = replace(prism("b", offset=(20.0, 0.0, 0.0)), reconstruction_id="building")
    report = {}
    thermal = make_thermal_scene(
        GeometryScene(FRAME, (a, b)), levels={"a": [3.0]}, partition_report=report
    )
    assert [space.solid_id for space in thermal.spaces] == ["a", "a", "b"]
    assert [space.storey_number for space in thermal.spaces] == [1, 2, None]
    assert report["b"] == {
        "status": "single_volume",
        "requested_levels_m": None,
        "space_ids": ["b"],
    }
    with pytest.raises(ReconstructionError, match="exact source solid IDs"):
        make_thermal_scene(
            GeometryScene(FRAME, (a, b)),
            levels={"building": [3.0]},
            partition_report={},
        )


@pytest.fixture(scope="module")
def unpartitioned_prism():
    return make_thermal_scene(GeometryScene(FRAME, (prism(),)))


@pytest.fixture(scope="module")
def partitioned_prism():
    return make_thermal_scene(GeometryScene(FRAME, (prism(),)), levels={"building": [3.0]})


@pytest.mark.parametrize(
    "cuts",
    [
        [0.0],
        [6.0],
        [-1.0],
        [7.0],
        [4.0, 3.0],
        [3.0, 3.0],
        [float("nan")],
        [float("inf")],
        [True],
        ["3"],
        3,
        None,
    ],
)
def test_invalid_cuts_are_strict_unless_fallback_is_requested(cuts, unpartitioned_prism):
    scene = GeometryScene(FRAME, (prism(),))
    with pytest.raises(ReconstructionError):
        make_thermal_scene(scene, levels={"building": cuts})
    report = {}
    thermal = make_thermal_scene(scene, levels={"building": cuts}, partition_report=report)
    assert len(thermal.spaces) == 1
    assert thermal.spaces[0].space_id == "building"
    assert thermal.spaces[0].storey_number is None
    assert thermal.geometry_hash == unpartitioned_prism.geometry_hash
    assert report["building"]["status"] == "fallback_single_volume"
    assert report["building"]["failure"]["code"] == "unsupported_zoning"
    assert report["building"]["failure"]["message"]


def test_disconnected_upper_storey_falls_back_atomically_per_solid():
    valley = profile_solid(((0.0, 7.0), (5.0, 4.0), (10.0, 7.0)))
    neighbour = prism("neighbour", offset=(10.0, 0.0, 0.0))
    scene = GeometryScene(FRAME, (valley, neighbour))
    levels = {"profile": [3.0, 5.0], "neighbour": [3.0]}
    with pytest.raises(ReconstructionError, match="connected"):
        make_thermal_scene(scene, levels=levels)
    report = {}
    thermal = make_thermal_scene(scene, levels=levels, partition_report=report)
    assert len(thermal.spaces) == 3
    assert report["profile"]["status"] == "fallback_single_volume"
    assert report["profile"]["space_ids"] == ["profile"]
    assert "connected" in report["profile"]["failure"]["message"]
    assert report["neighbour"]["status"] == "partitioned"
    assert_conserved(valley, thermal)
    assert_conserved(neighbour, thermal)


def test_horizontal_source_face_at_cut_is_not_replaced():
    solid = profile_solid(((0.0, 4.0), (5.0, 4.0), (5.0, 7.0), (10.0, 7.0)))
    report = {}
    thermal = make_thermal_scene(
        GeometryScene(FRAME, (solid,)), levels={"profile": [4.0]}, partition_report=report
    )
    assert report["profile"]["status"] == "fallback_single_volume"
    assert "coincident" in report["profile"]["failure"]["message"]
    assert_conserved(solid, thermal)


@pytest.mark.parametrize("failure", [GEOSException("topology error"), RuntimeError("bug")])
def test_fallback_only_catches_geometry_errors(monkeypatch, failure):
    def broken_partition(solid, cuts):
        raise failure

    monkeypatch.setattr(thermal_module, "_partition", broken_partition)
    scene = GeometryScene(FRAME, (prism(),))
    if isinstance(failure, RuntimeError):
        with pytest.raises(RuntimeError, match="bug"):
            make_thermal_scene(scene, levels={"building": [3.0]}, partition_report={})
    else:
        report = {}
        thermal = make_thermal_scene(scene, levels={"building": [3.0]}, partition_report=report)
        assert len(thermal.spaces) == 1
        assert report["building"]["failure"]["message"] == "topology error"
        with pytest.raises(GEOSException):
            make_thermal_scene(scene, levels={"building": [3.0]})


def test_partition_validation_failure_does_not_keep_partial_children(monkeypatch):
    real_partition = thermal_module._partition

    def broken_partition(solid, cuts):
        children = real_partition(solid, cuts)
        children[-1] = replace(children[-1], surfaces=children[-1].surfaces[:-1])
        return children

    monkeypatch.setattr(thermal_module, "_partition", broken_partition)
    report = {}
    solid = prism()
    thermal = make_thermal_scene(
        GeometryScene(FRAME, (solid,)), levels={"building": [2.0, 4.0]}, partition_report=report
    )
    assert [space.space_id for space in thermal.spaces] == ["building"]
    assert report["building"]["failure"]["code"] == "geometry_invalid"
    assert_conserved(solid, thermal)


@pytest.mark.parametrize("invalid", ["open", "overlap", "semantic"])
def test_report_never_weakens_source_validation(invalid):
    solid = prism()
    solids = [solid]
    if invalid == "open":
        solids = [replace(solid, surfaces=solid.surfaces[:-1])]
    elif invalid == "overlap":
        solids.append(prism("overlap", offset=(1.0, 0.0, 0.0)))
    else:
        solids = [
            replace(
                solid,
                surfaces=(replace(solid.surfaces[0], semantic="Unknown"), *solid.surfaces[1:]),
            )
        ]
    report = {}
    with pytest.raises(ReconstructionError):
        make_thermal_scene(
            GeometryScene(FRAME, tuple(solids)), levels={"building": [3.0]}, partition_report=report
        )
    assert report == {}


def test_combined_identifier_collision_is_strict_without_opt_in_fallback():
    a = prism("a")
    collision = prism("a:storey:1", offset=(20.0, 0.0, 0.0))
    scene = GeometryScene(FRAME, (a, collision))
    with pytest.raises(ReconstructionError, match="duplicate solid identifiers"):
        make_thermal_scene(scene, levels={"a": [3.0]})
    report = {}
    thermal = make_thermal_scene(scene, levels={"a": [3.0]}, partition_report=report)
    assert [space.space_id for space in thermal.spaces] == ["a", "a:storey:1"]
    assert report["a"]["status"] == "fallback_single_volume"
    assert report["a"]["requested_levels_m"] == [3.0]
    assert "Combined" in report["a"]["failure"]["message"]
    assert report["a:storey:1"]["status"] == "single_volume"
    assert_conserved(a, thermal)
    assert_conserved(collision, thermal)


def test_submillimetre_party_wall_corner_mismatch_is_unified_across_solids():
    # Independently reconstructed neighbours whose party-wall corners differ by 40 µm.
    a = prism("a")
    b = prism("b", height=6.00004, offset=(10.0, 0.0, 0.0))
    thermal = make_thermal_scene(GeometryScene(FRAME, (a, b)), levels={"a": [3.0], "b": [3.0]})
    shared = [s for space in thermal.spaces for s in space.surfaces if s.boundary == "surface"]
    walls = [s for s in shared if s.semantic == "WallSurface"]
    assert sum(face_plane(s)[4].area for s in walls) == pytest.approx(120.0)
    assert not any(
        "remainder" in s.face_id and s.semantic == "WallSurface"
        for space in thermal.spaces
        for s in space.surfaces
    )
    assert sum(space.volume for space in thermal.spaces) == pytest.approx(1200.0, abs=1e-2)
    assert any("adjacent solids within 0.0127 m are unified" in t for t in thermal.assumptions)
    assert validate_thermal_scene(thermal) is thermal


def test_zero_width_overlay_spikes_are_removed():
    # The ring doubles back from (5, 0) through (0.004, 0) to the origin.
    spiked = Polygon([(0.0, 0.0), (-1.0, 1.0), (0.004, 0.0), (5.0, -1e-10)])
    cleaned = thermal_module._without_spikes(spiked)
    assert list(cleaned.exterior.coords)[:-1] == [(0.0, 0.0), (-1.0, 1.0), (0.004, 0.0)]
    assert cleaned.area == pytest.approx(spiked.area)
    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert thermal_module._without_spikes(square).equals(square)


def test_submillimetre_party_wall_storey_strip_falls_back_to_original_envelopes():
    a = prism("a", height=6.0)
    # Roofs differ beyond the party-wall vertex unification bound; only the cuts nearly coincide.
    b = prism("b", height=6.05, offset=(10.0, 0.0, 0.0))
    scene = GeometryScene(FRAME, (a, b))
    levels = {"a": [3.0], "b": [3.00075]}
    for solid in (a, b):
        individual = make_thermal_scene(
            GeometryScene(FRAME, (solid,)), levels={solid.solid_id: levels[solid.solid_id]}
        )
        assert len(individual.spaces) == 2
    with pytest.raises(ReconstructionError) as failure:
        make_thermal_scene(scene, levels=levels)
    assert failure.value.code == "vertex_equivalence_collapsed_edge"
    report = {}
    thermal = make_thermal_scene(scene, levels=levels, partition_report=report)
    assert [space.space_id for space in thermal.spaces] == ["a", "b"]
    assert thermal.geometry_hash == make_thermal_scene(scene).geometry_hash
    for solid in (a, b):
        entry = report[solid.solid_id]
        assert entry["status"] == "fallback_single_volume"
        assert entry["requested_levels_m"] == levels[solid.solid_id]
        assert entry["space_ids"] == [solid.solid_id]
        assert entry["failure"]["code"] == "vertex_equivalence_collapsed_edge"
        assert "Combined" in entry["failure"]["message"]
        assert_conserved(solid, thermal)


@pytest.mark.parametrize(
    "failure",
    [
        ReconstructionError("geometry_invalid", "test combined failure"),
        GEOSException("topology error"),
    ],
)
def test_combined_matching_failure_restores_all_split_solids(monkeypatch, failure):
    a, b = prism("a"), prism("b", offset=(10.0, 0.0, 0.0))
    single = prism("single", offset=(30.0, 0.0, 0.0))
    real_match = thermal_module._match_boundaries
    combined_attempts = []

    def fail_partitioned_combination(spaces):
        if len({space.solid_id for space in spaces}) > 1:
            combined_attempts.append([space.space_id for space in spaces])
            if any(space.storey_number == 2 for space in spaces):
                raise failure
        return real_match(spaces)

    monkeypatch.setattr(thermal_module, "_match_boundaries", fail_partitioned_combination)
    scene = GeometryScene(FRAME, (a, b, single))
    levels = {"a": [2.0, 4.0], "b": [3.0], "single": []}
    with pytest.raises(type(failure)):
        make_thermal_scene(scene, levels=levels)
    combined_attempts.clear()
    report = {}
    thermal = make_thermal_scene(scene, levels=levels, partition_report=report)
    assert len(combined_attempts) == 2
    assert combined_attempts[-1] == ["a", "b", "single"]
    for solid in (a, b):
        assert report[solid.solid_id]["status"] == "fallback_single_volume"
        assert report[solid.solid_id]["space_ids"] == [solid.solid_id]
        assert str(failure) in report[solid.solid_id]["failure"]["message"]
        assert_conserved(solid, thermal)
    assert report["single"]["status"] == "single_volume"
    assert [space.storey_number for space in thermal.spaces] == [None, None, 1]
    assert (
        len([s for space in thermal.spaces for s in space.surfaces if s.boundary == "surface"]) == 2
    )


def test_combined_fallback_still_rejects_invalid_original_boundary():
    a, b = prism("a"), prism("b", offset=(20.0, 0.0, 0.0))
    b = replace(b, surfaces=(replace(b.surfaces[0], boundary="surface"), *b.surfaces[1:]))
    report = {}
    with pytest.raises(ReconstructionError, match="Unmatched thermal interface"):
        make_thermal_scene(
            GeometryScene(FRAME, (a, b)), levels={"a": [3.0]}, partition_report=report
        )
    assert report == {}


def test_combined_fallback_does_not_hide_unexpected_errors(monkeypatch):
    real_match = thermal_module._match_boundaries

    def broken_match(spaces):
        if len({space.solid_id for space in spaces}) > 1:
            raise RuntimeError("combined implementation bug")
        return real_match(spaces)

    monkeypatch.setattr(thermal_module, "_match_boundaries", broken_match)
    with pytest.raises(RuntimeError, match="implementation bug"):
        make_thermal_scene(
            GeometryScene(FRAME, (prism("a"), prism("b", offset=(10.0, 0.0, 0.0)))),
            levels={"a": [3.0]},
            partition_report={},
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"storey_number": None},
        {"storey_number": 0},
        {"storey_number": True},
        {"storey_number": 1.0},
        {"storey_number": 2},
        {"nominal_z": None},
        {"nominal_z": True},
        {"nominal_z": float("nan")},
        {"nominal_z": 1.0},
    ],
)
def test_metadata_is_validated_without_changing_geometry_hash(metadata, partitioned_prism):
    thermal = partitioned_prism
    changed = replace(thermal, spaces=(replace(thermal.spaces[0], **metadata), thermal.spaces[1]))
    assert thermal_geometry_hash(changed.frame, changed.spaces) == thermal.geometry_hash
    with pytest.raises(ReconstructionError, match="[Ss]torey"):
        validate_thermal_scene(changed)
