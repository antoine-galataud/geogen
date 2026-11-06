from dataclasses import replace

import laspy
import numpy as np
import pytest
from pyproj import CRS
from shapely import affinity
from shapely.geometry import Polygon, box

from geogen.reconstruction.evidence import _plane, _residuals, assess_evidence
from geogen.reconstruction.types import (
    GeometryFrame,
    GeometryScene,
    ReconstructionUnit,
    Solid,
    SourceIdentity,
    Surface,
)


FRAME = GeometryFrame("EPSG:2154", "NGF-IGN69", (652000.0, 6862000.0, 35.0))


def surface(polygon, z, semantic, face_id):
    def ring(coords):
        return tuple((x, y, z(x, y) if callable(z) else z) for x, y in coords)
    return Surface(
        face_id, ring(polygon.exterior.coords),
        tuple(ring(r.coords) for r in polygon.interiors), semantic=semantic,
    )


def unit(polygon):
    global_polygon = affinity.translate(polygon, xoff=FRAME.origin[0], yoff=FRAME.origin[1])
    return ReconstructionUnit(
        "building", SourceIdentity(), global_polygon.wkt, global_polygon.wkt,
        "original", outline_type="wall_footprint",
    )


def scene(polygon, roofs=None, ground_z=0.0, frame=FRAME):
    roofs = roofs if roofs is not None else [surface(polygon, 5.0, "RoofSurface", "roof")]
    return GeometryScene(frame, (
        Solid("solid", "building", tuple(roofs) + (
            surface(polygon, ground_z, "GroundSurface", "ground"),
        )),
    ))


def roof_points(polygon, z=5.0):
    from shapely.geometry import Point
    xmin, ymin, xmax, ymax = polygon.bounds
    return [
        (x, y, z(x, y) if callable(z) else z)
        for x in np.arange(xmin + 0.25, xmax, 0.5)
        for y in np.arange(ymin + 0.25, ymax, 0.5)
        if polygon.contains(Point(x, y))
    ]


def lidar(path, roofs, grounds=None, crs="EPSG:5698"):
    if grounds is None:
        grounds = [(x, -0.5, 0.0) for x in np.arange(0.25, 10, 0.5)]
    points = np.asarray([*roofs, *grounds], dtype=float).reshape((-1, 3))
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = np.asarray(FRAME.origin)
    header.scales = np.array([0.001, 0.001, 0.001])
    if crs is not None:
        header.add_crs(CRS.from_user_input(crs))
    data = laspy.LasData(header)
    if len(points):
        data.x, data.y, data.z = (points + np.asarray(FRAME.origin)).T
    data.classification = np.array([6] * len(roofs) + [2] * len(grounds), dtype=np.uint8)
    data.write(path)
    return path


def assess(tmp_path, polygon, model=None, points=None, grounds=None):
    path = lidar(tmp_path / "evidence.laz", roof_points(polygon) if points is None else points, grounds)
    return assess_evidence(model or scene(polygon), (unit(polygon),), path)["building"]


@pytest.mark.parametrize("vertical_reference", ["NGF-IGN69", "IGN69", "EPSG:5720"])
def test_box_with_reliable_input_passes_not_surveyed_fidelity(tmp_path, vertical_reference):
    polygon = box(0, 0, 10, 6)
    model = scene(polygon, frame=replace(FRAME, vertical_reference=vertical_reference))
    result = assess(tmp_path, polygon, model)
    assert result["fidelity_status"] == "evidence_passed"
    assert result["benchmark_reference"] == "not_evaluated"
    assert result["reasons"] == []
    metrics = result["metrics"]
    assert metrics["roof_horizontal_coverage"] == 1.0
    assert metrics["roof_point_count"] == 240
    assert metrics["working_class_2_count"] == 20
    assert metrics["original_lidar_point_count"] is None
    assert metrics["outline_extent_uncertainty"] == "wall_footprint_may_differ_from_roofprint"
    assert metrics["roof_residual_p95_m"] == pytest.approx(0.0)


def test_gable_bounded_slopes_pass(tmp_path):
    polygon = box(0, 0, 10, 6)
    height = lambda x, y: 5 + min(y, 6 - y) * 0.5
    roofs = [
        surface(box(0, 0, 10, 3), height, "RoofSurface", "south"),
        surface(box(0, 3, 10, 6), height, "RoofSurface", "north"),
    ]
    result = assess(tmp_path, polygon, scene(polygon, roofs), roof_points(polygon, height))
    assert result["fidelity_status"] == "evidence_passed"
    assert result["metrics"]["minimum_roof_surface_coverage"] == 1.0


def test_sparse_and_absent_roof_evidence_are_not_zero_error_passes(tmp_path):
    polygon = box(0, 0, 10, 6)
    sparse = assess(tmp_path, polygon, points=[(1, 1, 5)])
    assert sparse["fidelity_status"] == "needs_review"
    assert "insufficient_roof_points" in sparse["reasons"]
    absent = assess(tmp_path, polygon, points=[])
    assert absent["metrics"]["roof_point_count"] == 0
    assert absent["metrics"]["roof_residual_median_m"] is None
    assert absent["metrics"]["roof_residual_p95_m"] is None


def test_missing_wing_fails_even_with_infinite_plane_agreement(tmp_path):
    polygon = box(0, 0, 10, 6).union(box(10, 0, 14, 3))
    model = scene(polygon, [surface(box(0, 0, 10, 6), 5, "RoofSurface", "main")])
    result = assess(tmp_path, polygon, model)
    assert result["fidelity_status"] == "needs_review"
    assert "roof_outline_mismatch" in result["reasons"]
    assert result["metrics"]["roof_residual_p95_m"] > 0.5


@pytest.mark.parametrize("ground_z,roof_z,reason", [
    (1.0, 5.0, "ground_residual_threshold_exceeded"),
    (0.0, 6.0, "roof_residual_threshold_exceeded"),
])
def test_wrong_ground_or_roof_offset_fails(tmp_path, ground_z, roof_z, reason):
    polygon = box(0, 0, 10, 6)
    model = scene(polygon, [surface(polygon, roof_z, "RoofSurface", "roof")], ground_z)
    result = assess(tmp_path, polygon, model)
    assert result["fidelity_status"] == "needs_review"
    assert reason in result["reasons"]


def test_courtyard_preserved_and_filling_it_rejected(tmp_path):
    polygon = Polygon(box(0, 0, 10, 6).exterior.coords, [box(3, 2, 7, 4).exterior.coords])
    result = assess(tmp_path, polygon)
    assert result["fidelity_status"] == "evidence_passed"
    assert result["metrics"]["roof_gap_area_m2"] == 0.0
    filled = assess(tmp_path, polygon, scene(box(0, 0, 10, 6)))
    assert filled["fidelity_status"] == "needs_review"
    assert "roof_fills_courtyard" in filled["reasons"]
    assert "ground_fills_courtyard" in filled["reasons"]


def test_bounded_3d_residual_uses_hole_and_edge_distances():
    polygon = Polygon(box(0, 0, 10, 6).exterior.coords, [box(3, 2, 7, 4).exterior.coords])
    plane = _plane(surface(polygon, 5, "RoofSurface", "roof"))
    residuals = _residuals(np.array([[11, 3, 6], [5, 3, 5], [1, 1, 5]]), [plane])
    assert residuals == pytest.approx([2 ** 0.5, 1, 0])


def test_unknown_vertical_datum_and_semantics_block_acceptance(tmp_path):
    polygon = box(0, 0, 10, 6)
    path = lidar(tmp_path / "evidence.laz", roof_points(polygon))
    model = scene(polygon, frame=replace(FRAME, vertical_reference="unknown"))
    result = assess_evidence(model, (replace(unit(polygon), outline_type="unknown"),), path)["building"]
    assert result["fidelity_status"] == "needs_review"
    assert "unverified_vertical_reference" in result["reasons"]
    assert "undefined_outline_semantics" in result["reasons"]
    unknown_roof = surface(polygon, 5, "Unknown", "roof")
    result = assess(tmp_path, polygon, scene(polygon, [unknown_roof]))
    assert "undefined_surface_semantics" in result["reasons"]
    assert "missing_roof_surfaces" in result["reasons"]
    assert result["metrics"]["roof_residual_p95_m"] is None


def test_missing_lidar_returns_nullable_metrics(tmp_path):
    polygon = box(0, 0, 10, 6)
    result = assess_evidence(scene(polygon), (unit(polygon),), tmp_path / "absent.laz")["building"]
    assert result["fidelity_status"] == "needs_review"
    assert result["metrics"]["working_class_6_count"] is None
    assert result["metrics"]["roof_residual_median_m"] is None
    assert "lidar_read_failed" in result["reasons"]


def test_no_automatic_outlier_removal(tmp_path):
    polygon = box(0, 0, 10, 6)
    points = roof_points(polygon)
    points = [(x, y, z + 2 if i % 10 == 0 else z) for i, (x, y, z) in enumerate(points)]
    result = assess(tmp_path, polygon, points=points)
    assert result["metrics"]["roof_residual_median_m"] == 0.0
    assert result["metrics"]["roof_residual_p95_m"] == 2.0
    assert result["fidelity_status"] == "needs_review"


def test_deterministic_sampling_retains_exact_counts_and_coverage(tmp_path, monkeypatch):
    import geogen.reconstruction.evidence as evidence
    monkeypatch.setattr(evidence, "MAX_RESIDUAL_POINTS", 17)
    monkeypatch.setattr(evidence, "CHUNK_SIZE", 37)
    polygon = box(0, 0, 10, 6)
    first = assess(tmp_path, polygon)
    second = assess(tmp_path, polygon)
    assert first == second
    assert first["metrics"]["roof_point_count"] == 240
    assert first["metrics"]["roof_residual_sample_count"] == 16
    assert first["metrics"]["roof_horizontal_coverage"] == 1.0


def test_individual_roof_wing_without_returns_fails(tmp_path):
    main, wing = box(0, 0, 10, 6), box(10, 0, 11, 3)
    polygon = main.union(wing)
    roofs = [
        surface(main, 5, "RoofSurface", "main"),
        surface(wing, 5, "RoofSurface", "wing"),
    ]
    result = assess(tmp_path, polygon, scene(polygon, roofs), roof_points(main))
    assert result["metrics"]["roof_horizontal_coverage"] > 0.8
    assert result["metrics"]["minimum_roof_surface_coverage"] == 0.0
    assert "insufficient_roof_surface_coverage" in result["reasons"]
    assert result["fidelity_status"] == "needs_review"


def test_absent_ground_is_not_a_zero_residual(tmp_path):
    polygon = box(0, 0, 10, 6)
    result = assess(tmp_path, polygon, grounds=[])
    assert result["fidelity_status"] == "needs_review"
    assert result["metrics"]["ground_point_count"] == 0
    assert result["metrics"]["ground_residual_median_m"] is None
    assert "insufficient_ground_points" in result["reasons"]


def test_ground_outline_shift_is_not_hidden_by_roof_agreement(tmp_path):
    polygon = box(0, 0, 10, 6)
    model = scene(
        box(1, 0, 11, 6), [surface(polygon, 5, "RoofSurface", "roof")]
    )
    result = assess(tmp_path, polygon, model)
    assert result["metrics"]["roof_iou"] == 1.0
    assert "ground_outline_mismatch" in result["reasons"]
    assert result["fidelity_status"] == "needs_review"


def test_small_missing_component_is_not_hidden_by_overall_iou(tmp_path):
    main, annex = box(0, 0, 10, 6), box(11, 0, 11.5, 0.5)
    polygon = main.union(annex)
    result = assess(tmp_path, polygon, scene(main))
    assert result["metrics"]["roof_iou"] > 0.98
    assert min(result["metrics"]["roof_component_coverage"]) == 0.0
    assert "roof_outline_mismatch" in result["reasons"]


@pytest.mark.parametrize("vertical_reference,projected_crs", [
    ("synthetic", "EPSG:2154"),
    ("NGF-IGN69", "EPSG:4326"),
])
def test_unverified_input_frame_cannot_pass(tmp_path, vertical_reference, projected_crs):
    polygon = box(0, 0, 10, 6)
    frame = replace(FRAME, vertical_reference=vertical_reference, projected_crs=projected_crs)
    result = assess(tmp_path, polygon, scene(polygon, frame=frame))
    assert result["fidelity_status"] == "needs_review"


def test_vertical_roof_is_not_treated_as_a_roof_plane(tmp_path):
    polygon = box(0, 0, 10, 6)
    vertical = Surface(
        "wall_roof", ((0, 0, 0), (10, 0, 0), (10, 0, 5), (0, 0, 5)),
        semantic="RoofSurface",
    )
    result = assess(tmp_path, polygon, scene(polygon, [vertical]))
    assert result["fidelity_status"] == "needs_review"
    assert "invalid_roof_or_ground_surface" in result["reasons"]
    assert result["metrics"]["roof_residual_p95_m"] is None


def test_nonplanar_roof_cannot_pass(tmp_path):
    polygon = box(0, 0, 10, 6)
    warped = Surface(
        "warped", ((0, 0, 5), (10, 0, 5), (10, 6, 7), (0, 6, 5)),
        semantic="RoofSurface",
    )
    result = assess(tmp_path, polygon, scene(polygon, [warped]))
    assert "invalid_roof_or_ground_surface" in result["reasons"]
    assert result["fidelity_status"] == "needs_review"


def test_coverage_limit_does_not_invent_coverage(tmp_path, monkeypatch):
    import geogen.reconstruction.evidence as evidence
    monkeypatch.setattr(evidence, "MAX_CELLS", 10)
    polygon = box(0, 0, 10, 6)
    result = assess(tmp_path, polygon)
    assert result["metrics"]["roof_point_count"] == 240
    assert result["metrics"]["roof_horizontal_coverage"] is None
    assert "coverage_computation_limit" in result["reasons"]
    assert result["fidelity_status"] == "needs_review"


def test_roofprint_mismatch_is_flagged_without_geometry_mutation(tmp_path):
    polygon = box(0, 0, 10, 6)
    model = scene(
        box(1, 1, 9, 5), [surface(polygon, 5, "RoofSurface", "roof")]
    )
    original = repr(model)
    path = lidar(tmp_path / "evidence.laz", roof_points(polygon))
    roof_unit = replace(unit(polygon), outline_type="roofprint")
    result = assess_evidence(model, (roof_unit,), path)["building"]
    assert "ground_outline_mismatch" in result["reasons"]
    assert result["fidelity_status"] == "needs_review"
    assert repr(model) == original


@pytest.mark.parametrize("crs,reason", [
    (None, "missing_lidar_crs"),
    ("EPSG:3857", "contradictory_lidar_horizontal_crs"),
    ("EPSG:2154+5721", "contradictory_lidar_vertical_crs"),
])
def test_lidar_header_srs_must_not_be_missing_or_contradict_frame(tmp_path, crs, reason):
    polygon = box(0, 0, 10, 6)
    path = lidar(tmp_path / "header.laz", roof_points(polygon), crs=crs)
    result = assess_evidence(scene(polygon), (unit(polygon),), path)["building"]
    assert reason in result["reasons"]
    assert result["fidelity_status"] == "needs_review"


def test_horizontal_las_header_requires_verified_frame_vertical_reference(tmp_path):
    polygon = box(0, 0, 10, 6)
    path = lidar(tmp_path / "header.laz", roof_points(polygon), crs="EPSG:2154")
    result = assess_evidence(scene(polygon), (unit(polygon),), path)["building"]
    assert result["fidelity_status"] == "evidence_passed"
    model = scene(polygon, frame=replace(FRAME, vertical_reference="unknown"))
    result = assess_evidence(model, (unit(polygon),), path)["building"]
    assert result["fidelity_status"] == "needs_review"


@pytest.mark.parametrize("outline_type", ["groundprint", "unknown"])
def test_unsupported_outline_semantics_cannot_pass(tmp_path, outline_type):
    polygon = box(0, 0, 10, 6)
    path = lidar(tmp_path / "outline.laz", roof_points(polygon))
    result = assess_evidence(
        scene(polygon), (replace(unit(polygon), outline_type=outline_type),), path
    )["building"]
    assert result["fidelity_status"] == "needs_review"
    assert "undefined_outline_semantics" in result["reasons"]


def test_roofprint_can_pass_without_wall_extent_uncertainty(tmp_path):
    polygon = box(0, 0, 10, 6)
    path = lidar(tmp_path / "outline.laz", roof_points(polygon))
    result = assess_evidence(
        scene(polygon), (replace(unit(polygon), outline_type="roofprint"),), path
    )["building"]
    assert result["fidelity_status"] == "evidence_passed"
    assert result["metrics"]["outline_extent_uncertainty"] is None


def test_unknown_outline_with_exact_explicit_extent_assumption_can_pass(tmp_path):
    polygon = box(0, 0, 10, 6)
    path = lidar(tmp_path / "assumed.laz", roof_points(polygon))
    assumed = replace(
        unit(polygon), outline_type="unknown",
        decisions=("wall_extent_assumption=roofprint_extruded",),
    )
    result = assess_evidence(scene(polygon), (assumed,), path)["building"]
    assert result["fidelity_status"] == "evidence_passed"
    assert result["metrics"]["outline_extent_uncertainty"] == "unknown_outline_with_assumed_extruded_roofprint"
    assert result["metrics"]["wall_extent_assumption"] == "roofprint_extruded"


@pytest.mark.parametrize("bad_model,reason", [
    ("offset", "roof_residual_threshold_exceeded"),
    ("missing_wing", "roof_outline_mismatch"),
    ("unknown_surface", "undefined_surface_semantics"),
    ("sparse", "insufficient_roof_coverage"),
])
def test_extent_assumption_does_not_bypass_geometry_or_evidence_gates(tmp_path, bad_model, reason):
    polygon = box(0, 0, 10, 6)
    roof_polygon = box(0, 0, 8, 6) if bad_model == "missing_wing" else polygon
    roof = surface(
        roof_polygon, 6 if bad_model == "offset" else 5,
        "Unknown" if bad_model == "unknown_surface" else "RoofSurface", "roof",
    )
    points = roof_points(polygon)
    path = lidar(tmp_path / "assumed.laz", points[:1] if bad_model == "sparse" else points)
    assumed = replace(
        unit(polygon), outline_type="unknown",
        decisions=("wall_extent_assumption=roofprint_extruded",),
    )
    result = assess_evidence(scene(polygon, [roof]), (assumed,), path)["building"]
    assert result["fidelity_status"] == "needs_review"
    assert reason in result["reasons"]


@pytest.mark.parametrize("semantic", ["OuterFloorSurface", "OuterCeilingSurface"])
def test_defined_overhang_semantics_are_not_roof_substitutes(tmp_path, semantic):
    polygon = box(0, 0, 10, 6)
    overhang = surface(polygon, 5, semantic, "overhang")
    result = assess(tmp_path, polygon, scene(polygon, [overhang]))
    assert "undefined_surface_semantics" not in result["reasons"]
    assert "unsupported_overhang_surface_evidence" in result["reasons"]
    assert "missing_roof_surfaces" in result["reasons"]
    assert result["fidelity_status"] == "needs_review"
    result = assess(tmp_path, polygon, scene(
        polygon, [surface(polygon, 5, "RoofSurface", "roof"), overhang]
    ))
    assert result["metrics"]["roof_residual_p95_m"] == 0.0
    assert "unsupported_overhang_surface_evidence" in result["reasons"]
    assert result["fidelity_status"] == "needs_review"
