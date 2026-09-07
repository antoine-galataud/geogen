"""Tests of the geometry helpers."""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from geogen.bdnb import BuildingGroup
from geogen.geometry import (
    GeometryError,
    bounding_center,
    exterior_ring,
    footprints_from_group,
    is_geographic,
    parse_geometry,
    polygons,
    roof_apex,
    storeys_and_height,
    to_projected,
    to_wgs84,
)

WKT_RECTANGLE = (
    "POLYGON ((652000 6862000, 652020 6862000, 652020 6862010, 652000 6862010, 652000 6862000))"
)


def test_parse_geojson_geometry(geom_groupe: dict) -> None:
    geometry = parse_geometry(geom_groupe)

    assert geometry.geom_type == "MultiPolygon"
    assert geometry.area == pytest.approx(200.0)


def test_parse_wkt_geometry() -> None:
    assert parse_geometry(WKT_RECTANGLE).area == pytest.approx(200.0)


def test_parse_wkb_hexadecimal_geometry() -> None:
    hexadecimal = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]).wkb_hex

    assert parse_geometry(hexadecimal).area == pytest.approx(100.0)


def test_parse_invalid_geometry() -> None:
    with pytest.raises(GeometryError):
        parse_geometry(42)


def test_geographic_geometries_are_reprojected() -> None:
    geometry = parse_geometry(
        "POLYGON ((2.33 48.86, 2.331 48.86, 2.331 48.861, 2.33 48.861, 2.33 48.86))"
    )
    assert is_geographic(geometry)

    projected = to_projected(geometry)

    assert not is_geographic(projected)
    assert projected.bounds[0] == pytest.approx(651000, abs=5000)
    assert projected.bounds[1] == pytest.approx(6862000, abs=5000)


def test_projected_geometries_are_left_untouched(geom_groupe: dict) -> None:
    geometry = parse_geometry(geom_groupe)

    assert to_projected(geometry) is geometry


def test_to_wgs84() -> None:
    longitude, latitude = to_wgs84(652000.0, 6862000.0)

    assert longitude == pytest.approx(2.3458, abs=1e-3)
    assert latitude == pytest.approx(48.8562, abs=1e-3)


def test_polygons_drops_negligible_parts(geom_groupe: dict) -> None:
    geometry = parse_geometry(
        {
            "type": "MultiPolygon",
            "coordinates": geom_groupe["coordinates"]
            + [[[[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1], [0.0, 0.0]]]],
        }
    )

    assert [round(polygon.area) for polygon in polygons(geometry)] == [200]


def test_exterior_ring_is_clockwise_and_open() -> None:
    ring = exterior_ring(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]))

    assert ring == ((0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0))


def test_exterior_ring_simplifies_collinear_points() -> None:
    polygon = Polygon([(0, 0), (5, 0), (10, 0), (10, 10), (0, 10)])

    assert len(exterior_ring(polygon)) == 4


def test_exterior_ring_of_a_degenerate_polygon() -> None:
    with pytest.raises(GeometryError):
        exterior_ring(Polygon([(0, 0), (10, 0), (10, 1e-6)]))


@pytest.mark.parametrize(
    ("height", "storeys", "expected"),
    [
        (9.0, 3, (3, 9.0)),
        (None, 4, (4, 12.0)),
        (10.0, None, (3, 10.0)),
        (None, None, (1, 3.0)),
        (0.0, 0, (1, 3.0)),
    ],
)
def test_storeys_and_height(
    height: float | None, storeys: int | None, expected: tuple[int, float]
) -> None:
    assert storeys_and_height(height, storeys) == expected


def test_footprints_from_group(geom_groupe: dict) -> None:
    group = BuildingGroup(
        batiment_groupe_id="bdnb-0001",
        geometry=geom_groupe,
        height=9.0,
        storeys=3,
        ground_elevation=35.0,
    )

    footprints = footprints_from_group(group)

    assert len(footprints) == 1
    footprint = footprints[0]
    assert footprint.name == "bdnb-0001"
    assert footprint.storeys == 3
    assert footprint.height == 9.0
    assert footprint.storey_height == pytest.approx(3.0)
    assert footprint.ground_elevation == 35.0
    assert len(footprint.ring) == 4


def test_footprints_from_a_multipart_group(geom_groupe: dict) -> None:
    shifted = [
        [[[x + 100, y] for x, y in ring] for ring in polygon]
        for polygon in geom_groupe["coordinates"]
    ]
    group = BuildingGroup(
        batiment_groupe_id="bdnb-0001",
        geometry={"type": "MultiPolygon", "coordinates": geom_groupe["coordinates"] + shifted},
    )

    footprints = footprints_from_group(group)

    assert [footprint.name for footprint in footprints] == [
        "bdnb-0001 Part 1",
        "bdnb-0001 Part 2",
    ]


def test_footprints_from_a_group_without_geometry() -> None:
    assert footprints_from_group(BuildingGroup(batiment_groupe_id="bdnb-0001")) == []


def test_bounding_center(geom_groupe: dict) -> None:
    group = BuildingGroup(batiment_groupe_id="bdnb-0001", geometry=geom_groupe)

    assert bounding_center(footprints_from_group(group)) == (652010.0, 6862005.0)


def test_self_intersecting_polygons_are_repaired() -> None:
    bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
    assert not bowtie.is_valid

    parts = polygons(bowtie)

    assert len(parts) == 1
    assert parts[0].is_valid
    assert parts[0].area == pytest.approx(25.0)


def test_three_dimensional_geometries_are_flattened() -> None:
    geometry = parse_geometry(
        "POLYGON Z ((652000 6862000 33, 652020 6862000 33, 652020 6862010 33, 652000 6862000 33))"
    )

    assert not geometry.has_z
    assert len(exterior_ring(geometry)) == 3


def test_footprints_without_ground_elevation(geom_groupe: dict) -> None:
    group = BuildingGroup(batiment_groupe_id="bdnb-0001", geometry=geom_groupe)

    assert footprints_from_group(group)[0].ground_elevation is None


def test_footprints_carry_the_envelope(geom_groupe: dict) -> None:
    group = BuildingGroup(
        batiment_groupe_id="bdnb-0001",
        geometry=geom_groupe,
        glazed_ratio=22.0,
        glazing_orientations=("sud",),
        roof_material="Ardoises",
    )

    envelope = footprints_from_group(group, default_roof_pitch=35.0)[0].envelope

    assert envelope.window_to_wall_ratio == pytest.approx(0.22)
    assert envelope.glazing_orientations == ("south",)
    assert envelope.roof_pitch == pytest.approx(35.0)


def test_footprints_of_an_undocumented_building_have_an_empty_envelope(
    geom_groupe: dict,
) -> None:
    group = BuildingGroup(batiment_groupe_id="bdnb-0001", geometry=geom_groupe)

    envelope = footprints_from_group(group)[0].envelope

    assert envelope.window_to_wall_ratio is None
    assert envelope.roof_pitch is None


def test_roof_apex_is_the_farthest_point_from_the_boundary() -> None:
    ring = ((0.0, 0.0), (0.0, 10.0), (20.0, 10.0), (20.0, 0.0))

    apex = roof_apex(ring, 45.0, max_height=10.0)

    assert apex is not None
    x, y, height = apex
    assert (x, y) == pytest.approx((10.0, 5.0), abs=0.5)
    assert height == pytest.approx(5.0, abs=0.5)


def test_roof_apex_is_capped() -> None:
    ring = ((0.0, 0.0), (0.0, 10.0), (20.0, 10.0), (20.0, 0.0))

    assert roof_apex(ring, 45.0, max_height=2.0)[2] == pytest.approx(2.0)


def test_roof_apex_of_a_negligible_roof() -> None:
    ring = ((0.0, 0.0), (0.0, 1.0), (2.0, 1.0), (2.0, 0.0))

    assert roof_apex(ring, 30.0, max_height=6.0) is None
    assert roof_apex(ring, 0.0, max_height=6.0) is None


def test_footprints_carry_the_estimated_window_to_wall_ratio(geom_groupe: dict) -> None:
    group = BuildingGroup(batiment_groupe_id="bdnb-0001", geometry=geom_groupe)

    envelope = footprints_from_group(group, default_window_to_wall_ratio=15.0)[0].envelope

    assert envelope.window_to_wall_ratio == pytest.approx(0.15)
    assert envelope.roof_pitch is None
