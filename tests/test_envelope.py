"""Tests of the envelope approximated from the BDNB attributes."""

from __future__ import annotations

import pytest

from geogen.bdnb import BuildingGroup
from geogen.envelope import (
    DEFAULT_ROOF_PITCH,
    cardinal_point,
    envelope_from_group,
    glazing_orientations,
    roof_pitch_from_material,
    roof_pitch_from_walls,
    window_to_wall_ratio,
)


def group(**kwargs: object) -> BuildingGroup:
    return BuildingGroup(batiment_groupe_id="bdnb-0001", **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("azimuth", "expected"),
    [
        (0.0, "north"),
        (44.0, "north"),
        (90.0, "east"),
        (180.0, "south"),
        (270.0, "west"),
        (350.0, "north"),
        (-90.0, "west"),
    ],
)
def test_cardinal_point(azimuth: float, expected: str) -> None:
    assert cardinal_point(azimuth) == expected


def test_window_to_wall_ratio_reads_a_percentage() -> None:
    assert window_to_wall_ratio(group(glazed_ratio=16.5)) == pytest.approx(0.165)


def test_window_to_wall_ratio_reads_a_fraction() -> None:
    assert window_to_wall_ratio(group(glazed_ratio=0.2)) == pytest.approx(0.2)


def test_window_to_wall_ratio_falls_back_on_the_dpe_areas() -> None:
    ratio = window_to_wall_ratio(
        group(glazed_areas=(("north", 10.0), ("south", 20.0)), wall_area=200.0)
    )

    assert ratio == pytest.approx(0.15)


def test_window_to_wall_ratio_is_capped() -> None:
    assert window_to_wall_ratio(group(glazed_ratio=98.0)) == pytest.approx(0.9)


@pytest.mark.parametrize(
    "attributes",
    [
        {},
        {"glazed_ratio": 0.0},
        {"glazed_ratio": 0.005},
        {"glazed_areas": (("north", 10.0),)},
        {"wall_area": 200.0},
        {"glazed_areas": (("north", 10.0),), "wall_area": 0.0},
    ],
)
def test_window_to_wall_ratio_without_usable_data(attributes: dict) -> None:
    assert window_to_wall_ratio(group(**attributes)) is None


def test_glazing_orientations_are_translated() -> None:
    orientations = glazing_orientations(
        group(glazing_orientations=("Sud", "nord-est", "ouest", "horizontal"))
    )

    assert orientations == ("south", "north", "east", "west")


def test_glazing_orientations_without_data() -> None:
    assert glazing_orientations(group(glazing_orientations=("horizontal",))) == ()
    assert glazing_orientations(group()) == ()


def test_roof_pitch_from_walls_weighs_the_faces_by_area() -> None:
    walls = [
        {"wall_type": "roof", "inclination": 150.0, "area": 60.0},
        {"wall_type": "roof", "inclination": 170.0, "area": 20.0},
        {"wall_type": "vertical", "inclination": 90.0, "area": 100.0},
        {"wall_type": "floor", "inclination": 0.0, "area": 60.0},
    ]

    assert roof_pitch_from_walls(walls) == pytest.approx(25.0)


def test_roof_pitch_from_walls_of_a_flat_roof() -> None:
    assert roof_pitch_from_walls([{"wall_type": "roof", "inclination": 180.0}]) == 0.0


def test_roof_pitch_from_walls_without_roof_face() -> None:
    assert roof_pitch_from_walls([{"wall_type": "vertical", "inclination": 90.0}]) is None
    assert roof_pitch_from_walls([]) is None


@pytest.mark.parametrize("material", ["Tuiles", "ardoise", "Zinc Aluminium", "tuiles béton"])
def test_roof_pitch_from_a_sloped_material(material: str) -> None:
    assert roof_pitch_from_material(material, 35.0) == 35.0


@pytest.mark.parametrize("material", ["Béton", "toiture terrasse", "Bitume"])
def test_roof_pitch_from_a_flat_material(material: str) -> None:
    assert roof_pitch_from_material(material, 35.0) == 0.0


@pytest.mark.parametrize("material", [None, "", "indetermine"])
def test_roof_pitch_from_an_unknown_material(material: str | None) -> None:
    assert roof_pitch_from_material(material, 35.0) is None


def test_envelope_prefers_the_measured_roof_faces() -> None:
    envelope = envelope_from_group(
        group(
            roof_material="Tuiles",
            walls=({"wall_type": "roof", "inclination": 180.0},),
        )
    )

    assert envelope.roof_pitch is None


def test_envelope_falls_back_on_the_roof_material() -> None:
    envelope = envelope_from_group(group(roof_material="Tuiles"))

    assert envelope.roof_pitch == DEFAULT_ROOF_PITCH


def test_envelope_of_an_undocumented_building() -> None:
    assert envelope_from_group(group()) == envelope_from_group(group())
    envelope = envelope_from_group(group())

    assert envelope.roof_pitch is None
    assert envelope.window_to_wall_ratio is None
    assert envelope.glazing_orientations == ()


def test_envelope_of_a_documented_building() -> None:
    envelope = envelope_from_group(
        group(
            glazed_ratio=18.0,
            glazing_orientations=("nord", "sud"),
            walls=({"wall_type": "roof", "inclination": 140.0, "area": 50.0},),
        )
    )

    assert envelope.window_to_wall_ratio == pytest.approx(0.18)
    assert envelope.glazing_orientations == ("north", "south")
    assert envelope.roof_pitch == pytest.approx(40.0)


@pytest.mark.parametrize("upper_floor", ["Combles perdus", "Combles aménagés sous rampants"])
def test_roof_pitch_from_the_upper_floor_of_an_attic(upper_floor: str) -> None:
    envelope = envelope_from_group(group(roof_type=upper_floor))

    assert envelope.roof_pitch == DEFAULT_ROOF_PITCH


def test_roof_pitch_from_a_flat_upper_floor() -> None:
    assert envelope_from_group(group(roof_type="Toiture terrasse")).roof_pitch is None


def test_the_roof_material_wins_over_the_upper_floor() -> None:
    envelope = envelope_from_group(group(roof_material="Tuiles", roof_type="Dalle béton"))

    assert envelope.roof_pitch == DEFAULT_ROOF_PITCH


def test_glazing_orientations_fall_back_on_the_glazed_areas() -> None:
    orientations = glazing_orientations(group(glazed_areas=(("north", 10.0), ("south", 20.0))))

    assert orientations == ("north", "south")


def test_glazing_orientations_prefer_the_published_ones() -> None:
    orientations = glazing_orientations(
        group(glazing_orientations=("est",), glazed_areas=(("north", 10.0),))
    )

    assert orientations == ("east",)


def test_window_to_wall_ratio_falls_back_on_the_given_estimate() -> None:
    assert window_to_wall_ratio(group(), 20.0) == pytest.approx(0.2)
    assert window_to_wall_ratio(group(), 0.25) == pytest.approx(0.25)


def test_the_estimated_window_to_wall_ratio_is_capped() -> None:
    assert window_to_wall_ratio(group(), 98.0) == pytest.approx(0.9)


def test_the_bdnb_wins_over_the_estimated_window_to_wall_ratio() -> None:
    assert window_to_wall_ratio(group(glazed_ratio=16.5), 40.0) == pytest.approx(0.165)
    ratio = window_to_wall_ratio(
        group(glazed_areas=(("north", 10.0), ("south", 20.0)), wall_area=200.0), 40.0
    )

    assert ratio == pytest.approx(0.15)


@pytest.mark.parametrize("estimate", [None, 0.0])
def test_window_to_wall_ratio_without_data_nor_estimate(estimate: float | None) -> None:
    assert window_to_wall_ratio(group(), estimate) is None


def test_the_estimated_window_to_wall_ratio_completes_the_envelope() -> None:
    envelope = envelope_from_group(group(), default_window_to_wall_ratio=25.0)

    assert envelope.window_to_wall_ratio == pytest.approx(0.25)
    assert envelope.glazing_orientations == ()
    assert envelope.roof_pitch is None


def test_the_estimated_window_to_wall_ratio_keeps_the_known_orientations() -> None:
    envelope = envelope_from_group(
        group(glazing_orientations=("sud",)), default_window_to_wall_ratio=25.0
    )

    assert envelope.window_to_wall_ratio == pytest.approx(0.25)
    assert envelope.glazing_orientations == ("south",)
