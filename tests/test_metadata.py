from geogen.bdnb import BuildingGroup
from geogen.metadata import (
    building_group_description,
    building_group_metadata,
    portfolio_metadata,
)


def test_building_group_metadata():
    group = BuildingGroup(
        batiment_groupe_id="bg-123",
        height=15.2,
        storeys=5,
        ground_elevation=48.7,
        footprint_area=420.0,
        address="122 Rue Amelot",
        city="Paris",
        fictitious_geometry=False,
        glazed_ratio=25.0,
        roof_material="tuile",
        roof_type="combles",
    )

    data = building_group_metadata(group)

    assert data["bdnb_id"] == "bg-123"
    assert data["footprint_area_m2"] == 420.0
    assert data["number_of_storeys"] == 5
    assert data["height_m"] == 15.2
    assert data["estimated_floor_area_m2"] == 2100.0
    assert data["address"] == "122 Rue Amelot"
    assert data["city"] == "Paris"


def test_building_group_description():
    group = BuildingGroup(
        batiment_groupe_id="bg-123",
        height=13.0,
        storeys=5,
        footprint_area=115.0,
        address="122 Rue Amelot 75011 Paris 11e Arrondissement",
        city="Paris",
    )

    description = building_group_description(group)

    assert description == (
        "5-storey building located at 122 Rue Amelot, in Paris, "
        "with an approximate footprint of 115 m², "
        "an estimated floor area of 575 m², and a height of 13.0 m."
    )


def test_portfolio_metadata():
    group_1 = BuildingGroup(
        batiment_groupe_id="bg-1",
        height=12.0,
        storeys=4,
        footprint_area=300.0,
        address="A1",
        city="Paris",
    )
    group_2 = BuildingGroup(
        batiment_groupe_id="bg-2",
        height=18.0,
        storeys=6,
        footprint_area=500.0,
        address="A2",
        city="Paris",
    )

    data = portfolio_metadata(
        {"bg-1": group_1, "bg-2": group_2},
        source_addresses=["A1", "A2"],
    )

    assert data["building_count"] == 2
    assert data["total_footprint_area_m2"] == 800.0
    assert data["total_estimated_floor_area_m2"] == 4200.0
    assert len(data["buildings"]) == 2
