import json
from types import MappingProxyType

import pytest

from geogen.bdnb import BuildingGroup
from geogen.metadata import (
    building_description,
    building_group_description,
    building_group_metadata,
    building_metadata,
    portfolio_metadata,
    save_metadata_json,
)
from geogen.models import Building, OsBuilding


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


def test_mixed_provider_metadata_and_json(tmp_path):
    french = BuildingGroup(
        "os-same-id",
        height=12,
        storeys=4,
        footprint_area=100,
        address="122 Rue Amelot 75011 Paris",
        city="Paris",
        glazed_ratio=25,
    )
    british = OsBuilding(
        "same-id",
        height=9,
        storeys=3,
        footprint_area=200,
        address="10 Downing Street, London SW1A 2AA",
        roof_shape="flat",
    )
    # Equal normalized codes across providers must retain independent provenance.
    data = portfolio_metadata(
        MappingProxyType({"bdnb:os-same-id": french, "os:same-id": british}),
        source_addresses=(b.address for b in (french, british)),
    )
    fr, uk = data["buildings"]
    assert fr["building_id"] == uk["building_id"] == "os-same-id"
    assert fr["provider"] == "bdnb"
    assert fr["country"] == "FR"
    assert fr["bdnb_id"] == "os-same-id"
    assert fr["glazing_ratio"] == 25
    assert "os_id" not in fr
    assert uk["provider"] == "ordnance_survey"
    assert uk["country"] == "UK"
    assert uk["crs"] == "EPSG:27700"
    assert uk["os_id"] == "same-id"
    assert uk["roof_shape"] == "flat"
    assert uk["glazing_ratio"] is None
    assert "bdnb_id" not in uk
    assert british.address in uk["description"]
    assert "75011" not in fr["description"]
    assert data["building_count"] == 2
    assert data["total_footprint_area_m2"] == 300
    assert data["total_estimated_floor_area_m2"] == 1000
    path = save_metadata_json(data, tmp_path / "nested" / "metadata.json")
    assert json.loads(path.read_text(encoding="utf-8")) == data
    assert "m²" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("country", ["FR", "UK", "US"])
def test_generic_building_is_not_mislabeled_by_country_or_prefix(country):
    building = Building("os-123", country=country)
    data = building_metadata(building)
    assert data["building_id"] == "os-123"
    assert data["country"] == country
    assert data["provider"] is None
    assert "bdnb_id" not in data
    assert "os_id" not in data
    assert data["height_m"] is None
    assert data["estimated_floor_area_m2"] is None
    assert building_description(building) == "Building."
    assert building_group_metadata(building) == data
    assert building_group_description(building) == building_description(building)


def test_non_french_description_does_not_strip_five_digit_address_component():
    building = Building("external", country="US", address="Unit 12345 Example Street")
    assert building.address in building_description(building)


def test_unknown_and_zero_measurements_in_portfolios():
    unknown = OsBuilding("unknown", footprint_area=20)
    known = BuildingGroup("known", footprint_area=0, storeys=0)
    data = portfolio_metadata(b for b in (unknown, known))
    assert data["total_footprint_area_m2"] == 20
    assert data["total_estimated_floor_area_m2"] == 0
    assert data["buildings"][0]["estimated_floor_area_m2"] is None
    assert data["buildings"][1]["estimated_floor_area_m2"] == 0
    for records in ([], [Building("unknown")]):
        empty = portfolio_metadata(records)
        assert empty["building_count"] == len(records)
        assert empty["total_footprint_area_m2"] is None
        assert empty["total_estimated_floor_area_m2"] is None


def test_standalone_bdnb_identifier_is_preserved_in_metadata():
    building = BuildingGroup.from_row({"rnb_id": "standalone"})
    data = building_metadata(building)
    assert data["provider"] == "bdnb"
    assert data["bdnb_id"] == data["building_id"] == "standalone"
