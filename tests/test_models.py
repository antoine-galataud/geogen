from dataclasses import FrozenInstanceError, replace

import pytest

from geogen import Building as PublicBuilding
from geogen import OsBuilding as PublicOsBuilding
from geogen.bdnb import Address as LegacyAddress
from geogen.bdnb import BuildingGroup as LegacyBuildingGroup
from geogen.geometry import footprints_from_group
from geogen.models import Address, Building, BuildingGroup, OsBuilding
from geogen.ordnance_survey import building_from_feature
from geogen.providers import BdnbProvider


def test_models_are_centralized_and_legacy_imports_are_preserved():
    assert LegacyAddress is Address
    assert LegacyBuildingGroup is BuildingGroup
    assert PublicBuilding is Building
    assert PublicOsBuilding is OsBuilding
    assert issubclass(BuildingGroup, Building)
    assert issubclass(OsBuilding, Building)


@pytest.mark.parametrize(
    "group_id,construction_id,expected",
    [("group", "construction", "group"), ("", "construction", "construction"), ("", None, "")],
)
def test_bdnb_identity_is_recomputed_on_replace(group_id, construction_id, expected):
    building = BuildingGroup("original", height=12, walls=({"area": 20},))
    updated = replace(
        building, batiment_groupe_id=group_id, batiment_construction_id=construction_id
    )
    assert updated.code == expected
    assert updated.height == 12
    assert updated.walls == building.walls
    assert updated.provider == "bdnb"
    assert building.code == "original"
    with pytest.raises(FrozenInstanceError):
        updated.height = 9


def test_bdnb_row_factory_preserves_subclasses_and_source_fields():
    class CustomBuildingGroup(BuildingGroup):
        pass

    building = CustomBuildingGroup.from_row(
        {
            "batiment_groupe_id": None,
            "rnb_id": "standalone",
            "hauteur_mean": "12.5",
            "nb_niveau": "4",
            "surface_vitree_nord": "8",
            "surface_mur_exterieur": "40",
            "l_orientation_baie_vitree": "{nord,sud}",
            "contient_fictive_geom_groupe": "false",
            "mat_toit_txt": "tuile",
        }
    )
    assert isinstance(building, CustomBuildingGroup)
    assert building.code == "standalone"
    assert building.batiment_construction_id == "standalone"
    assert building.height == 12.5
    assert building.storeys == 4
    assert building.glazed_areas == (("north", 8.0),)
    assert building.wall_area == 40.0
    assert building.glazing_orientations == ("nord", "sud")
    assert building.fictitious_geometry is False
    assert building.roof_material == "tuile"


@pytest.mark.parametrize(
    "ring,expected_crs",
    [
        (((2.0, 48.0), (2.0, 48.0001), (2.0001, 48.0001), (2.0, 48.0)), "EPSG:4326"),
        (((700000, 6600000), (700000, 6600010), (700010, 6600010), (700000, 6600000)), "EPSG:2154"),
    ],
)
def test_bdnb_adapter_preserves_subclass_and_detects_crs(monkeypatch, ring, expected_crs):
    raw = {"type": "Polygon", "coordinates": [ring]}
    group = BuildingGroup("group", "construction", geometry=raw, height=6, storeys=2)
    calls = []

    def fetch(query, *, max_buildings):
        calls.append((query, max_buildings))
        return [group]

    provider = BdnbProvider(None)
    monkeypatch.setattr(provider.client, "buildings_for_address", fetch)
    try:
        building = provider.buildings_for_address("Some Address")[0]
        assert provider.buildings_for_address("  some   address ")[0] is building
        assert len(calls) == 1
        assert isinstance(building, BuildingGroup)
        assert building.batiment_construction_id == "construction"
        assert building.crs == expected_crs
        assert group.crs is None
        assert footprints_from_group(building) == footprints_from_group(group)
    finally:
        provider.close()


def test_os_feature_keeps_native_identifier_and_replacement_semantics():
    building = building_from_feature(
        {
            "id": "fallback",
            "properties": {"osid": "os-native-id"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)]],
            },
        }
    )
    assert isinstance(building, OsBuilding)
    assert building.os_id == "os-native-id"
    assert building.code == "os-os-native-id"
    assert building.country == "UK"
    assert building.crs == "EPSG:27700"
    assert building.provider == "ordnance_survey"
    updated = replace(building, os_id="changed", height=9)
    assert updated.code == "os-changed"
    assert updated.height == 9
    assert updated.geometry == building.geometry
