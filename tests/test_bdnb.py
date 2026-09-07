"""Tests of the BDNB API client."""

from __future__ import annotations

import pytest
import responses

from geogen.bdnb import (
    ENVELOPE_COLUMNS,
    AddressNotFoundError,
    BdnbClient,
    BdnbError,
    BuildingGroup,
    unknown_column,
    unknown_table,
)
from tests.conftest import BASE_URL, unknown_column_error, unknown_table_error


@pytest.fixture
def client() -> BdnbClient:
    return BdnbClient("secret-key", base_url=BASE_URL)


@responses.activate
def test_geocode_sends_the_api_key_and_parses_features(client: BdnbClient) -> None:
    responses.get(
        f"{BASE_URL}/geocodage",
        json={
            "type": "FeatureCollection",
            "features": [
                {
                    "properties": {
                        "id": "75102_7043_00001",
                        "label": "1 rue de la Paix 75002 Paris",
                    }
                }
            ],
        },
    )

    addresses = client.geocode("1 rue de la Paix Paris")

    assert [address.cle_interop_adr for address in addresses] == ["75102_7043_00001"]
    assert addresses[0].label == "1 rue de la Paix 75002 Paris"
    request = responses.calls[0].request
    assert request.headers["X-Gravitee-Api-Key"] == "secret-key"
    assert "q=1+rue+de+la+Paix+Paris" in request.url


@responses.activate
def test_find_addresses_falls_back_on_the_adresse_table(client: BdnbClient) -> None:
    responses.get(f"{BASE_URL}/geocodage", status=404)
    responses.get(
        f"{BASE_URL}/donnees/adresse",
        json=[{"cle_interop_adr": "75102_7043_00001", "libelle_adresse": "1 rue de la Paix"}],
    )

    addresses = client.find_addresses("1 rue de la Paix, Paris")

    assert [address.cle_interop_adr for address in addresses] == ["75102_7043_00001"]
    assert "libelle_adresse=ilike.%2A1%2Arue%2Ade%2Ala%2APaix%2AParis%2A" in (
        responses.calls[1].request.url
    )


@responses.activate
def test_building_group_ids_filters_on_the_address_key(client: BdnbClient) -> None:
    responses.get(
        f"{BASE_URL}/donnees/rel_batiment_groupe_adresse",
        json=[{"batiment_groupe_id": "bdnb-0001"}, {"batiment_groupe_id": "bdnb-0001"}],
    )

    assert client.building_group_ids("75102_7043_00001") == ["bdnb-0001"]
    assert "cle_interop_adr=eq.75102_7043_00001" in responses.calls[0].request.url


@responses.activate
def test_building_group_parses_a_row(client: BdnbClient, building_row: dict) -> None:
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_complet", json=[building_row])

    building = client.building_group("bdnb-0001")

    assert building == BuildingGroup(
        batiment_groupe_id="bdnb-0001",
        geometry=building_row["geom_groupe"],
        height=9.0,
        storeys=3,
        ground_elevation=35.0,
        footprint_area=200.0,
        fictitious_geometry=False,
        address="1 rue de la Paix 75002 Paris",
        city="Paris",
        glazed_ratio=20.0,
        glazing_orientations=("nord", "sud"),
        roof_material="tuiles",
        roof_type="Combles perdus",
    )
    assert "batiment_groupe_id=eq.bdnb-0001" in responses.calls[0].request.url
    assert "geom_groupe" in responses.calls[0].request.url


@responses.activate
def test_building_group_returns_none_when_unknown(client: BdnbClient) -> None:
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_complet", json=[])

    assert client.building_group("bdnb-0001") is None


@responses.activate
def test_buildings_for_address(client: BdnbClient, building_row: dict) -> None:
    responses.get(
        f"{BASE_URL}/geocodage",
        json={"features": [{"properties": {"id": "75102_7043_00001"}}]},
    )
    responses.get(
        f"{BASE_URL}/donnees/rel_batiment_groupe_adresse",
        json=[{"batiment_groupe_id": "bdnb-0001"}],
    )
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_complet", json=[building_row])

    buildings = client.buildings_for_address("1 rue de la Paix Paris")

    assert [building.batiment_groupe_id for building in buildings] == ["bdnb-0001"]


@responses.activate
def test_buildings_for_unknown_address(client: BdnbClient) -> None:
    responses.get(f"{BASE_URL}/geocodage", json={"features": []})
    responses.get(f"{BASE_URL}/donnees/adresse", json=[])

    with pytest.raises(AddressNotFoundError):
        client.buildings_for_address("nowhere")


@responses.activate
def test_server_errors_are_wrapped(client: BdnbClient) -> None:
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_complet", status=500)

    with pytest.raises(BdnbError):
        client.building_group("bdnb-0001")


@responses.activate
def test_unexpected_payloads_are_rejected(client: BdnbClient) -> None:
    responses.get(f"{BASE_URL}/donnees/rel_batiment_groupe_adresse", json={"message": "nope"})

    with pytest.raises(BdnbError):
        client.building_group_ids("75102_7043_00001")


@responses.activate
def test_geocode_ignores_malformed_features(client: BdnbClient) -> None:
    responses.get(f"{BASE_URL}/geocodage", json={"features": [{}, "oops", {"properties": None}]})

    assert client.geocode("1 rue de la Paix Paris") == []


@responses.activate
def test_building_group_downloads_the_walls(
    client: BdnbClient, building_row: dict, wall_dict_row: dict
) -> None:
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_complet", json=[building_row])
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_wall_dict", json=[wall_dict_row])

    building = client.building_group("bdnb-0001")

    assert building is not None
    assert building.walls == tuple(wall_dict_row["wall_dict"])
    assert "batiment_groupe_id=eq.bdnb-0001" in responses.calls[1].request.url


@responses.activate
def test_building_group_walls_are_optional(client: BdnbClient, building_row: dict) -> None:
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_complet", json=[building_row])
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_wall_dict", status=404)

    building = client.building_group("bdnb-0001")

    assert building is not None
    assert building.walls == ()


@responses.activate
def test_building_walls_parse_a_serialized_dict(client: BdnbClient) -> None:
    responses.get(
        f"{BASE_URL}/donnees/batiment_groupe_wall_dict",
        json=[{"wall_dict": '[{"wall_type": "roof", "inclination": 150.0}, "oops"]'}],
    )

    assert client.building_walls("bdnb-0001") == ({"wall_type": "roof", "inclination": 150.0},)


@responses.activate
def test_building_group_skips_the_columns_the_bdnb_does_not_publish(
    client: BdnbClient, building_row: dict
) -> None:
    """The open BDNB serves a narrower batiment_groupe_complet than the complete one."""
    published = {
        column: value
        for column, value in building_row.items()
        if column != "materiaux_toiture_simplifie"
    }
    responses.get(
        f"{BASE_URL}/donnees/batiment_groupe_complet",
        json=unknown_column_error("batiment_groupe_complet", "materiaux_toiture_simplifie"),
        status=400,
    )
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_complet", json=[published])
    responses.get(
        f"{BASE_URL}/donnees/batiment_groupe_wall_dict",
        json=unknown_table_error("batiment_groupe_wall_dict"),
        status=404,
    )

    building = client.building_group("bdnb-0001")

    assert building is not None
    assert building.height == 9.0
    assert building.glazed_ratio == 20.0
    assert building.glazing_orientations == ("nord", "sud")
    assert building.roof_material == "Tuiles"
    assert building.walls == ()
    assert "materiaux_toiture_simplifie" not in responses.calls[1].request.url


@responses.activate
def test_the_glazing_is_read_from_the_dpe_when_the_percentage_is_missing(
    client: BdnbClient, building_row: dict, dpe_row: dict
) -> None:
    published = dict(building_row, pourcentage_surface_baie_vitree_exterieur=None)
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_complet", json=[published])
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_dpe_representatif_logement", json=[dpe_row])
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_wall_dict", json=[])

    building = client.building_group("bdnb-0001")

    assert building is not None
    assert building.glazed_ratio is None
    assert building.wall_area == 200.0
    assert building.glazed_areas == (("north", 10.0), ("south", 20.0))
    assert "surface_vitree_nord" in responses.calls[1].request.url


@responses.activate
def test_the_dpe_is_not_read_when_the_percentage_is_known(
    client: BdnbClient, building_row: dict
) -> None:
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_complet", json=[building_row])
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_wall_dict", json=[])

    client.building_group("bdnb-0001")

    assert not any("dpe_representatif_logement" in call.request.url for call in responses.calls)


@responses.activate
def test_the_roof_material_is_read_from_its_own_table(
    client: BdnbClient, building_row: dict
) -> None:
    published = dict(
        building_row,
        materiaux_toiture_simplifie=None,
        mat_toit_txt=None,
        type_plancher_haut_deperditif=None,
    )
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_complet", json=[published])
    responses.get(
        f"{BASE_URL}/donnees/batiment_groupe_ffo_bat", json=[{"mat_toit_txt": "Ardoises"}]
    )
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_wall_dict", json=[])

    building = client.building_group("bdnb-0001")

    assert building is not None
    assert building.roof_material == "Ardoises"
    ffo_call = responses.calls[1].request.url
    assert "mat_toit_txt" in ffo_call
    assert not any(
        "batiment_groupe_synthese_enveloppe" in call.request.url for call in responses.calls
    )


@responses.activate
def test_columns_and_tables_are_only_rejected_once(client: BdnbClient, building_row: dict) -> None:
    published = {
        column: value for column, value in building_row.items() if column not in ENVELOPE_COLUMNS
    }
    for column in ENVELOPE_COLUMNS:
        responses.get(
            f"{BASE_URL}/donnees/batiment_groupe_complet",
            json=unknown_column_error("batiment_groupe_complet", column),
            status=400,
        )
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_complet", json=[published])
    tables = (
        "batiment_groupe_dpe_representatif_logement",
        "batiment_groupe_ffo_bat",
        "batiment_groupe_synthese_enveloppe",
        "batiment_groupe_wall_dict",
    )
    for table in tables:
        responses.get(f"{BASE_URL}/donnees/{table}", json=unknown_table_error(table), status=404)

    first = client.building_group("bdnb-0001")
    second = client.building_group("bdnb-0002")

    assert first is not None and second is not None
    assert first.glazed_ratio is None and second.glazed_ratio is None
    urls = [call.request.url for call in responses.calls]
    rejected = len(ENVELOPE_COLUMNS) + len(tables)
    assert len(urls) == rejected + 2
    assert "pourcentage_surface_baie_vitree_exterieur" not in urls[-1]


@responses.activate
def test_a_missing_required_column_is_an_error(client: BdnbClient) -> None:
    responses.get(
        f"{BASE_URL}/donnees/batiment_groupe_complet",
        json=unknown_column_error("batiment_groupe_complet", "geom_groupe"),
        status=400,
    )

    with pytest.raises(BdnbError, match="geom_groupe"):
        client.building_group("bdnb-0001")


@responses.activate
def test_an_unrelated_error_is_not_swallowed(client: BdnbClient) -> None:
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_complet", status=500)

    with pytest.raises(BdnbError, match="500"):
        client.building_group("bdnb-0001")


def test_unknown_column_reads_the_postgrest_error() -> None:
    error = BdnbError(
        "boom",
        status=400,
        payload=unknown_column_error("batiment_groupe_complet", "surface_facade_vitree"),
    )

    assert error.code == "42703"
    assert unknown_column(error) == "surface_facade_vitree"
    assert not unknown_table(error)


def test_unknown_table_reads_the_postgrest_error() -> None:
    error = BdnbError("boom", status=404, payload=unknown_table_error("batiment_groupe_wall_dict"))

    assert unknown_table(error)
    assert unknown_column(error) is None


def test_an_error_without_a_body_is_neither_a_column_nor_a_table() -> None:
    error = BdnbError("boom", status=500, payload=None)

    assert error.code is None
    assert error.detail == ""
    assert unknown_column(error) is None
    assert not unknown_table(error)


def test_the_code_of_a_building_group_is_its_identifier() -> None:
    assert BuildingGroup(batiment_groupe_id="bdnb-0001").code == "bdnb-0001"


def test_the_code_falls_back_on_the_building_when_it_has_no_group() -> None:
    building = BuildingGroup.from_row({"batiment_groupe_id": None, "rnb_id": "AB12CD34EF56"})

    assert building.batiment_construction_id == "AB12CD34EF56"
    assert building.code == "AB12CD34EF56"


def test_a_building_group_without_any_code() -> None:
    assert BuildingGroup.from_row({}).code == ""
