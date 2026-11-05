"""Optional representative-DPE thermal fields retain their actual semantics."""

import pytest
import responses

from geogen.bdnb import DPE_TABLE, THERMAL_COLUMNS, BdnbClient, building_group_from_row
from geogen.envelope import envelope_from_group
from geogen.metadata import building_metadata
from geogen.reconstruction.cache import building_snapshot, restore_buildings
from tests.conftest import BASE_URL, unknown_column_error


def test_thermal_fields_and_replay_preserve_raw_floor_u_and_window_uw():
    building = building_group_from_row(
        {
            "batiment_groupe_id": "g",
            "u_mur_exterieur": "0.4",
            "u_plancher_haut_deperditif": 0.25,
            "u_plancher_bas_brut_deperditif": 0.5,
            "u_plancher_bas_final_deperditif": 0.1,
            "uw": 1.6,
            "u_baie_vitree": 1.4,
            "facteur_solaire_baie_vitree": 0.6,
        }
    )
    assert building.wall_u_value == 0.4
    assert building.roof_u_value == 0.25
    assert building.floor_u_value == 0.5
    assert building.window_u_value == 1.6
    assert building.window_solar_factor == 0.6
    assert restore_buildings(building_snapshot({"FR:g": building})) == {"FR:g": building}
    assert building_metadata(building)["thermal_properties"]["window_uw_w_m2k"] == 1.6


def test_ground_adjusted_u_and_shutter_ujn_do_not_replace_missing_inputs():
    building = building_group_from_row(
        {"u_plancher_bas_final_deperditif": 0.2, "u_baie_vitree": 1.4}
    )
    assert building.floor_u_value is None
    assert building.window_u_value is None


def test_thermal_attributes_do_not_break_legacy_geometry_cache_keys():
    known = envelope_from_group(building_group_from_row({"u_mur_exterieur": 0.4}))
    unknown = envelope_from_group(building_group_from_row({}))
    assert known.thermal_properties["values"]["wall_u_value"] == 0.4
    assert hash(known) == hash(unknown)


@pytest.mark.parametrize("value", [None, "", "nan", float("inf"), True, -0.5])
def test_invalid_thermal_inputs_remain_unknown(value):
    building = building_group_from_row({column: value for column in THERMAL_COLUMNS.values()})
    assert all(getattr(building, field) is None for field in THERMAL_COLUMNS)


def test_zero_transmittance_is_invalid_but_zero_solar_factor_is_valid():
    building = building_group_from_row({column: 0 for column in THERMAL_COLUMNS.values()})
    assert building.wall_u_value is None
    assert building.window_solar_factor == 0
    assert building_group_from_row({"facteur_solaire_baie_vitree": 2}).window_solar_factor is None


@responses.activate
def test_missing_numeric_column_is_optional_and_cached(building_row):
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_complet", json=[building_row])
    responses.get(
        f"{BASE_URL}/{DPE_TABLE}",
        status=400,
        json=unknown_column_error("batiment_groupe_dpe_representatif_logement", "uw"),
    )
    responses.get(f"{BASE_URL}/{DPE_TABLE}", json=[{"u_mur_exterieur": 0.4}])
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_wall_dict", json=[])
    with BdnbClient(None, base_url=BASE_URL) as client:
        first = client.building_group("bdnb-0001")
        second = client.building_group("bdnb-0001")
    assert first.wall_u_value == second.wall_u_value == 0.4
    assert first.window_u_value is second.window_u_value is None
    requests = [call.request for call in responses.calls if DPE_TABLE in call.request.url]
    assert len(requests) == 3
    assert "uw" in requests[0].params["select"].split(",")
    assert all("uw" not in request.params["select"].split(",") for request in requests[1:])
