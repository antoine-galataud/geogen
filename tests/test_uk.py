"""Offline OS contract fixtures and full exporter integration tests."""

import json
from urllib.parse import parse_qs, urlsplit

import openstudio
import pytest
import responses
from click.testing import CliRunner

from geogen.cli import main
from geogen.geometry import footprints_from_group
from geogen.models import AddressNotFoundError, ProviderError
from geogen.ordnance_survey import (
    BNG_URI,
    NGD_URL,
    PLACES_URL,
    OsClient,
    OsError,
    building_from_feature,
)
from geogen.osm import build_model
from geogen.providers import discover_country

ITEMS = f"{NGD_URL}/collections/bld-fts-building-4/items"
ADDRESS = "10 Downing Street, London SW1A 2AA"


def feature(code="building-1", offset=0):
    x, y = 530000 + offset, 180000
    return {
        "type": "Feature",
        "id": code,
        "properties": {
            "osid": code,
            "height_relativeroofbase_m": 9,
            "height_relativemax_m": 12,
            "numberoffloors": 3,
            "height_absolutemin_m": 15,
            "geometry_area_m2": 200,
            "roofshapeaspect_shape": "Flat",
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[x, y], [x + 20, y], [x + 20, y + 10], [x, y + 10], [x, y]]],
        },
    }


def address(uprn="100", offset=0, **kwargs):
    return {
        "UPRN": uprn,
        "ADDRESS": ADDRESS,
        "MATCH": 1.0,
        "COUNTRY_CODE": "E",
        "X_COORDINATE": 530005 + offset,
        "Y_COORDINATE": 180005,
        **kwargs,
    }


def register(buildings=None, addresses=None):
    responses.get(
        f"{PLACES_URL}/find", json={"results": [{"DPA": row} for row in (addresses or [address()])]}
    )
    responses.get(
        ITEMS,
        json={
            "type": "FeatureCollection",
            "features": buildings if buildings is not None else [feature()],
        },
    )


@pytest.mark.parametrize(
    "query,country",
    [
        ("10 Downing Street SW1A2AA", "UK"),
        ("London, UK", "UK"),
        ("Edinburgh, Scotland", "UK"),
        ("Cardiff, Wales", "UK"),
        ("122 Rue Amelot 75011 Paris", "FR"),
        ("1 rue de la Paix Paris", "FR"),
        ("1 avenue Victor Hugo, France", "FR"),
    ],
)
def test_routing(query, country):
    assert discover_country(query) == country


@pytest.mark.parametrize("query", ["10 High Street", "Paris", "SW1A2AA, France", ""])
def test_ambiguous_country_requires_context(query):
    with pytest.raises(ProviderError):
        discover_country(query)


@pytest.mark.parametrize("query", ["Belfast BT1 1AA", "Jersey JE2 3AA", "Northern Ireland"])
def test_coverage_rejected_without_api_calls(query):
    with pytest.raises(ProviderError, match="Great Britain"):
        discover_country(query)


def test_override():
    assert discover_country("10 High Street", "GB") == "UK"


@responses.activate
def test_os_request_contract_and_cache():
    register()
    with OsClient("secret") as client:
        first = client.buildings_for_address(ADDRESS)
        assert client.buildings_for_address(ADDRESS.lower()) == first
    assert len(responses.calls) == 2
    request = responses.calls[1].request
    params = parse_qs(urlsplit(request.url).query)
    assert params["crs"] == [BNG_URI]
    assert params["bbox-crs"] == [BNG_URI]
    assert params["limit"] == ["100"]
    assert request.headers["key"] == "secret"
    assert "secret" not in request.url
    assert first[0].crs == "EPSG:27700"


@responses.activate
def test_neighbour_is_not_selected():
    register(buildings=[feature(offset=100)])
    with OsClient("secret") as client, pytest.raises(AddressNotFoundError):
        client.buildings_for_address(ADDRESS)


@responses.activate
def test_pagination_and_building_limit():
    responses.get(f"{PLACES_URL}/find", json={"results": [{"DPA": address()}]})
    responses.get(
        ITEMS,
        json={"features": [feature()], "links": [{"rel": "next", "href": ITEMS + "?offset=100"}]},
    )
    responses.get(ITEMS + "?offset=100", json={"features": [feature("second")]})
    with OsClient("secret") as client:
        assert len(client.buildings_for_address(ADDRESS, max_buildings=1)) == 1
        assert len(client.buildings_for_address(ADDRESS, max_buildings=5)) == 2
    assert len(responses.calls) == 3


@responses.activate
def test_ambiguous_places_result():
    responses.get(
        f"{PLACES_URL}/find", json={"results": [{"DPA": address()}, {"DPA": address("200")}]}
    )
    with OsClient("secret") as client, pytest.raises(OsError, match="Ambiguous"):
        client.buildings_for_address(ADDRESS)
    assert len(responses.calls) == 1


@pytest.mark.parametrize("status", [401, 403, 429, 500])
@responses.activate
def test_api_errors(status):
    responses.get(f"{PLACES_URL}/find", status=status)
    with OsClient("secret") as client, pytest.raises(OsError, match=str(status)) as error:
        client.buildings_for_address(ADDRESS)
    assert "secret" not in str(error.value)


def test_missing_key():
    with pytest.raises(OsError, match="OS_API_KEY"):
        OsClient(None)


def test_normalization_and_fallbacks():
    raw = feature()
    raw["properties"] = {
        "height_relativemax_m": 12,
        "numberoffloors": "NaN",
        "roofshapeaspect_shape": "Pitched",
    }
    building = building_from_feature(raw)
    assert building.height == 12
    assert building.storeys is None
    assert building.roof_shape == "flat"  # avoid double-counting the roof
    assert footprints_from_group(building)[0].storeys == 4


def test_bng_dimensions_and_site_location():
    footprints = footprints_from_group(building_from_feature(feature()))
    model = build_model(footprints)
    assert model.getSite().latitude() == pytest.approx(51.503, abs=0.01)
    assert model.getSite().longitude() == pytest.approx(-0.128, abs=0.01)
    assert model.getSpaces()[0].floorArea() == pytest.approx(200)
    assert len(model.getSpaces()) == 3


@pytest.mark.parametrize("output_format", ["osm", "idf"])
@responses.activate
def test_uk_cli_all_exports(tmp_path, output_format):
    register()
    output, svg = tmp_path / f"uk.{output_format}", tmp_path / "uk.svg"
    metadata = tmp_path / "uk.json"
    result = CliRunner(env={"OS_API_KEY": "secret"}).invoke(
        main,
        [
            ADDRESS,
            ADDRESS,
            "--output-format",
            output_format,
            "-o",
            str(output),
            "--svg-output",
            str(svg),
            "--json-output",
            str(metadata),
            "--window-to-wall-ratio",
            "0.2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "1 building(s)" in result.output
    assert "3 space(s)" in result.output
    assert output.exists() and svg.exists()
    data = json.loads(metadata.read_text())
    building = data["buildings"][0]
    assert data["building_count"] == 1
    assert data["total_estimated_floor_area_m2"] == 600
    assert building["building_id"] == "os-building-1"
    assert building["os_id"] == "building-1"
    assert building["provider"] == "ordnance_survey"
    assert building["country"] == "UK"
    assert building["roof_shape"] == "flat"
    assert "bdnb_id" not in building
    assert ADDRESS in building["description"]
    assert "window" in svg.read_text()
    assert len(responses.calls) == 2
    if output_format == "idf":
        assert openstudio.IdfFile.load(openstudio.path(str(output))).is_initialized()
    else:
        model = openstudio.osversion.VersionTranslator().loadModel(openstudio.path(str(output)))
        assert model.is_initialized()
        assert len(model.get().getSubSurfaces()) == 12


@responses.activate
def test_multiple_uk_buildings(tmp_path):
    responses.get(f"{PLACES_URL}/find", json={"results": [{"DPA": address()}]})
    responses.get(f"{PLACES_URL}/find", json={"results": [{"DPA": address("200", 100)}]})
    responses.get(ITEMS, json={"features": [feature()]})
    responses.get(ITEMS, json={"features": [feature("second", 100)]})
    result = CliRunner().invoke(
        main,
        [
            ADDRESS,
            "12 Downing Street SW1A 2AA",
            "--os-api-key",
            "secret",
            "-o",
            str(tmp_path / "multi.osm"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "2 building(s)" in result.output
    assert "6 space(s)" in result.output


@responses.activate
def test_mixed_countries_share_projected_frame(tmp_path, building_row):
    from tests.conftest import BASE_URL
    from tests.test_cli import _register_api

    _register_api(building_row)
    register()
    output = tmp_path / "mixed.osm"
    result = CliRunner().invoke(
        main,
        [
            "1 rue de la Paix Paris",
            ADDRESS,
            "--base-url",
            BASE_URL,
            "--os-api-key",
            "secret",
            "-o",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "2 building(s)" in result.output
    assert "7 space(s)" in result.output


@responses.activate
def test_out_of_coverage_places_response():
    responses.get(f"{PLACES_URL}/find", json={"results": [{"DPA": address(COUNTRY_CODE="N")}]})
    with OsClient("secret") as client, pytest.raises(OsError, match="Great Britain"):
        client.buildings_for_address("Belfast")
    assert len(responses.calls) == 1


@responses.activate
def test_bad_json():
    responses.get(f"{PLACES_URL}/find", body="not json")
    with OsClient("secret") as client, pytest.raises(OsError, match="invalid JSON"):
        client.buildings_for_address(ADDRESS)


@responses.activate
def test_pagination_cannot_forward_key_to_another_origin():
    responses.get(f"{PLACES_URL}/find", json={"results": [{"DPA": address()}]})
    responses.get(
        ITEMS,
        json={
            "features": [feature()],
            "links": [{"rel": "next", "href": "https://example.com/items"}],
        },
    )
    with OsClient("secret") as client, pytest.raises(OsError, match="pagination"):
        client.buildings_for_address(ADDRESS)
    assert len(responses.calls) == 2


def test_multipolygon_and_pitched_roof():
    raw = feature()
    raw["geometry"] = {
        "type": "MultiPolygon",
        "coordinates": [
            feature()["geometry"]["coordinates"],
            feature(offset=100)["geometry"]["coordinates"],
        ],
    }
    raw["properties"]["roofshapeaspect_shape"] = "Pitched"
    footprints = footprints_from_group(building_from_feature(raw))
    assert len(footprints) == 2
    model = build_model(footprints)
    assert len(model.getSpaces()) == 8  # three floors and attic per polygon


@responses.activate
def test_no_os_key_does_not_query_france(tmp_path):
    result = CliRunner(env={"OS_API_KEY": ""}).invoke(
        main, [ADDRESS, "-o", str(tmp_path / "missing.osm")]
    )
    assert result.exit_code == 1
    assert "OS_API_KEY" in result.output
    assert len(responses.calls) == 0
