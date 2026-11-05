"""Acquisition contracts and typed source identity regression tests."""

from urllib.parse import parse_qs, urlsplit

import pytest
import requests
import responses
from shapely import from_wkt as shape_from_wkt
from shapely.geometry import MultiPolygon, box, mapping, shape

from geogen.bdnb import RELATIONSHIP_TABLES, BdnbClient, BdnbError, building_group_from_row
from geogen.models import BuildingGroup
from geogen.reconstruction.ign import (
    BUILDING_LAYER, TILE_LAYER, WFS_URL, IgnClient, IgnError,
    bounded_bounds, build_pdal_pipeline, copc_url,
)
from geogen.reconstruction.reconcile import linked_bdtopo_ids, resolve_units
from geogen.reconstruction.types import ReconstructionError

BASE = "https://bdnb.example/v1"
TILE = "https://data.geopf.fr/telechargement/download/LiDARHD/tile.copc.laz"


def group(identifier="g1", geometry=None, **kwargs):
    return BuildingGroup(
        batiment_groupe_id=identifier, geometry=mapping(box(650000, 6860000, 650010, 6860010))
        if geometry is None else mapping(geometry), crs="EPSG:2154", **kwargs,
    )


def feature(identifier="BAT1", geometry=None, **properties):
    return {
        "type": "Feature", "id": "batiment." + identifier,
        "geometry": mapping(geometry if geometry is not None else box(650000, 6860000, 650010, 6860010)),
        "properties": {"cleabs": identifier, **properties},
    }


def collection(features, matched=None):
    result = {
        "type": "FeatureCollection", "features": features,
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::2154"}},
        "numberReturned": len(features),
    }
    if matched is not None:
        result["numberMatched"] = matched
    return result


def tile(**kwargs):
    return {
        "id": "metadata.1", "url": TILE, "original_url": TILE,
        "horizontal_crs": "EPSG:2154", "vertical_reference": "IGN69",
        "acquisition_epoch": "22LHDKE", "coordonnees_nw": "0650-6861",
        "metadata": {"geometry": mapping(box(650000, 6860000, 651000, 6861000))},
        **kwargs,
    }


@responses.activate
def test_relationship_pages_preserve_all_rows_and_separate_missing_tables():
    first, second = RELATIONSHIP_TABLES
    responses.get(BASE + "/" + first, json=[
        {"batiment_groupe_id": "g1", "batiment_construction_id": "c1", "rnb_id": "r1"},
    ])
    responses.get(BASE + "/" + first, json=[
        {"batiment_groupe_id": "g1", "batiment_construction_id": "c2"},
    ])
    responses.get(BASE + "/" + first, json=[])
    responses.get(BASE + "/" + second, status=404, json={"code": "PGRST205", "message": "missing"})
    result = BdnbClient(None, base_url=BASE).relationship_rows(["g1", "g1"])
    assert len(result["rows"]) == 2
    assert result["rows"][0]["rnb_id"] == "r1"
    assert len(result["pages"]) == 4
    assert result["tables"][first]["status"] == "available"
    assert result["tables"][second]["status"] == "unavailable"
    assert "_source_table" not in result["pages"][0]["payload"][0]
    assert parse_qs(urlsplit(responses.calls[1].request.url).query)["offset"] == ["1"]


@responses.activate
@pytest.mark.parametrize("payload,status", [
    ({"code": "PGRST205"}, 403), ({"message": "missing"}, 404),
    (None, 200), ([{"batiment_groupe_id": "g1"}, "broken"], 200),
])
def test_relationship_errors_are_not_absence(payload, status):
    responses.get(BASE + "/" + RELATIONSHIP_TABLES[0], json=payload, status=status)
    with pytest.raises(BdnbError):
        BdnbClient(None, base_url=BASE).relationship_rows(["g1"])


@responses.activate
def test_relationship_timeout_is_not_absence():
    responses.get(BASE + "/" + RELATIONSHIP_TABLES[0], body=requests.Timeout("timeout"))
    with pytest.raises(BdnbError):
        BdnbClient(None, base_url=BASE).relationship_rows(["g1"])


@responses.activate
def test_optional_explicit_rnb_relationship_table_is_not_an_alias():
    for table in RELATIONSHIP_TABLES:
        responses.get(BASE + "/" + table, json=[])
    endpoint = BASE + "/donnees/rel_batiment_groupe_rnb"
    responses.get(endpoint, json=[{"batiment_groupe_id": "g1", "rnb_id": "R1"}])
    responses.get(endpoint, json=[])
    result = BdnbClient(None, base_url=BASE).relationship_rows(["g1"], include_rnb=True)
    assert result["rows"][0]["rnb_id"] == "R1"
    assert "batiment_construction_id" not in result["rows"][0]


@responses.activate
def test_missing_table_mid_pagination_does_not_return_partial_relationships():
    endpoint = BASE + "/" + RELATIONSHIP_TABLES[0]
    responses.get(endpoint, json=[{"batiment_groupe_id": "g1", "batiment_construction_id": "C1"}])
    responses.get(endpoint, json={"code": "PGRST205"}, status=404)
    with pytest.raises(BdnbError):
        BdnbClient(None, base_url=BASE).relationship_rows(["g1"])


@responses.activate
def test_wfs_pages_are_complete_and_outlines_not_clipped():
    large = box(650000, 6860000, 650030, 6860030)
    responses.get(WFS_URL, json=collection([feature("BAT1", large)], 2))
    responses.get(WFS_URL, json=collection([feature("BAT2")], 2))
    result = IgnClient().buildings((650000, 6860000, 650005, 6860005))
    assert shape(result["features"][0]["geometry"]).equals(large)
    assert len(result["features"]) == 2
    query = parse_qs(urlsplit(responses.calls[1].request.url).query)
    assert query["startIndex"] == ["1"]
    assert query["typeNames"] == [BUILDING_LAYER]
    assert len(result["pages"]) == 2


@responses.activate
def test_unknown_wfs_count_continues_after_server_capped_page():
    responses.get(WFS_URL, json=collection([feature("BAT1")]))
    responses.get(WFS_URL, json=collection([feature("BAT2")]))
    responses.get(WFS_URL, json=collection([]))
    assert len(IgnClient().buildings(ids=["BAT1", "BAT2"])["features"]) == 2
    assert "bbox" not in parse_qs(urlsplit(responses.calls[0].request.url).query)


@responses.activate
def test_repeated_or_incomplete_wfs_page_fails_closed():
    responses.get(WFS_URL, json=collection([feature()], 2))
    responses.get(WFS_URL, json=collection([], 2))
    with pytest.raises(IgnError, match="fewer features"):
        IgnClient().buildings(ids=["BAT1"])


@responses.activate
def test_wfs_crs_mismatch_and_huge_result_count_fail_before_returning_data():
    wrong = collection([feature()], 1)
    wrong["crs"]["properties"]["name"] = "EPSG:4326"
    responses.get(WFS_URL, json=wrong)
    responses.get(WFS_URL, json=collection([], 1000000))
    for code in ["crs_mismatch", "feature_limit"]:
        with pytest.raises(IgnError) as caught:
            IgnClient().buildings(ids=["BAT1"])
        assert caught.value.code == code


@responses.activate
@pytest.mark.parametrize("status,code", [(401, "provider_auth"), (403, "provider_auth"), (503, "provider_unavailable")])
def test_provider_failures_are_distinct(status, code):
    responses.get(WFS_URL, status=status)
    with pytest.raises(IgnError) as caught:
        IgnClient().buildings(ids=["BAT1"])
    assert caught.value.code == code


@responses.activate
def test_provider_timeout_and_malformed_payload_are_distinct():
    responses.get(WFS_URL, body=requests.Timeout("timeout"))
    responses.get(WFS_URL, json={"features": []})
    for code in ["provider_timeout", "malformed_provider_response"]:
        with pytest.raises(IgnError) as caught:
            IgnClient().buildings(ids=["BAT1"])
        assert caught.value.code == code


@responses.activate
def test_actual_ign_tile_schema_and_original_metadata_are_preserved():
    tile_feature = {
        "type": "Feature", "id": "metadata.1",
        "geometry": mapping(box(650000, 6860000, 651000, 6861000)),
        "properties": {
            "url_npl": TILE, "coordonnees_nw": "0650-6861", "code_mission": "22LHDKE",
            "systeme_planimetrique": "LAMB93", "systeme_altimetrique": "IGN69",
            "date_debut_acquisition": "2023-03-03Z", "date_fin_acquisition": "2023-03-03Z",
            "date_edition": "2025-06-06Z", "nombre_points": 1234,
        },
    }
    responses.get(WFS_URL, json=collection([tile_feature], 1))
    result = IgnClient().tiles((650000, 6860000, 650010, 6860010))
    acquired = result["tiles"][0]
    assert acquired["metadata"]["properties"] == tile_feature["properties"]
    assert acquired["original_url"] == TILE
    assert acquired["url"] == TILE.replace("/telechargement/", "/chunk/telechargement/")
    assert acquired["horizontal_crs"] == "EPSG:2154"
    assert acquired["vertical_reference"] == "IGN69"
    assert parse_qs(urlsplit(responses.calls[0].request.url).query)["typeNames"] == [TILE_LAYER]


@responses.activate
def test_no_intersecting_tiles_has_a_distinct_no_coverage_error():
    responses.get(WFS_URL, json=collection([], 0))
    with pytest.raises(IgnError) as caught:
        IgnClient().tiles((650000, 6860000, 650010, 6860010))
    assert caught.value.code == "no_coverage"
    from geogen.reconstruction.ign import validate_acquisition
    with pytest.raises(IgnError) as outside:
        validate_acquisition([tile()], (652000, 6860000, 652010, 6860010))
    assert outside.value.code == "no_coverage"


@pytest.mark.parametrize("url", [
    "http://data.geopf.fr/telechargement/a.copc.laz",
    "https://data.geopf.fr.evil.example/telechargement/a.copc.laz",
    "https://data.geopf.fr@evil.example/telechargement/a.copc.laz",
    "******data.geopf.fr/telechargement/a.copc.laz",
    "https://data.geopf.fr/elsewhere/a.copc.laz",
    "https://data.geopf.fr/telechargement/a.copc.laz?token=secret",
    "https://data.geopf.fr/telechargement/%2e%2e/a.copc.laz",
    "file:///somewhere/a.copc.laz",
])
def test_invalid_tile_urls_fail_closed(url):
    with pytest.raises(IgnError):
        copc_url(url)


def test_chunk_url_rewrite_is_exact_and_idempotent():
    rewritten = TILE.replace("/telechargement/", "/chunk/telechargement/")
    assert copc_url(TILE) == rewritten
    assert copc_url(rewritten) == rewritten


def test_two_pdal_readers_have_explicit_merge_and_original_class_artifact():
    second = tile(
        id="metadata.2", coordonnees_nw="0651-6861", url=TILE.replace("tile.", "other."),
        metadata={"geometry": mapping(box(651000, 6860000, 652000, 6861000))},
    )
    stages = build_pdal_pipeline(
        [tile(), second], (650000, 6860000, 650010, 6860010), "crop.original.laz"
    )["pipeline"]
    assert stages[2]["type"] == "filters.merge"
    assert stages[2]["inputs"] == ["tile_0", "tile_1"]
    assert [stage["type"] for stage in stages[3:]] == ["filters.crop", "writers.las"]
    assert stages[-1]["filename"] == "crop.original.laz"
    assert all(stages[-1]["scale_" + axis] == .001 for axis in "xyz")
    assert all(stages[-1]["offset_" + axis] == "auto" for axis in "xyz")
    assert not any(stage["type"] == "filters.expression" for stage in stages)


def test_pdal_crop_polygon_restricts_readers_and_crop_to_one_region():
    parts = MultiPolygon(
        [box(650000, 6860000, 650002, 6860002), box(650008, 6860008, 650010, 6860010)]
    )
    stages = build_pdal_pipeline(
        [tile()], (650000, 6860000, 650010, 6860010), "crop.laz", polygon=parts
    )["pipeline"]
    assert shape_from_wkt(stages[0]["polygon"]).equals(parts)
    crop = stages[2]
    assert crop["type"] == "filters.crop" and "bounds" not in crop
    assert shape_from_wkt(crop["polygon"]).equals(parts)
    with pytest.raises(IgnError) as caught:
        build_pdal_pipeline(
            [tile()], (650000, 6860000, 650010, 6860010), "crop.laz",
            polygon=box(650000, 6860000, 650020, 6860010),
        )
    assert caught.value.code == "invalid_extent"


@pytest.mark.parametrize("change,code", [
    ({"horizontal_crs": "EPSG:4326"}, "crs_mismatch"),
    ({"vertical_reference": "ELLIPSOID"}, "vertical_reference_mismatch"),
    ({"acquisition_epoch": "23LHDKE"}, "acquisition_epoch_conflict"),
])
def test_pdal_coordinate_and_epoch_conflicts_fail_closed(change, code):
    with pytest.raises(IgnError) as caught:
        build_pdal_pipeline([tile(), tile(**change)], (650000, 6860000, 650010, 6860010), "crop.laz")
    assert caught.value.code == code


def test_tile_footprint_union_must_cover_complete_crop_and_preserve_holes():
    from geogen.reconstruction.ign import validate_acquisition
    first = tile()
    second = tile(
        id="metadata.2", coordonnees_nw="0651-6861", url=TILE.replace("tile.", "other."),
        metadata={"geometry": mapping(box(651000, 6860000, 652000, 6861000))},
    )
    bounds = (650990, 6860001, 651010, 6860010)
    assert len(validate_acquisition([first, second], bounds)) == 2
    with pytest.raises(IgnError) as caught:
        validate_acquisition([first], bounds)
    assert caught.value.code == "tile_coverage_incomplete"
    first["metadata"]["geometry"] = mapping(
        box(650000, 6860000, 651000, 6861000).difference(box(650005, 6860005, 650008, 6860008))
    )
    with pytest.raises(IgnError, match="does not cover"):
        validate_acquisition([first], (650001, 6860001, 650010, 6860010))


def header_bytes(crs):
    import io
    import laspy
    from pyproj import CRS

    header = laspy.LasHeader(point_format=6, version="1.4")
    if crs is not None:
        header.add_crs(CRS.from_user_input(crs))
    output = io.BytesIO()
    header.write_to(output)
    return output.getvalue()


def serve_header(data):
    def callback(request):
        last = int(request.headers["Range"].split("-")[-1])
        return 206, {"Content-Range": f"bytes 0-{last}/{len(data)}"}, data[:last + 1]
    responses.add_callback(responses.GET, copc_url(TILE), callback=callback)


@responses.activate
@pytest.mark.parametrize("crs,source", [
    ("EPSG:2154", "ign_catalog_metadata"), ("EPSG:2154+5720", "las_header"),
])
def test_original_source_header_is_checked_before_pdal_merge(crs, source):
    data = header_bytes(crs)
    serve_header(data)
    record, = IgnClient().verify_tile_headers([tile()])
    assert record["horizontal_crs"] == "EPSG:2154"
    assert record["vertical_reference"] == "IGN69"
    assert record["vertical_reference_source"] == source
    assert len(record["header_sha256"]) == 64
    assert len(responses.calls) == 2


@responses.activate
@pytest.mark.parametrize("crs,code", [
    (None, "source_reference_unverified"), ("EPSG:27700", "source_crs_mismatch"),
    ("EPSG:2154+5773", "source_vertical_reference_mismatch"),
])
def test_source_header_missing_or_contradictory_reference_fails_closed(crs, code):
    serve_header(header_bytes(crs))
    with pytest.raises(IgnError) as caught:
        IgnClient().verify_tile_headers([tile()])
    assert caught.value.code == code


@responses.activate
def test_header_probe_never_accepts_a_server_ignoring_range():
    responses.get(copc_url(TILE), body=header_bytes("EPSG:2154"), status=200)
    with pytest.raises(IgnError) as caught:
        IgnClient().verify_tile_headers([tile()])
    assert caught.value.code == "range_unavailable"


@pytest.mark.parametrize("change", [
    {}, {"coordonnees_nw": "0651-6861"},
    {"url": TILE.replace("tile.", "other.")},
    {"coordonnees_nw": "0651-6861", "url": TILE.replace("tile.", "other.")},
])
def test_duplicate_tile_identifiers_urls_and_footprints_fail_closed(change):
    from geogen.reconstruction.ign import validate_acquisition
    with pytest.raises(IgnError) as caught:
        validate_acquisition([tile(), tile(**change)], (650001, 6860001, 650010, 6860010))
    assert caught.value.code == "duplicate_tile"


@responses.activate
def test_huge_and_fictitious_placeholders_never_trigger_queries():
    building = group(geometry=box(0, 0, 1000000, 1000000), fictitious_geometry=True)
    with pytest.raises(IgnError, match="Fictitious"):
        bounded_bounds({"g1": building})
    with pytest.raises(IgnError, match="2 km"):
        IgnClient().buildings((0, 0, 1000000, 1000000))
    assert not responses.calls


@responses.activate
def test_source_facade_queries_explicit_ids_then_bounded_context():
    target = feature()
    context = feature("BAT2", box(650020, 6860000, 650030, 6860010))
    responses.get(WFS_URL, json=collection([target], 1))
    responses.get(WFS_URL, json=collection([target, context], 2))
    building = group(fictitious_geometry=True, geometry=box(0, 0, 1000000, 1000000))
    result = IgnClient().buildings_for_sources(
        {"g1": building}, [{"batiment_groupe_id": "g1", "bdtopo_bat_id": "BAT1"}],
    )
    assert len(result["features"]) == 2
    assert len(result["pages"]) == 2
    assert result["query_bounds"] == (649970, 6859970, 650040, 6860040)
    assert "CQL_FILTER" in parse_qs(urlsplit(responses.calls[0].request.url).query)
    assert "bbox" in parse_qs(urlsplit(responses.calls[1].request.url).query)


@responses.activate
def test_source_facade_rejects_fictitious_extent():
    building = group(fictitious_geometry=True, geometry=box(0, 0, 1000000, 1000000))
    with pytest.raises(IgnError, match="Fictitious"):
        IgnClient().buildings_for_sources({"g1": building}, [])
    assert not responses.calls


def test_compatibility_rnb_fallback_never_becomes_construction_identity():
    building = building_group_from_row({
        "batiment_groupe_id": "g1", "rnb_id": "RNB-ONLY",
        "geom_groupe": feature()["geometry"],
    })
    assert building.batiment_construction_id == "RNB-ONLY"
    # Provider normalization supplies a CRS; raw compatibility rows do not.
    from dataclasses import replace
    building = replace(building, crs="EPSG:2154")
    unit, = resolve_units({"g1": building}, [], [feature()])
    assert unit.identity.bdnb_construction_ids == ()
    assert unit.identity.rnb_ids == ()
    assert unit.identity.bdnb_group_ids == ("g1",)


def test_explicit_typed_rows_and_rnb_links_survive():
    rows = [{"batiment_groupe_id": "g1", "batiment_construction_id": "C1", "rnb_id": "R1"}]
    unit, = resolve_units({"g1": group()}, rows, [feature(identifiants_rnb='["R1"]')])
    assert unit.identity.bdnb_construction_ids == ("C1",)
    assert unit.identity.rnb_ids == ("R1",)
    link = next(link for link in unit.links if link.source_type == "bdnb_group")
    assert link.method == "source_relation"
    assert "explicit_rnb" in link.reasons


def test_specified_bdtopo_bat_cleabs_is_an_explicit_identity_field():
    rows = [{"batiment_groupe_id": "g1", "bdtopo_bat_cleabs": "BAT1"}]
    assert linked_bdtopo_ids(rows) == ("BAT1",)
    unit, = resolve_units({"g1": group(fictitious_geometry=True)}, rows, [feature()])
    assert unit.identity.bdtopo_ids == ("BAT1",)
    link = next(link for link in unit.links if link.source_type == "bdnb_group")
    assert link.method == "source_relation"
    assert "explicit_bdtopo" in link.reasons


def test_unrelated_group_constructions_do_not_leak_into_physical_identities():
    rows = [
        {"batiment_groupe_id": "g1", "batiment_construction_id": "C1", "rnb_id": "R1"},
        {"batiment_groupe_id": "g1", "batiment_construction_id": "C2", "rnb_id": "R2"},
    ]
    unit, = resolve_units({"g1": group()}, rows, [feature(identifiants_rnb='["R1"]')])
    assert unit.identity.bdnb_construction_ids == ("C1",)
    assert unit.identity.rnb_ids == ("R1",)
    unknown = [link for link in unit.links if link.source_id in {"C2", "R2"}]
    assert len(unknown) == 2
    assert all(link.relationship == "unresolved" and link.confidence == "unresolved" for link in unknown)


def test_many_groups_one_physical_unit_are_not_duplicates():
    left = box(650000, 6860000, 650005, 6860010)
    right = box(650005, 6860000, 650010, 6860010)
    units = resolve_units({"g1": group("g1", left), "g2": group("g2", right)}, [], [feature()])
    assert len(units) == 1
    assert units[0].identity.bdnb_group_ids == ("g1", "g2")
    links = [link for link in units[0].links if link.source_type == "bdnb_group"]
    assert all(link.source_coverage == 1 and link.physical_coverage == .5 for link in links)
    assert all(link.relationship == "shared_physical_unit" and link.method == "spatial_verified" for link in links)


def test_one_group_multiple_outlines_uses_union_coverage_and_keeps_context():
    left = box(650000, 6860000, 650005, 6860010)
    right = box(650005, 6860000, 650010, 6860010)
    units = resolve_units(
        {"g1": group()}, [], [feature("left", left), feature("right", right),
                              feature("context", box(650050, 6860050, 650060, 6860060))],
    )
    assert len(units) == 3
    assert sum(unit.role == "target" for unit in units) == 2
    assert sum(unit.role == "context" for unit in units) == 1
    assert all(
        link.relationship == "group_member"
        for unit in units for link in unit.links if link.source_type == "bdnb_group"
    )


def test_multipart_ids_and_hashes_stable_under_ring_and_input_order():
    first = box(650000, 6860000, 650010, 6860010)
    second = box(650020, 6860000, 650030, 6860010)
    geometry = MultiPolygon([first, second])
    buildings = {"g1": group(geometry=geometry)}
    original = resolve_units(buildings, [], [feature(geometry=geometry)])
    reordered = resolve_units(buildings, [], [feature(geometry=MultiPolygon([second.reverse(), first.reverse()]))])
    assert len(original) == 2
    assert [(u.reconstruction_id, u.geometry_revision) for u in original] == [
        (u.reconstruction_id, u.geometry_revision) for u in reordered
    ]
    assert all(unit.original_wkt and unit.working_wkt for unit in original)


def test_overlapping_spatial_candidates_remain_ambiguous():
    with pytest.raises(ReconstructionError) as caught:
        resolve_units({"g1": group("g1"), "g2": group("g2")}, [], [feature()])
    assert caught.value.code == "unresolved_identity"


def test_fictitious_group_requires_link():
    building = group(fictitious_geometry=True, geometry=box(0, 0, 1000000, 1000000))
    with pytest.raises(ReconstructionError, match="No unambiguous"):
        resolve_units({"g1": building}, [], [feature()])


def test_native_bdtopo_declares_approximation_without_claiming_known_source_extent():
    unit, = resolve_units({"g1": group()}, [], [feature()])
    assert unit.outline_type == "unknown"
    assert "bdtopo_outline" in unit.decisions
    assert "wall_extent_assumption=roofprint_extruded" in unit.decisions


def test_source_links_use_only_contract_enums():
    unit, = resolve_units(
        {"g1": group()},
        [{"batiment_groupe_id": "g1", "batiment_construction_id": "C1", "rnb_id": "R1"}],
        [feature(identifiants_rnb='["R1"]')],
    )
    assert all(link.method in {"source_relation", "spatial_verified", "manual"} for link in unit.links)
    assert all(link.relationship in {
        "one_to_one", "group_member", "shared_physical_unit", "unresolved",
    } for link in unit.links)
    assert all(link.attribute_scope in {
        "group", "construction", "dwelling", "unknown",
    } for link in unit.links)
