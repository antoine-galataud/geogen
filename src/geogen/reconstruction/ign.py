"""Bounded, paginated IGN acquisition without truncating physical outlines."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import struct
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlsplit, urlunsplit

import requests
from pyproj import CRS
from shapely import normalize
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from geogen.geometry import parse_geometry
from geogen.models import Building
from geogen.reconstruction.types import ReconstructionError

WFS_URL = "https://data.geopf.fr/wfs/ows"
BUILDING_LAYER = "BDTOPO_V3:batiment"
TILE_LAYER = "IGNF_LIDAR-HD_METADONNEE:metadata"
MAX_EXTENT = 2000.0
MAX_FEATURES = 2000
MAX_TILES = 32
PAGE_SIZE = 500
NEIGHBORHOOD_BUFFER = 30.0


class IgnError(ReconstructionError):
    """Provider availability is distinct from absent buildings or tile coverage."""


def checked_bounds(bounds: Iterable[float]) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(value) for value in bounds)
    except (TypeError, ValueError) as error:
        raise IgnError("invalid_extent", "Expected four finite projected coordinates") from error
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise IgnError("invalid_extent", "Expected four finite projected coordinates")
    west, south, east, north = values
    if east <= west or north <= south or east - west > MAX_EXTENT or north - south > MAX_EXTENT:
        raise IgnError("extent_limit", "Acquisition extent must be positive and at most 2 km per axis")
    return values


def bounded_bounds(buildings: dict[str, Building] | Iterable[Building]):
    """Never use a fictitious group's potentially enormous placeholder to query IGN."""
    values = buildings.values() if isinstance(buildings, dict) else buildings
    extents = []
    for building in values:
        if building.fictitious_geometry:
            raise IgnError("fictitious_outline", "Fictitious groups require linked IDs or a manual outline")
        if building.crs is None or CRS.from_user_input(building.crs) != CRS.from_epsg(2154):
            raise IgnError("crs_mismatch", "Acquisition outlines must explicitly declare EPSG:2154")
        geometry = parse_geometry(building.geometry)
        if geometry.is_empty or not geometry.is_valid:
            raise IgnError("invalid_outline", "Acquisition outline is empty or invalid")
        extents.append(geometry.bounds)
    if not extents:
        raise IgnError("missing_outline", "No trustworthy neighborhood outline")
    return checked_bounds((
        min(item[0] for item in extents) - NEIGHBORHOOD_BUFFER,
        min(item[1] for item in extents) - NEIGHBORHOOD_BUFFER,
        max(item[2] for item in extents) + NEIGHBORHOOD_BUFFER,
        max(item[3] for item in extents) + NEIGHBORHOOD_BUFFER,
    ))


def copc_url(url: str) -> str:
    """Use IGN's range endpoint only for its exact public download host and path."""
    if not isinstance(url, str) or any(char.isspace() or ord(char) < 32 for char in url):
        raise IgnError("invalid_tile_url", "Tile URL must be a public IGN HTTPS URL")
    try:
        parts = urlsplit(url)
    except ValueError as error:
        raise IgnError("invalid_tile_url", "Malformed IGN tile URL") from error
    decoded = unquote(parts.path)
    if (
        parts.scheme != "https" or parts.netloc != "data.geopf.fr"
        or parts.username or parts.password or parts.query or parts.fragment
        or not parts.path.startswith(("/telechargement/", "/chunk/telechargement/"))
        or ".." in decoded.split("/") or "\\" in decoded
        or not decoded.lower().endswith(".copc.laz")
    ):
        raise IgnError("invalid_tile_url", "Expected an unsigned data.geopf.fr /telechargement/ COPC URL")
    if parts.path.startswith("/telechargement/"):
        parts = parts._replace(path="/chunk" + parts.path)
    return urlunsplit(parts)


def _horizontal(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"LAMB93", "LAMBERT-93", "LAMBERT 93", "EPSG:2154"}:
        return "EPSG:2154"
    raise IgnError("crs_mismatch", f"Unsupported or missing LiDAR horizontal reference: {value!r}")


def _vertical(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"IGN69", "IGN 69", "NGF-IGN69", "NGF IGN69", "EPSG:5720"}:
        return "IGN69"
    raise IgnError("vertical_reference_mismatch", f"Unsupported or missing LiDAR vertical reference: {value!r}")


def validate_acquisition(
    tiles: Iterable[dict[str, Any]], bounds=None,
) -> tuple[dict[str, Any], ...]:
    tiles = tuple(tiles)
    if not tiles:
        raise IgnError("no_coverage", "No LiDAR tile intersects the requested extraction extent")
    if len(tiles) > MAX_TILES:
        raise IgnError("tile_limit", "LiDAR acquisition exceeds the safe limit of 32 tiles")
    epochs = set()
    tile_epochs: dict[str, tuple[str, str, str]] = {}
    urls, identifiers, footprints = set(), set(), []
    footprint_hashes = set()
    for tile in tiles:
        _horizontal(tile.get("horizontal_crs"))
        _vertical(tile.get("vertical_reference"))
        url = copc_url(tile.get("url"))
        epoch = str(tile.get("acquisition_epoch") or "")
        if not epoch:
            raise IgnError("missing_acquisition_epoch", "LiDAR acquisition epoch is not documented")
        epochs.add(epoch)
        key = str(tile.get("coordonnees_nw") or tile.get("id") or tile["url"])
        interval = (epoch, str(tile.get("date_debut_acquisition") or ""),
                    str(tile.get("date_fin_acquisition") or ""))
        if key in tile_epochs and tile_epochs[key] != interval:
            raise IgnError("acquisition_epoch_conflict", f"Conflicting acquisition epochs for {key}")
        if url in urls or key in identifiers:
            raise IgnError("duplicate_tile", "LiDAR tile inventory contains a duplicate URL or tile identifier")
        urls.add(url)
        identifiers.add(key)
        tile_epochs[key] = interval
        if bounds is not None:
            try:
                geometry = parse_geometry(tile["metadata"]["geometry"])
            except Exception as error:
                raise IgnError("tile_coverage_unverified", "A LiDAR tile lacks a usable footprint") from error
            if (
                not isinstance(geometry, (Polygon, MultiPolygon))
                or geometry.is_empty or not geometry.is_valid
                or not all(math.isfinite(value) for value in geometry.bounds)
            ):
                raise IgnError("tile_coverage_unverified", "LiDAR tile footprints must be valid polygons")
            fingerprint = normalize(geometry).wkb
            if fingerprint in footprint_hashes:
                raise IgnError("duplicate_tile", "LiDAR tile inventory contains a duplicate footprint")
            footprint_hashes.add(fingerprint)
            footprints.append(geometry)
    if len(epochs) != 1:
        raise IgnError("acquisition_epoch_conflict", "Mixing LiDAR acquisition missions is not supported")
    # IGN's reprojected metadata edges differ from their kilometre grid by millimetres.
    if bounds is not None:
        coverage = unary_union(footprints).buffer(.02)
        requested = box(*checked_bounds(bounds))
        if coverage.intersection(requested).area == 0:
            raise IgnError("no_coverage", "No LiDAR tile intersects the requested extraction extent")
        if not coverage.covers(requested):
            raise IgnError("tile_coverage_incomplete", "LiDAR tile footprint union does not cover the extraction bounds")
    return tiles


class IgnClient:
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._session = requests.Session()

    def close(self):
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def _header_range(self, url: str, size: int):
        try:
            with self._session.get(
                url, headers={"Range": f"bytes=0-{size - 1}", "Accept-Encoding": "identity"},
                stream=True, timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                if copc_url(response.url) != url:
                    raise IgnError("invalid_tile_url", "LiDAR header request redirected to a different source")
                content_range = response.headers.get("Content-Range", "")
                match = re.fullmatch(rf"bytes 0-{size - 1}/(\d+)", content_range)
                if response.status_code != 206 or not match or int(match.group(1)) < size:
                    raise IgnError("range_unavailable", "IGN did not honor a bounded header range request")
                data = response.raw.read(size + 1)
                if len(data) != size:
                    raise IgnError("invalid_copc_header", "LiDAR header range has an unexpected length")
                return data, {
                    "url": url, "content_range": content_range,
                    "total_bytes": int(match.group(1)),
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                }
        except requests.Timeout as error:
            raise IgnError("provider_timeout", "IGN LiDAR header request timed out") from error
        except requests.HTTPError as error:
            status = error.response.status_code if error.response is not None else None
            raise IgnError(
                "provider_auth" if status in {401, 403} else "provider_unavailable",
                f"IGN LiDAR header request returned HTTP {status}",
            ) from error
        except requests.RequestException as error:
            raise IgnError("provider_unavailable", "IGN LiDAR header transport failed") from error

    def verify_tile_headers(self, tiles) -> list[dict[str, Any]]:
        """Check each source before merging; read at most 1 MiB of header/VLRs per tile."""
        import laspy

        tiles = tiles["tiles"] if isinstance(tiles, dict) else tiles
        verified = []
        for tile in validate_acquisition(tiles):
            url = copc_url(tile["url"])
            prefix, first_request = self._header_range(url, 375)
            if prefix[:4] != b"LASF" or prefix[24:26] != bytes((1, 4)):
                raise IgnError("invalid_copc_header", "COPC source must have a LAS 1.4 header")
            point_offset = struct.unpack_from("<I", prefix, 96)[0]
            if not 375 <= point_offset <= 1024 * 1024:
                raise IgnError("header_limit", "COPC header/VLR region exceeds the bounded read limit")
            data, second_request = self._header_range(url, point_offset)
            if data[:375] != prefix or any(
                first_request[key] != second_request[key]
                for key in ("etag", "last_modified", "total_bytes")
            ):
                raise IgnError("source_changed", "LiDAR source changed during header acquisition")
            try:
                header = laspy.LasHeader.read_from(io.BytesIO(data), read_evlrs=False)
                crs = header.parse_crs()
            except Exception as error:
                raise IgnError("source_reference_unverified", "Cannot parse the source LAS coordinate reference") from error
            if crs is None:
                raise IgnError("source_reference_unverified", "Source LAS header does not declare a CRS")
            components = crs.sub_crs_list if crs.is_compound else [crs]
            horizontal = [component for component in components if component.is_projected]
            vertical = [component for component in components if component.is_vertical]
            if len(horizontal) != 1 or horizontal[0] != CRS.from_epsg(2154):
                raise IgnError("source_crs_mismatch", "Source LAS horizontal CRS contradicts EPSG:2154 metadata")
            if vertical and (len(vertical) != 1 or vertical[0] != CRS.from_epsg(5720)):
                raise IgnError("source_vertical_reference_mismatch", "Source LAS vertical CRS contradicts IGN69 metadata")
            verified.append({
                "url": url, "header_sha256": hashlib.sha256(data).hexdigest(),
                "horizontal_crs": "EPSG:2154", "vertical_reference": "IGN69",
                "vertical_reference_source": "las_header" if vertical else "ign_catalog_metadata",
                "source_crs_wkt": crs.to_wkt(), "point_count": int(header.point_count),
                "requests": [first_request, second_request],
            })
        return verified

    def _features(self, layer: str, query: dict[str, Any], limit: int):
        features, pages, seen = [], [], set()
        offset, expected = 0, None
        while True:
            params = {
                "service": "WFS", "version": "2.0.0", "request": "GetFeature",
                "typeNames": layer, "outputFormat": "application/json",
                "srsName": "EPSG:2154", "count": PAGE_SIZE, "startIndex": offset,
                "sortBy": "cleabs" if layer == BUILDING_LAYER else "coordonnees_nw",
                **query,
            }
            try:
                response = self._session.get(WFS_URL, params=params, timeout=self.timeout)
                response.raise_for_status()
            except requests.Timeout as error:
                raise IgnError("provider_timeout", "IGN WFS request timed out") from error
            except requests.HTTPError as error:
                status = error.response.status_code if error.response is not None else None
                code = "provider_auth" if status in {401, 403} else "provider_unavailable"
                raise IgnError(code, f"IGN WFS returned HTTP {status}") from error
            except requests.RequestException as error:
                raise IgnError("provider_unavailable", "IGN WFS transport failed") from error
            try:
                payload = response.json()
            except ValueError as error:
                raise IgnError("malformed_provider_response", "IGN WFS returned non-JSON data") from error
            rows = payload.get("features") if isinstance(payload, dict) else None
            if (
                not isinstance(rows, list) or payload.get("type") != "FeatureCollection"
                or any(not isinstance(row, dict) or row.get("type") != "Feature"
                       or not isinstance(row.get("properties"), dict)
                       or not isinstance(row.get("geometry"), dict) for row in rows)
            ):
                raise IgnError("malformed_provider_response", "Invalid IGN WFS FeatureCollection")
            try:
                declared = payload["crs"]["properties"]["name"]
                if CRS.from_user_input(declared) != CRS.from_epsg(2154):
                    raise ValueError("wrong CRS")
            except (KeyError, TypeError, ValueError) as error:
                raise IgnError("crs_mismatch", "IGN response must declare EPSG:2154") from error
            matched = payload.get("numberMatched", payload.get("totalFeatures"))
            if matched not in (None, "unknown"):
                try:
                    matched = int(matched)
                except (ValueError, TypeError) as error:
                    raise IgnError("malformed_provider_response", "Invalid WFS result count") from error
                if matched < 0 or (expected is not None and matched != expected):
                    raise IgnError("incomplete_pagination", "IGN feature count changed during pagination")
                expected = matched
                if expected > limit:
                    raise IgnError("feature_limit", f"IGN query exceeds bounded feature count {limit}")
            if payload.get("numberReturned", len(rows)) != len(rows):
                raise IgnError("malformed_provider_response", "WFS numberReturned disagrees with page")
            pages.append({"url": response.url, "params": params, "payload": payload})
            if not rows:
                if expected is not None and offset != expected:
                    raise IgnError("incomplete_pagination", "IGN returned fewer features than advertised")
                break
            signature = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
            if signature in seen:
                raise IgnError("incomplete_pagination", "IGN WFS pagination did not advance")
            seen.add(signature)
            features.extend(rows)
            offset += len(rows)
            if offset > limit or (expected is not None and offset > expected):
                raise IgnError("feature_limit", "IGN feature count exceeded the advertised or safe bound")
            if expected is not None and offset == expected:
                break
        return {"features": features, "pages": pages, "crs": "EPSG:2154"}

    def buildings(self, bounds=None, *, ids: Iterable[str] = ()):
        ids = sorted(set(ids))
        if ids:
            if len(ids) > 100 or any(not re.fullmatch(r"[\w.:-]+", value) for value in ids):
                raise IgnError("invalid_source_id", "Expected at most 100 valid BD TOPO identifiers")
            query = {"CQL_FILTER": "cleabs IN (" + ",".join("'" + value + "'" for value in ids) + ")"}
        elif bounds is not None:
            query = {"bbox": ",".join(map(str, checked_bounds(bounds))) + ",EPSG:2154"}
        else:
            raise IgnError("missing_outline", "BD TOPO acquisition requires linked IDs or a bounded neighborhood")
        return self._features(BUILDING_LAYER, query, MAX_FEATURES)

    def buildings_for_sources(
        self, buildings: dict[str, Building], relations
    ):
        """Acquire linked outlines plus bounded context, never a placeholder extent."""
        from geogen.reconstruction.reconcile import linked_bdtopo_ids

        ids = linked_bdtopo_ids(relations)
        explicit = self.buildings(ids=ids) if ids else None
        trustworthy = dict(buildings)
        if explicit is not None:
            acquired = {str(feature["properties"].get("cleabs") or "") for feature in explicit["features"]}
            if set(ids) - acquired:
                raise IgnError("linked_building_unavailable", "An explicitly linked BD TOPO building is unavailable")
            extent_sources = [
                Building(str(index), geometry=feature["geometry"], crs="EPSG:2154")
                for index, feature in enumerate(explicit["features"])
            ]
            relation_rows = relations.get("rows", ()) if isinstance(relations, dict) else relations
            linked_groups = {
                str(row.get("batiment_groupe_id") or "")
                for row in relation_rows if linked_bdtopo_ids([row])
            }
            extent_sources.extend(
                building for group_id, building in trustworthy.items()
                if not building.fictitious_geometry and group_id not in linked_groups
            )
        else:
            extent_sources = list(trustworthy.values())
        bounds = bounded_bounds(extent_sources)
        result = self.buildings(bounds)
        if explicit is not None:
            merged = {}
            for feature in explicit["features"] + result["features"]:
                source_id = feature["properties"].get("cleabs")
                if not source_id:
                    raise IgnError("missing_source_id", "BD TOPO response lacks cleabs")
                if source_id in merged and merged[source_id] != feature:
                    raise IgnError("conflicting_source_geometry", f"BD TOPO feature changed during acquisition: {source_id}")
                merged[source_id] = feature
            if len(merged) > MAX_FEATURES:
                raise IgnError("feature_limit", "Combined linked and context features exceed the safe bound")
            result["features"] = list(merged.values())
            result["pages"] = explicit["pages"] + result["pages"]
        result["query_bounds"] = bounds
        return result

    def tiles(self, bounds):
        bounds = checked_bounds(bounds)
        result = self._features(
            TILE_LAYER, {"bbox": ",".join(map(str, bounds)) + ",EPSG:2154"},
            MAX_TILES,
        )
        tiles = []
        for feature in result["features"]:
            properties = feature["properties"]
            url = properties.get("url_npl")
            epoch = properties.get("code_mission") or (
                str(properties.get("date_debut_acquisition") or "") + "/"
                + str(properties.get("date_fin_acquisition") or "")
            )
            if epoch == "/":
                epoch = None
            tiles.append({
                "id": feature.get("id"), "url": copc_url(url), "original_url": url,
                "coordonnees_nw": properties.get("coordonnees_nw"),
                "horizontal_crs": _horizontal(properties.get("systeme_planimetrique")),
                "vertical_reference": _vertical(properties.get("systeme_altimetrique")),
                "acquisition_epoch": epoch,
                "date_debut_acquisition": properties.get("date_debut_acquisition"),
                "date_fin_acquisition": properties.get("date_fin_acquisition"),
                "date_edition": properties.get("date_edition"),
                "metadata": feature,
            })
        result["tiles"] = list(validate_acquisition(tiles, bounds))
        return result


def build_pdal_pipeline(tiles, bounds, output_path):
    """Write the original classified crop, declaring 1 mm output quantization."""
    if isinstance(tiles, dict):
        tiles = tiles["tiles"]
    bounds = checked_bounds(bounds)
    tiles = validate_acquisition(tiles, bounds)
    output = Path(output_path)
    if output.suffix.lower() not in {".las", ".laz"}:
        raise IgnError("invalid_output_path", "Point-cloud output must use .las or .laz")
    west, south, east, north = bounds
    crop = f"([{west},{east}],[{south},{north}])"
    pipeline: list[dict[str, Any]] = [
        {"type": "readers.copc", "filename": copc_url(tile["url"]), "bounds": crop,
         "tag": f"tile_{index}"}
        for index, tile in enumerate(tiles)
    ]
    pipeline += [
        {"type": "filters.merge", "inputs": [f"tile_{i}" for i in range(len(tiles))], "tag": "merged"},
        {"type": "filters.crop", "bounds": crop, "inputs": ["merged"], "tag": "original_crop"},
        {"type": "writers.las", "filename": str(output), "inputs": ["original_crop"],
         "forward": "all", "extra_dims": "all", "minor_version": 4, "dataformat_id": 6,
         "scale_x": .001, "scale_y": .001, "scale_z": .001,
         "offset_x": "auto", "offset_y": "auto", "offset_z": "auto"},
    ]
    return {"pipeline": pipeline}
