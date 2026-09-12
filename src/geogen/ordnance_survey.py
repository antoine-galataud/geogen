"""OS Places address resolution and OS NGD Building Features geometry.

Only Great Britain is covered by the building dataset. All spatial requests and responses
explicitly use British National Grid. No nearest-building guess is made.
"""

from __future__ import annotations

import math
from typing import Any
from urllib.parse import urljoin, urlsplit

import requests
from shapely.geometry import Point, shape

from geogen.models import AddressNotFoundError, Building, ProviderError

PLACES_URL = "https://api.os.uk/search/places/v1"
NGD_URL = "https://api.os.uk/features/ngd/ofa/v1"
BNG = "EPSG:27700"
BNG_URI = "http://www.opengis.net/def/crs/EPSG/0/27700"


class OsError(ProviderError):
    """OS authentication, transport, schema or matching error."""


def number(value: Any, *, positive: bool = False) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and (not positive or result > 0) else None


def building_from_feature(feature: dict, address: str | None = None) -> Building:
    """Normalize the NGD Building v4 schema to metric data."""
    props = feature.get("properties") or {}
    code = props.get("osid") or feature.get("id")
    if not code or not feature.get("geometry"):
        raise OsError("OS building feature is missing its identifier or geometry")
    height = number(props.get("height_relativeroofbase_m"), positive=True)
    roof_shape = str(props.get("roofshapeaspect_shape") or "").lower()
    if height is None:
        height = number(props.get("height_relativemax_m"), positive=True)
        # A maximum height already includes the roof: do not add another attic.
        if height is not None:
            roof_shape = "flat"
    floors = number(props.get("numberoffloors"), positive=True)
    elevation = number(props.get("height_absolutemin_m"))
    return Building(
        code=f"os-{code}",
        country="UK",
        crs=BNG,
        geometry=feature["geometry"],
        height=height,
        storeys=int(floors) if floors and floors.is_integer() else None,
        ground_elevation=elevation,
        footprint_area=number(props.get("geometry_area_m2"), positive=True),
        address=address,
        roof_shape=roof_shape if roof_shape in {"flat", "pitched"} else None,
        roof_material=props.get("roofmaterial_primarymaterial"),
    )


class OsClient:
    """Reusable, cached UK provider; key needs Places and NGD Features access."""

    def __init__(
        self,
        api_key: str | None,
        *,
        timeout: float = 30,
        session: requests.Session | None = None,
        places_url: str = PLACES_URL,
        ngd_url: str = NGD_URL,
        collection: str | None = None,
    ) -> None:
        if not api_key:
            raise OsError(
                "UK addresses require OS_API_KEY (or --os-api-key), with OS Places and NGD Features access"
            )
        self.timeout = timeout
        self.places_url = places_url.rstrip("/")
        self.ngd_url = ngd_url.rstrip("/")
        self.collection = collection or "bld-fts-building-4"
        if self.collection != "bld-fts-building-4":
            raise OsError(
                "Only the OS NGD Building v4 collection (bld-fts-building-4) is supported"
            )
        self._session = session or requests.Session()
        # Header authentication avoids exposing the API key in request URLs/logs.
        self._session.headers.update({"key": api_key, "Accept": "application/json"})
        self._addresses: dict[str, dict] = {}
        self._buildings: dict[str, list[Building]] = {}

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> OsClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _get(self, url: str, params: dict | None = None) -> dict:
        try:
            response = self._session.get(
                url, params=params, timeout=self.timeout, allow_redirects=False
            )
            if 300 <= response.status_code < 400:
                raise OsError("OS API returned an unexpected redirect")
            response.raise_for_status()
            payload = response.json()
        except requests.HTTPError as error:
            status = error.response.status_code if error.response is not None else "unknown"
            raise OsError(
                f"OS API request failed (HTTP {status}); check access, quota and configuration"
            ) from None
        except ValueError:
            raise OsError("OS API returned invalid JSON") from None
        except requests.RequestException:
            raise OsError("OS API request failed (connection or timeout)") from None
        if not isinstance(payload, dict):
            raise OsError("OS API returned an unexpected response")
        return payload

    def _pages(self, url: str, params: dict, key: str):
        seen = set()
        while url:
            if url in seen:
                raise OsError("OS API returned a repeated pagination link")
            seen.add(url)
            payload = self._get(url, params)
            rows = payload.get(key)
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise OsError(f"OS API response is missing a valid {key} list")
            yield rows
            link = next(
                (
                    link.get("href")
                    for link in payload.get("links", [])
                    if link.get("rel") == "next"
                ),
                None,
            )
            if not link:
                break
            next_url = urljoin(url, link)
            # Never forward authentication to another origin or endpoint.
            if (urlsplit(next_url).scheme, urlsplit(next_url).netloc, urlsplit(next_url).path) != (
                urlsplit(url).scheme,
                urlsplit(url).netloc,
                urlsplit(url).path,
            ):
                raise OsError("OS API returned an unexpected pagination destination")
            url, params = next_url, None

    def _address(self, query: str) -> dict:
        cache_key = " ".join(query.casefold().split())
        if cache_key in self._addresses:
            return self._addresses[cache_key]
        payload = self._get(
            f"{self.places_url}/find",
            {
                "query": query,
                "dataset": "DPA",
                "maxresults": 2,
                "output_srs": BNG,
                "minmatch": 0.8,
            },
        )
        print(payload)
        if not isinstance(payload.get("results", []), list):
            raise OsError("OS Places returned invalid results")
        candidates = [
            row["DPA"]
            for row in payload.get("results", [])
            if isinstance(row, dict) and isinstance(row.get("DPA"), dict)
        ]
        if not candidates:
            raise AddressNotFoundError(f"No UK address found for {query!r}")
        candidates.sort(key=lambda row: number(row.get("MATCH")) or 0, reverse=True)
        best = candidates[0]
        score = number(best.get("MATCH")) or 0
        if score < 0.8:
            raise AddressNotFoundError(
                f"No confident UK address match for {query!r}; include the postcode"
            )
        if (
            len(candidates) > 1
            and candidates[1].get("UPRN") != best.get("UPRN")
            and (score - (number(candidates[1].get("MATCH")) or 0) < 0.05)
        ):
            raise OsError(
                f"Ambiguous UK address {query!r}; include a building number/name and full postcode"
            )
        if best.get("COUNTRY_CODE") not in {"E", "W", "S"}:
            raise OsError(
                "OS Building Features covers Great Britain only (England, Wales and Scotland)"
            )
        if not best.get("UPRN"):
            raise OsError("OS Places address is missing its UPRN")
        self._addresses[cache_key] = best
        return best

    def buildings_for_address(self, query: str, *, max_buildings: int = 10) -> list[Building]:
        if max_buildings < 1:
            raise ValueError("max_buildings must be positive")
        address = self._address(query)
        uprn = str(address["UPRN"])
        if uprn not in self._buildings:
            x, y = number(address.get("X_COORDINATE")), number(address.get("Y_COORDINATE"))
            if x is None or y is None:
                raise OsError("OS Places address has no usable British National Grid coordinates")
            point = Point(x, y)
            collection = self.collection
            params = {
                "bbox": f"{x-1},{y-1},{x+1},{y+1}",
                "bbox-crs": BNG_URI,
                "crs": BNG_URI,
                "limit": 100,
            }
            found: dict[str, Building] = {}
            for features in self._pages(
                f"{self.ngd_url}/collections/{collection}/items", params, "features"
            ):
                for feature in features:
                    try:
                        geometry = shape(feature.get("geometry"))
                    except (TypeError, ValueError, AttributeError, KeyError):
                        raise OsError("OS returned an invalid building geometry") from None
                    # bbox is only a candidate filter; never model unrelated neighbours.
                    if (
                        geometry.geom_type not in {"Polygon", "MultiPolygon"}
                        or not geometry.is_valid
                    ):
                        raise OsError("OS returned a non-polygon or invalid building geometry")
                    if geometry.covers(point):
                        building = building_from_feature(feature, address.get("ADDRESS"))
                        found.setdefault(building.code, building)
            if not found:
                raise AddressNotFoundError(
                    f"No OS building footprint contains the address point for {query!r}; "
                    "the address may be outside its building footprint"
                )
            self._buildings[uprn] = list(found.values())
        return self._buildings[uprn][:max_buildings]
