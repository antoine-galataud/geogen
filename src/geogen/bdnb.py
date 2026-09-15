"""Minimal client for the BDNB (Base de Données Nationale des Bâtiments) open API.

The BDNB API is a PostgREST service exposed behind a Gravitee gateway at
``https://api.bdnb.io/v1/bdnb``.  Table endpoints accept the usual PostgREST
query parameters (``select``, ``limit``, ``offset``, ``order`` and column
filters such as ``batiment_groupe_id=eq.<id>``) and return plain JSON arrays.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import replace
from typing import Any

import requests

from geogen.models import NOT_PROVIDED, Address
from geogen.models import AddressNotFoundError as ProviderAddressNotFoundError
from geogen.models import BuildingGroup, ProviderError

LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.bdnb.io/v1/bdnb"
API_KEY_HEADER = "X-Gravitee-Api-Key"
API_KEY_ENV_VAR = "BDNB_API_KEY"
DEFAULT_TIMEOUT = 30.0

#: Table joining every ``batiment_groupe`` data source of the BDNB.
BUILDING_GROUP_TABLE = "donnees/batiment_groupe_complet"

#: Columns without which no geometry can be built.
REQUIRED_COLUMNS = ("batiment_groupe_id", "geom_groupe")

#: Columns of ``batiment_groupe_complet`` needed to build a geometry model.
GEOMETRY_COLUMNS = (
    "batiment_groupe_id",
    "geom_groupe",
    "hauteur_mean",
    "nb_niveau",
    "altitude_sol_mean",
    "s_geom_groupe",
    "contient_fictive_geom_groupe",
    "libelle_adr_principale_ban",
    "libelle_commune_insee",
)

#: Columns used to approximate the envelope (fenestration and roof).
#:
#: ``materiaux_toiture_simplifie`` comes from ``batiment_groupe_synthese_enveloppe``, which
#: the open edition of the BDNB does not publish; it is skipped when rejected.
ENVELOPE_COLUMNS = (
    "pourcentage_surface_baie_vitree_exterieur",
    "l_orientation_baie_vitree",
    "mat_toit_txt",
    "materiaux_toiture_simplifie",
    "type_plancher_haut_deperditif",
    "type_isolation_mur_exterieur",
    "type_isolation_plancher_haut",
    "type_isolation_plancher_bas",
    "type_vitrage",
)

BUILDING_GROUP_COLUMNS = GEOMETRY_COLUMNS + ENVELOPE_COLUMNS

#: Table holding the DPE of the representative dwelling of a building group.
DPE_TABLE = "donnees/batiment_groupe_dpe_representatif_logement"

#: Glazed and wall areas of the representative dwelling, which give a glazing ratio.
#:
#: ``batiment_groupe_complet`` does not join them, they are read from the DPE table when it
#: does not publish a glazing percentage.
GLAZING_COLUMNS = (
    "surface_mur_exterieur",
    "surface_vitree_nord",
    "surface_vitree_est",
    "surface_vitree_sud",
    "surface_vitree_ouest",
)

#: DPE classifications exported as building metadata.
ENVELOPE_CLASSIFICATION_COLUMNS = (
    "type_isolation_mur_exterieur",
    "type_isolation_plancher_haut",
    "type_isolation_plancher_bas",
    "type_vitrage",
)

DPE_COLUMNS = GLAZING_COLUMNS + ENVELOPE_CLASSIFICATION_COLUMNS

#: Glazed area of a facade, per cardinal direction.
GLAZED_AREA_COLUMNS = (
    ("north", "surface_vitree_nord"),
    ("east", "surface_vitree_est"),
    ("south", "surface_vitree_sud"),
    ("west", "surface_vitree_ouest"),
)

#: Tables holding the roof material, read when the merged table does not publish it.
ROOF_TABLES = (
    ("donnees/batiment_groupe_ffo_bat", ("mat_toit_txt",)),
    ("donnees/batiment_groupe_synthese_enveloppe", ("materiaux_toiture_simplifie",)),
)

#: Columns describing the roof, in decreasing order of reliability.
ROOF_COLUMNS = ("materiaux_toiture_simplifie", "mat_toit_txt", "type_plancher_haut_deperditif")

#: Table describing the exterior walls, only served by the complete BDNB.
WALL_DICT_TABLE = "donnees/batiment_groupe_wall_dict"

#: Columns of ``batiment_groupe_wall_dict``, which describes the exterior walls.
WALL_DICT_COLUMNS = ("batiment_groupe_id", "wall_dict")

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

#: PostgreSQL code returned when a selected column is not part of a table.
UNKNOWN_COLUMN_CODE = "42703"

#: PostgREST and PostgreSQL codes returned when a table is not part of the schema.
UNKNOWN_TABLE_CODES = frozenset({"PGRST205", "42P01"})

_UNKNOWN_COLUMN_RE = re.compile(r"column\s+\"?(?:[\w.]+\.)?(\w+)\"?\s+does not exist", re.I)


class BdnbError(ProviderError):
    """Raised when the BDNB API cannot be reached or returns an error."""

    def __init__(self, message: str, *, status: int | None = None, payload: Any = None) -> None:
        super().__init__(message)
        #: HTTP status code, when the API answered.
        self.status = status
        #: Body of the error, as published by PostgREST.
        self.payload = payload

    @property
    def code(self) -> str | None:
        """Error code reported by PostgREST, if any."""
        if isinstance(self.payload, dict) and self.payload.get("code") is not None:
            return str(self.payload["code"])
        return None

    @property
    def detail(self) -> str:
        """Message reported by PostgREST, if any."""
        if isinstance(self.payload, dict) and self.payload.get("message") is not None:
            return str(self.payload["message"])
        return ""


class AddressNotFoundError(BdnbError, ProviderAddressNotFoundError):
    """Raised when an address does not match any entry of the BDNB."""


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    return None if number is None else int(number)


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "t", "vrai", "1", "yes"}
    return None


def _as_code(value: Any) -> str | None:
    """Read an identifier column, ignoring the empty ones."""
    if value is None:
        return None
    return str(value).strip() or None


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    """Read a PostgREST array, given either as a JSON array or as ``{a,b}``."""
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if item is not None)
    if isinstance(value, str):
        items = value.strip().strip("{}").split(",")
        return tuple(item.strip().strip('"') for item in items if item.strip())
    return ()


def _parse_wall_dict(value: Any) -> tuple[Any, ...]:
    """Read a ``wall_dict`` column, serialized either as JSON or as a list."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            LOGGER.debug("Ignoring an unreadable wall_dict")
            return ()
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return ()
    return tuple(wall for wall in value if isinstance(wall, dict))


def _roof_is_unknown(row: dict[str, Any]) -> bool:
    """Tell whether a row describes neither the material nor the shape of the roof."""
    return all(row.get(column) is None for column in ROOF_COLUMNS)


def _glazed_areas(row: dict[str, Any]) -> tuple[tuple[str, float], ...]:
    """Read the glazed area of each facade of a DPE row, in m²."""
    areas = []
    for cardinal, column in GLAZED_AREA_COLUMNS:
        area = _as_float(row.get(column))
        if area is not None and area > 0:
            areas.append((cardinal, area))
    return tuple(areas)


def unknown_column(error: BdnbError) -> str | None:
    """Return the column the BDNB refused to select, if that is what failed."""
    if error.code != UNKNOWN_COLUMN_CODE:
        return None
    match = _UNKNOWN_COLUMN_RE.search(error.detail)
    return match.group(1) if match else None


def unknown_table(error: BdnbError) -> bool:
    """Tell whether the BDNB failed because a table is not part of its schema."""
    return error.code in UNKNOWN_TABLE_CODES


def _error_payload(response: requests.Response | None) -> Any:
    """Read the JSON body PostgREST returns to describe an error."""
    if response is None:
        return None
    try:
        return response.json()
    except ValueError:
        return None


def building_group_from_row(
    row: dict[str, Any], *, building_class: type[BuildingGroup] = BuildingGroup
) -> BuildingGroup:
    """Parse a ``batiment_groupe_complet`` row without duplicating model fields."""
    return building_class(
        batiment_groupe_id=str(row.get("batiment_groupe_id") or ""),
        batiment_construction_id=_as_code(row.get("batiment_construction_id"))
        or _as_code(row.get("rnb_id")),
        geometry=row.get("geom_groupe"),
        height=_as_float(row.get("hauteur_mean")),
        storeys=_as_int(row.get("nb_niveau")),
        ground_elevation=_as_float(row.get("altitude_sol_mean")),
        footprint_area=_as_float(row.get("s_geom_groupe")),
        fictitious_geometry=_as_bool(row.get("contient_fictive_geom_groupe")),
        address=row.get("libelle_adr_principale_ban"),
        city=row.get("libelle_commune_insee"),
        glazed_ratio=_as_float(row.get("pourcentage_surface_baie_vitree_exterieur")),
        glazed_areas=_glazed_areas(row),
        wall_area=_as_float(row.get("surface_mur_exterieur")),
        glazing_orientations=_as_str_tuple(row.get("l_orientation_baie_vitree")),
        roof_material=row.get("materiaux_toiture_simplifie") or row.get("mat_toit_txt"),
        roof_type=row.get("type_plancher_haut_deperditif"),
        wall_insulation=row.get("type_isolation_mur_exterieur", NOT_PROVIDED),
        upper_floor_insulation=row.get("type_isolation_plancher_haut", NOT_PROVIDED),
        lower_floor_insulation=row.get("type_isolation_plancher_bas", NOT_PROVIDED),
        glazing_type=row.get("type_vitrage", NOT_PROVIDED),
    )


class BdnbClient:
    """Thin wrapper around the BDNB open API endpoints used by geogen."""

    def __init__(
        self,
        api_key: str | None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key:
            LOGGER.info(
                "Using the BDNB API open endpoints without an API key, which may be rate limited.",
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        #: Columns the BDNB refused to select, per table.
        self._unknown_columns: dict[str, set[str]] = defaultdict(set)
        #: Tables the BDNB does not publish.
        self._unknown_tables: set[str] = set()
        self._session = session or requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        if api_key:
            self._session.headers.update({API_KEY_HEADER: api_key})

    def __enter__(self) -> BdnbClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        LOGGER.debug("GET %s %s", url, params)
        try:
            response = self._session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as error:
            response = error.response
            status = response.status_code if response is not None else None
            payload = _error_payload(response)
            detail = payload.get("message") if isinstance(payload, dict) else None
            reason = f"{status} {detail}" if detail else str(status)
            raise BdnbError(
                f"BDNB API request to {url} failed ({reason})", status=status, payload=payload
            ) from error
        except requests.RequestException as error:
            raise BdnbError(f"BDNB API request to {url} failed: {error}") from error
        except ValueError as error:
            raise BdnbError(f"BDNB API returned an invalid response for {url}") from error

    def _rows(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Query a PostgREST table endpoint, which answers with a JSON array."""
        payload = self._get(path, params)
        if payload is None:
            return []
        if not isinstance(payload, list):
            raise BdnbError(f"BDNB API returned an unexpected response for {path}")
        return [row for row in payload if isinstance(row, dict)]

    def geocode(self, query: str, *, limit: int = 5) -> list[Address]:
        """Geocode a free form address with the BDNB geocoder."""
        payload = self._get("geocodage", {"q": query, "limit": limit})
        features = payload.get("features") if isinstance(payload, dict) else None
        addresses = []
        for feature in features or []:
            properties = feature.get("properties") if isinstance(feature, dict) else None
            if not isinstance(properties, dict):
                continue
            key = properties.get("id") or properties.get("cle_interop_adr")
            if key:
                addresses.append(
                    Address(
                        cle_interop_adr=str(key),
                        label=properties.get("label") or properties.get("libelle_adresse"),
                        code_commune_insee=properties.get("code_commune_insee"),
                    )
                )
        return addresses

    def search_addresses(self, query: str, *, limit: int = 5) -> list[Address]:
        """Search the ``adresse`` table with a case insensitive pattern.

        Used as a fallback when the (undocumented) geocoding endpoint is unavailable.
        """
        words = _WORD_RE.findall(query)
        if not words:
            return []
        pattern = "*{}*".format("*".join(words))
        rows = self._rows(
            "donnees/adresse",
            {
                "libelle_adresse": f"ilike.{pattern}",
                "select": "cle_interop_adr,libelle_adresse,code_commune_insee",
                "limit": limit,
            },
        )
        return [
            Address(
                cle_interop_adr=str(row["cle_interop_adr"]),
                label=row.get("libelle_adresse"),
                code_commune_insee=row.get("code_commune_insee"),
            )
            for row in rows
            if row.get("cle_interop_adr")
        ]

    def find_addresses(self, query: str, *, limit: int = 5) -> list[Address]:
        """Resolve an address, falling back on the ``adresse`` table lookup."""
        try:
            addresses = self.geocode(query, limit=limit)
        except BdnbError as error:
            LOGGER.debug("Geocoding of %r failed (%s), falling back", query, error)
            addresses = []
        if not addresses:
            addresses = self.search_addresses(query, limit=limit)
        return addresses

    def building_group_ids(self, cle_interop_adr: str, *, limit: int = 10) -> list[str]:
        """List the identifiers of the building groups serving an address."""
        rows = self._rows(
            "donnees/rel_batiment_groupe_adresse",
            {
                "cle_interop_adr": f"eq.{cle_interop_adr}",
                "select": "batiment_groupe_id",
                "limit": limit,
            },
        )
        ids = []
        for row in rows:
            group_id = row.get("batiment_groupe_id")
            if group_id and group_id not in ids:
                ids.append(str(group_id))
        return ids

    def building_group(self, batiment_groupe_id: str) -> BuildingGroup | None:
        """Download the data of a single building group.

        Envelope columns the BDNB does not publish are looked up in the table they originate
        from, and dropped when that table is not published either.  The envelope is then
        approximated from whatever the API did return.
        """
        row = self._table_row(BUILDING_GROUP_TABLE, batiment_groupe_id, BUILDING_GROUP_COLUMNS)
        if row is None:
            return None
        self._complete_envelope(row, batiment_groupe_id)
        group = BuildingGroup.from_row(row)
        walls = self._building_walls(batiment_groupe_id)
        return replace(group, walls=walls) if walls else group

    def _complete_envelope(self, row: dict[str, Any], batiment_groupe_id: str) -> None:
        """Look up the envelope attributes ``batiment_groupe_complet`` does not publish.

        The open edition joins fewer tables into ``batiment_groupe_complet`` than the
        complete one, so the fenestration and the roof are looked up in the table they
        originate from when the merged one knows nothing about them.
        """
        classifications_missing = any(
            column not in row for column in ENVELOPE_CLASSIFICATION_COLUMNS
        )
        if row.get("pourcentage_surface_baie_vitree_exterieur") is None or classifications_missing:
            self._merge_row(
                row,
                DPE_TABLE,
                DPE_COLUMNS,
                batiment_groupe_id,
                preserve_null_columns=ENVELOPE_CLASSIFICATION_COLUMNS,
            )
        for table, columns in ROOF_TABLES:
            if not _roof_is_unknown(row):
                break
            self._merge_row(row, table, columns, batiment_groupe_id)

    def _merge_row(
        self,
        row: dict[str, Any],
        table: str,
        columns: tuple[str, ...],
        batiment_groupe_id: str,
        *,
        preserve_null_columns: tuple[str, ...] = (),
    ) -> None:
        """Add to ``row`` the columns of ``table``, ignoring what the BDNB cannot serve."""
        try:
            extra = self._table_row(table, batiment_groupe_id, columns)
        except BdnbError as error:
            LOGGER.debug("Reading %s failed (%s), skipping it", table, error)
            return
        for column, value in (extra or {}).items():
            if column in preserve_null_columns and column not in row:
                row[column] = value
            elif value is not None and row.get(column) is None:
                row[column] = value

    def _table_row(
        self, table: str, batiment_groupe_id: str, columns: tuple[str, ...]
    ) -> dict[str, Any] | None:
        """Read the row of a building group, skipping what the BDNB does not publish.

        The BDNB is published in several editions, and the open one serves fewer tables and
        fewer columns than the complete data model.  Rather than giving up on the whole
        request, a column the API rejects is dropped and the request is retried; rejected
        columns and tables are remembered so they are only ever requested once.
        """
        if table in self._unknown_tables:
            return None
        wanted = [column for column in columns if column not in self._unknown_columns[table]]
        while wanted:
            try:
                rows = self._rows(
                    table,
                    {
                        "batiment_groupe_id": f"eq.{batiment_groupe_id}",
                        "select": ",".join(wanted),
                        "limit": 1,
                    },
                )
            except BdnbError as error:
                if unknown_table(error):
                    LOGGER.info("The BDNB does not publish the %s table, skipping it", table)
                    self._unknown_tables.add(table)
                    return None
                column = unknown_column(error)
                if column is None or column not in wanted or column in REQUIRED_COLUMNS:
                    raise
                LOGGER.info("The BDNB does not publish %s.%s, skipping it", table, column)
                self._unknown_columns[table].add(column)
                wanted.remove(column)
                continue
            return rows[0] if rows else None
        return None

    def building_walls(self, batiment_groupe_id: str) -> tuple[Any, ...]:
        """Download the exterior walls (``wall_dict``) of a building group.

        Each wall is a mapping describing one exterior surface of the building, with its
        ``wall_type`` (``floor``, ``roof`` or ``vertical``), ``inclination``, ``azimuth``,
        ``area`` and adjacency.
        """
        row = self._table_row(WALL_DICT_TABLE, batiment_groupe_id, WALL_DICT_COLUMNS)
        if row is None:
            return ()
        return _parse_wall_dict(row.get("wall_dict"))

    def _building_walls(self, batiment_groupe_id: str) -> tuple[Any, ...]:
        """Download the exterior walls, tolerating an unavailable table."""
        try:
            return self.building_walls(batiment_groupe_id)
        except BdnbError as error:
            LOGGER.debug("No wall description for %s (%s)", batiment_groupe_id, error)
            return ()

    def buildings_for_address(self, query: str, *, max_buildings: int = 10) -> list[BuildingGroup]:
        """Return the building groups located at ``query``.

        Raises:
            AddressNotFoundError: if the address is unknown to the BDNB.
        """
        addresses = self.find_addresses(query, limit=1)
        if not addresses:
            raise AddressNotFoundError(f"No BDNB address found for {query!r}")
        address = addresses[0]
        LOGGER.info("Address %r matched %s", query, address.label or address.cle_interop_adr)
        group_ids = self.building_group_ids(address.cle_interop_adr, limit=max_buildings)
        if not group_ids:
            raise AddressNotFoundError(f"No building found at address {query!r}")
        buildings = []
        for group_id in group_ids[:max_buildings]:
            building = self.building_group(group_id)
            if building is not None:
                buildings.append(building)
        return buildings
