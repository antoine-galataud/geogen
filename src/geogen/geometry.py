"""Geometry helpers turning BDNB geometries into OpenStudio ready footprints.

BDNB geometries are expressed in Lambert-93 (EPSG:2154), a projected system whose unit is
the metre, which is also the unit used by OpenStudio models. Footprints therefore only need
to be translated to a local origin, no reprojection is required unless the API returns
geographic coordinates.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from pyproj import Transformer
from shapely import force_2d, wkb, wkt
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import Polygon, orient
from shapely.ops import polylabel, transform

from geogen.bdnb import BuildingGroup
from geogen.envelope import DEFAULT_ROOF_PITCH, Envelope, envelope_from_group

LOGGER = logging.getLogger(__name__)

#: Coordinate reference system used by the BDNB geometries.
LAMBERT_93 = "EPSG:2154"
WGS84 = "EPSG:4326"

DEFAULT_STOREY_HEIGHT = 3.0
DEFAULT_SIMPLIFY_TOLERANCE = 0.1
#: Footprints smaller than this area (m²) are dropped, they cannot be extruded.
MIN_FOOTPRINT_AREA = 1.0
#: Roofs lower than this height (m) are kept flat.
MIN_ROOF_HEIGHT = 0.5
_MIN_VERTEX_DISTANCE = 1e-3
_POLYLABEL_TOLERANCE = 0.5


class GeometryError(ValueError):
    """Raised when a BDNB geometry cannot be interpreted."""


@dataclass(frozen=True)
class Footprint:
    """A building footprint ready to be extruded into storeys."""

    name: str
    #: Exterior ring, in clockwise order, without the closing point.
    ring: tuple[tuple[float, float], ...]
    height: float
    storeys: int
    #: Ground altitude of the building, ``None`` when the BDNB does not know it.
    ground_elevation: float | None = None
    #: Fenestration and roof approximated from the BDNB.
    envelope: Envelope = field(default_factory=Envelope)

    @property
    def storey_height(self) -> float:
        """Floor to floor height of a single storey."""
        return self.height / self.storeys


@lru_cache(maxsize=None)
def _transformer(source_crs: str, target_crs: str) -> Transformer:
    return Transformer.from_crs(source_crs, target_crs, always_xy=True)


def parse_geometry(raw: Any) -> BaseGeometry:
    """Parse a BDNB geometry given as GeoJSON, WKT or (E)WKB hexadecimal.

    Any Z ordinate is dropped, footprints are two dimensional.
    """
    return force_2d(_parse_geometry(raw))


def _parse_geometry(raw: Any) -> BaseGeometry:
    if isinstance(raw, BaseGeometry):
        return raw
    if isinstance(raw, dict):
        try:
            return shape(raw)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise GeometryError(f"Unsupported GeoJSON geometry: {raw!r}") from error
    if isinstance(raw, str):
        value = raw.strip()
        if value.startswith("{"):
            try:
                return shape(json.loads(value))
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                raise GeometryError(f"Unsupported GeoJSON geometry: {raw!r}") from error
        try:
            if value and len(value) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in value):
                return wkb.loads(value, hex=True)
            return wkt.loads(value)
        except Exception as error:  # shapely raises various parse errors
            raise GeometryError(f"Unsupported geometry: {raw!r}") from error
    raise GeometryError(f"Unsupported geometry: {raw!r}")


def is_geographic(geometry: BaseGeometry) -> bool:
    """Tell whether a geometry looks like longitude/latitude degrees."""
    min_x, min_y, max_x, max_y = geometry.bounds
    return abs(min_x) <= 180 and abs(max_x) <= 180 and abs(min_y) <= 90 and abs(max_y) <= 90


def to_projected(geometry: BaseGeometry, crs: str = LAMBERT_93) -> BaseGeometry:
    """Return ``geometry`` in a metric CRS, reprojecting degrees if needed."""
    if not is_geographic(geometry):
        return geometry
    LOGGER.debug("Geometry looks geographic, reprojecting to %s", crs)
    return transform(_transformer(WGS84, crs).transform, geometry)


def to_wgs84(x: float, y: float, crs: str = LAMBERT_93) -> tuple[float, float]:
    """Convert projected coordinates to (longitude, latitude) degrees."""
    return _transformer(crs, WGS84).transform(x, y)


def polygons(geometry: BaseGeometry) -> list[Polygon]:
    """Explode a (multi)polygon into its non-negligible polygon parts."""
    if geometry.is_empty:
        return []
    if not geometry.is_valid:
        LOGGER.debug("Repairing an invalid %s geometry", geometry.geom_type)
        geometry = geometry.buffer(0)
        if geometry.is_empty:
            return []
    if isinstance(geometry, Polygon):
        parts = [geometry]
    elif hasattr(geometry, "geoms"):
        parts = [part for geom in geometry.geoms for part in polygons(geom)]
    else:
        raise GeometryError(f"Geometry {geometry.geom_type} is not a surface")
    return [part for part in parts if part.area >= MIN_FOOTPRINT_AREA]


def exterior_ring(
    polygon: Polygon, simplify_tolerance: float = DEFAULT_SIMPLIFY_TOLERANCE
) -> tuple[tuple[float, float], ...]:
    """Return the clockwise exterior ring of ``polygon``, without duplicates.

    OpenStudio requires the outward normal of a floor print to point downwards, which means
    the vertices must be given in clockwise order.
    """
    if simplify_tolerance > 0:
        simplified = polygon.simplify(simplify_tolerance, preserve_topology=True)
        if isinstance(simplified, Polygon) and simplified.area >= MIN_FOOTPRINT_AREA:
            polygon = simplified
    clockwise = orient(polygon, sign=-1.0)
    ring: list[tuple[float, float]] = []
    for x, y in clockwise.exterior.coords:
        point = (float(x), float(y))
        if ring and _close(ring[-1], point):
            continue
        ring.append(point)
    if len(ring) > 1 and _close(ring[0], ring[-1]):
        ring.pop()
    if len(ring) < 3:
        raise GeometryError("A footprint needs at least three distinct vertices")
    return tuple(ring)


def _close(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return (
        abs(first[0] - second[0]) < _MIN_VERTEX_DISTANCE
        and abs(first[1] - second[1]) < _MIN_VERTEX_DISTANCE
    )


def storeys_and_height(
    height: float | None,
    storeys: int | None,
    default_storey_height: float = DEFAULT_STOREY_HEIGHT,
) -> tuple[int, float]:
    """Complete the missing height or storey count of a building."""
    if storeys is not None and storeys <= 0:
        storeys = None
    if height is not None and height <= 0:
        height = None
    if height is None and storeys is None:
        return 1, default_storey_height
    if height is None:
        return storeys, storeys * default_storey_height  # type: ignore[operator]
    if storeys is None:
        return max(1, round(height / default_storey_height)), height
    return storeys, height


def footprints_from_group(
    group: BuildingGroup,
    *,
    default_storey_height: float = DEFAULT_STOREY_HEIGHT,
    simplify_tolerance: float = DEFAULT_SIMPLIFY_TOLERANCE,
    default_roof_pitch: float = DEFAULT_ROOF_PITCH,
    default_window_to_wall_ratio: float | None = None,
    crs: str = LAMBERT_93,
) -> list[Footprint]:
    """Convert a BDNB building group into extrudable footprints."""
    if group.geometry is None:
        LOGGER.warning("Building %s has no geometry", group.code)
        return []
    geometry = to_projected(parse_geometry(group.geometry), crs)
    parts = polygons(geometry)
    if not parts:
        LOGGER.warning("Building %s has an empty geometry", group.code)
        return []
    storeys, height = storeys_and_height(group.height, group.storeys, default_storey_height)
    envelope = envelope_from_group(
        group,
        default_roof_pitch=default_roof_pitch,
        default_window_to_wall_ratio=default_window_to_wall_ratio,
    )
    footprints = []
    for index, polygon in enumerate(parts, start=1):
        name = group.code
        if len(parts) > 1:
            name = f"{name} Part {index}"
        try:
            ring = exterior_ring(polygon, simplify_tolerance)
        except GeometryError as error:
            LOGGER.warning("Skipping footprint %s: %s", name, error)
            continue
        footprints.append(
            Footprint(
                name=name,
                ring=ring,
                height=height,
                storeys=storeys,
                ground_elevation=group.ground_elevation,
                envelope=envelope,
            )
        )
    return footprints


def roof_apex(
    ring: tuple[tuple[float, float], ...], pitch: float, max_height: float
) -> tuple[float, float, float] | None:
    """Return the apex of a hip roof of ``pitch`` degrees covering ``ring``.

    The apex is the point of the footprint that is the farthest away from its boundary,
    raised so that the widest roof faces have the requested pitch. ``None`` is returned when
    the resulting roof would be negligible.
    """
    if pitch <= 0:
        return None
    polygon = Polygon(ring)
    if not polygon.is_valid:
        return None
    try:
        centre = polylabel(polygon, tolerance=_POLYLABEL_TOLERANCE)
    except Exception as error:  # shapely raises various topological errors
        LOGGER.debug("Cannot locate the apex of a roof: %s", error)
        return None
    radius = centre.distance(polygon.exterior)
    height = min(math.tan(math.radians(pitch)) * radius, max_height)
    if height < MIN_ROOF_HEIGHT:
        return None
    return float(centre.x), float(centre.y), float(height)


def bounding_center(footprints: Iterable[Footprint]) -> tuple[float, float]:
    """Return the centre of the bounding box of a set of footprints."""
    xs = [x for footprint in footprints for x, _ in footprint.ring]
    ys = [y for footprint in footprints for _, y in footprint.ring]
    if not xs:
        raise GeometryError("Cannot compute an origin without any footprint")
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
