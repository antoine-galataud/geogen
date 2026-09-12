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
from shapely.geometry import LineString, shape
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import Polygon, orient
from shapely.ops import polylabel
from shapely.ops import split as split_polygon
from shapely.ops import transform, unary_union

from geogen.envelope import DEFAULT_ROOF_PITCH, Envelope, envelope_from_group
from geogen.models import Building

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
    crs: str = LAMBERT_93
    holes: tuple[tuple[tuple[float, float], ...], ...] = ()

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
    group: Building,
    *,
    default_storey_height: float = DEFAULT_STOREY_HEIGHT,
    simplify_tolerance: float = DEFAULT_SIMPLIFY_TOLERANCE,
    default_roof_pitch: float = DEFAULT_ROOF_PITCH,
    default_window_to_wall_ratio: float | None = None,
    crs: str | None = None,
) -> list[Footprint]:
    """Convert a BDNB building group into extrudable footprints."""
    if group.geometry is None:
        LOGGER.warning("Building %s has no geometry", group.code)
        return []
    source_crs = getattr(group, "crs", None)
    crs = crs or (source_crs if source_crs != WGS84 else None) or LAMBERT_93
    geometry = parse_geometry(group.geometry)
    if source_crs is None:
        geometry = to_projected(geometry, crs)  # historical BDNB public objects
    elif source_crs != crs:
        geometry = transform(_transformer(source_crs, crs).transform, geometry)
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
            # Simplify the whole polygon once, preserving the relationship of its rings.
            simplified = polygon.simplify(simplify_tolerance, preserve_topology=True)
            if isinstance(simplified, Polygon) and simplified.is_valid:
                polygon = simplified
            polygon = orient(polygon, sign=-1.0)
            ring = exterior_ring(polygon, 0)
            holes = tuple(
                tuple((float(x), float(y)) for x, y in hole.coords[:-1])
                for hole in polygon.interiors
            )
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
                crs=crs,
                holes=holes,
            )
        )
    return footprints


def roof_apex(
    ring: tuple[tuple[float, float], ...], pitch: float, max_height: float
) -> tuple[float, float, float] | None:
    """Return the apex of a hip roof of ``pitch`` degrees covering ``ring``.

    Legacy estimator retained for Python callers; exporters now use roof_faces.
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


def footprint_polygon(footprint: Footprint) -> Polygon:
    """Complete footprint, including every courtyard, in metric coordinates."""
    return Polygon(footprint.ring, footprint.holes)


def _check_partition(polygon: Polygon, patches: list[Polygon]) -> None:
    tolerance = max(1e-6, polygon.area * 1e-8)
    if not patches or any(not p.is_valid or p.interiors or p.area <= 0 for p in patches):
        raise GeometryError("Partition contains an invalid or holed surface")
    union = unary_union(patches)
    if (
        union.symmetric_difference(polygon).area > tolerance
        or sum(p.area for p in patches) - union.area > tolerance
    ):
        raise GeometryError("Partition does not preserve footprint area")


def _open_courtyards(polygon: Polygon) -> list[Polygon]:
    """Open holes using short cuts, rather than cuts at every facade vertex.

    Candidate lines pass through a hole's interior point. Choose the line with the least new
    seam length, accounting for fragment count. Recurse only on pieces that still have
    holes. There is no snapping, buffering or deletion of small courtyards or narrow
    building features.
    """
    pending, result = [polygon], []
    while pending:
        part = pending.pop()
        if not part.interiors:
            result.append(part)
            continue
        hole = max((Polygon(r) for r in part.interiors), key=lambda p: p.area)
        point = hole.representative_point()
        bounds = part.bounds
        length = math.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 2
        # Also try the dominant building axes: the result must not depend on
        # a building happening to align with the map's X/Y axes.
        corners = list(part.minimum_rotated_rectangle.exterior.coords)
        angles = {math.pi * i / 12 for i in range(12)}
        for start, end in zip(corners, corners[1:]):
            angles.add(math.atan2(end[1] - start[1], end[0] - start[0]) % math.pi)
        best = None
        for angle in sorted(angles):
            dx, dy = math.cos(angle) * length, math.sin(angle) * length
            line = LineString([(point.x - dx, point.y - dy), (point.x + dx, point.y + dy)])
            pieces = [
                p for p in split_polygon(part, line).geoms if isinstance(p, Polygon) and p.area > 0
            ]
            if len(pieces) < 2 or sum(len(p.interiors) for p in pieces) >= len(part.interiors):
                continue
            seams = max(0.0, (sum(p.length for p in pieces) - part.length) / 2)
            score = seams * (1 + 0.25 * (len(pieces) - 2))
            if best is None or score < best[0]:
                best = score, pieces
        if best is None:
            raise GeometryError("Could not open courtyard rings without changing geometry")
        pending.extend(best[1])
    return result


def _convex_patches(polygon: Polygon) -> list[Polygon]:
    """Triangulate a simple polygon and merge adjacent triangles into convex wings.

    Used only if the roof engine cannot process a complex simple polygon. Long shared edges
    are removed first to avoid retaining skinny triangles.
    """
    import openstudio

    ox, oy = polygon.centroid.coords[0]
    ring = exterior_ring(polygon.simplify(0), 0)
    triangles = openstudio.computeTriangulation(
        openstudio.Point3dVector([openstudio.Point3d(x - ox, y - oy, 0) for x, y in ring]), []
    )
    patches = [Polygon([(v.x() + ox, v.y() + oy) for v in triangle]) for triangle in triangles]
    _check_partition(polygon, patches)
    # All mutations remain unions of the exact triangles, so no area can vanish.
    while True:
        best = None
        for i, first in enumerate(patches):
            for j in range(i + 1, len(patches)):
                second = patches[j]
                shared = first.boundary.intersection(second.boundary).length
                if shared <= 1e-8:
                    continue
                merged = first.union(second)
                if (
                    not isinstance(merged, Polygon)
                    or merged.interiors
                    or merged.convex_hull.area - merged.area > max(1e-8, merged.area * 1e-10)
                ):
                    continue
                if best is None or shared > best[0]:
                    best = shared, i, j, merged
        if best is None:
            break
        _, i, j, merged = best
        patches[i] = merged
        patches.pop(j)
    _check_partition(polygon, patches)
    return patches


@lru_cache(maxsize=128)
def floor_patches(footprint: Footprint, *, split: bool = False) -> list[Polygon]:
    """Return a small set of simple surfaces, retaining exact courtyard topology.

    Floors/ceilings only need holes opened; concave surfaces are allowed. Roof fallback can
    request convex pieces with split=True. This separation avoids propagating unnecessary
    roof subdivisions into floors and ceilings.
    """
    polygon = footprint_polygon(footprint)
    if not polygon.is_valid or polygon.is_empty:
        raise GeometryError(f"Invalid footprint {footprint.name!r}")
    patches = _open_courtyards(polygon)
    if split:
        patches = [piece for patch in patches for piece in _convex_patches(patch)]
    _check_partition(polygon, patches)
    return patches


@lru_cache(maxsize=128)
def roof_sections(footprint: Footprint, max_height: float) -> tuple:
    """Validated hip-roof faces, with Z relative to the eaves.

    OpenStudio's hip-roof generator builds ridges/valleys for simple polygons. Courtyard
    footprints use the hole-free wings from floor_patches, retaining open courtyards but
    introducing inferred eave valleys at wing seams. Any invalid result falls back to a
    complete flat roof, never a spanning fan.
    """
    import openstudio

    pitch = footprint.envelope.roof_pitch
    if not pitch or max_height <= 0:
        return ()
    ox, oy = bounding_center([footprint])

    def generate(patches):
        result = []
        for patch in patches:
            points = openstudio.Point3dVector(
                [
                    openstudio.Point3d(x - ox, y - oy, 0)
                    for x, y in exterior_ring(patch.simplify(0), 0)
                ]
            )
            generated = openstudio.generateHipRoof(points, pitch)
            faces = [
                tuple((v.x() + ox, v.y() + oy, v.z()) for v in face)
                for face in generated
                if len(face) >= 3
            ]
            projected = [Polygon([(x, y) for x, y, z in face]) for face in faces]
            tolerance = max(1e-6, patch.area * 1e-8)
            if not faces or any(not q.is_valid or q.area <= 0 for q in projected):
                raise GeometryError("Invalid hip-roof faces")
            union = unary_union(projected)
            if (
                union.symmetric_difference(patch).area > tolerance
                or sum(q.area for q in projected) - union.area > tolerance
            ):
                raise GeometryError("Hip-roof faces do not cover the footprint exactly")
            result.append((patch, tuple(faces)))
        return result

    try:
        try:
            result = generate(floor_patches(footprint))
        except (RuntimeError, ValueError):
            LOGGER.warning("Using partitioned hip roofs for complex footprint %s", footprint.name)
            result = generate(floor_patches(footprint, split=True))
        height = max(z for patch, faces in result for face in faces for x, y, z in face)
        if not math.isfinite(height) or min(height, max_height) < MIN_ROOF_HEIGHT:
            return ()
        # Scale all heights together, preserving planar faces and shared ridges.
        scale = min(1.0, max_height / height)
        return tuple(
            (patch, tuple(tuple((x, y, z * scale) for x, y, z in face) for face in faces))
            for patch, faces in result
        )
    except (RuntimeError, ValueError) as error:
        LOGGER.warning("Cannot construct hip roof for %s: %s; keeping flat", footprint.name, error)
        return ()


def roof_faces(footprint: Footprint, max_height: float) -> tuple:
    """Shared roof faces for preview/export, flattened from closed roof sections."""
    return tuple(face for patch, faces in roof_sections(footprint, max_height) for face in faces)
