"""Vector SVG rendering of geogen building footprints.

The SVG is a lightweight orthographic 3D preview intended for dashboards and
report generation.  It is derived from the same :class:`~geogen.geometry.Footprint`
objects used to build the OpenStudio model, so it does not require a browser,
OpenGL or a second rendering engine.

The renderer deliberately keeps the output simple: opaque wall/roof polygons,
storey separators and approximate windows.  CSS classes are written into the
SVG so a downstream report generator can restyle the drawing if required.
"""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from heapq import heappop, heappush
from pathlib import Path

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from geogen.envelope import DEFAULT_MAX_ROOF_HEIGHT, cardinal_point
from geogen.geometry import (
    Footprint,
    bounding_center,
    exterior_ring,
    floor_patches,
    roof_faces,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_SVG_WIDTH = 1200
DEFAULT_SVG_HEIGHT = 900
DEFAULT_SVG_MARGIN = 40.0
DEFAULT_CAMERA_AZIMUTH = 45.0
DEFAULT_CAMERA_ELEVATION = 28.0

_SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", _SVG_NS)


class SvgError(RuntimeError):
    """Raised when an SVG preview cannot be generated or saved."""


Point3D = tuple[float, float, float]
Point2D = tuple[float, float]


@dataclass(frozen=True)
class _Projection:
    origin: tuple[float, float]
    azimuth: float
    elevation: float

    def camera(self, point: Point3D) -> tuple[float, float, float]:
        """Return orthographic camera coordinates ``(x, y_up, depth)``."""
        x, y, z = point
        x -= self.origin[0]
        y -= self.origin[1]

        azimuth = math.radians(self.azimuth)
        elevation = math.radians(self.elevation)

        # First rotate the plan around Z, then tilt the view around X.
        xr = math.cos(azimuth) * x - math.sin(azimuth) * y
        yr = math.sin(azimuth) * x + math.cos(azimuth) * y

        screen_y = math.cos(elevation) * z - math.sin(elevation) * yr
        depth = math.cos(elevation) * yr + math.sin(elevation) * z
        return xr, screen_y, depth

    def project(self, point: Point3D) -> Point2D:
        x, y, _ = self.camera(point)
        return x, y

    def depth(self, points: Iterable[Point3D]) -> float:
        depths = [self.camera(point)[2] for point in points]
        return sum(depths) / len(depths) if depths else 0.0


@dataclass
class _Drawable:
    """A painter-sorted group of SVG primitives."""

    depth: float
    primitives: list[tuple[str, tuple[Point3D, ...], str]] = field(default_factory=list)

    def add_polygon(self, points: Iterable[Point3D], css_class: str) -> None:
        self.primitives.append(("polygon", tuple(points), css_class))

    def add_polyline(self, points: Iterable[Point3D], css_class: str) -> None:
        self.primitives.append(("polyline", tuple(points), css_class))


def _reference_elevation(footprints: Iterable[Footprint]) -> float:
    elevations = [
        footprint.ground_elevation
        for footprint in footprints
        if footprint.ground_elevation is not None
    ]
    return min(elevations) if elevations else 0.0


def _base_elevation(footprint: Footprint, reference_elevation: float) -> float:
    if footprint.ground_elevation is None:
        return 0.0
    return footprint.ground_elevation - reference_elevation


def _wall_azimuth(start: tuple[float, float], end: tuple[float, float]) -> float:
    """Return wall outward azimuth in degrees clockwise from north.

    Footprint rings are clockwise in :mod:`geogen.geometry`, therefore the
    exterior normal is the left-hand normal of each directed edge.
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    normal_x = -dy
    normal_y = dx
    return math.degrees(math.atan2(normal_x, normal_y)) % 360.0


def _interpolate_edge(
    start: tuple[float, float], end: tuple[float, float], fraction: float, z: float
) -> Point3D:
    return (
        start[0] + (end[0] - start[0]) * fraction,
        start[1] + (end[1] - start[1]) * fraction,
        z,
    )


def _window_polygon(
    start: tuple[float, float],
    end: tuple[float, float],
    z0: float,
    storey_height: float,
    ratio: float,
) -> tuple[Point3D, Point3D, Point3D, Point3D]:
    """Approximate one centred window whose area follows the requested WWR."""
    # Keep a report-friendly rectangular aspect while preserving approximately
    # the same wall-area ratio.  This is a visual approximation only; the OSM
    # remains the authoritative geometry for simulation.
    height_fraction = min(0.65, max(0.30, math.sqrt(ratio)))
    width_fraction = min(0.90, max(0.08, ratio / height_fraction))

    t0 = (1.0 - width_fraction) / 2.0
    t1 = 1.0 - t0
    window_height = storey_height * height_fraction
    window_z0 = z0 + (storey_height - window_height) / 2.0
    window_z1 = window_z0 + window_height

    return (
        _interpolate_edge(start, end, t0, window_z0),
        _interpolate_edge(start, end, t1, window_z0),
        _interpolate_edge(start, end, t1, window_z1),
        _interpolate_edge(start, end, t0, window_z1),
    )


def _wall_drawable(
    footprint: Footprint,
    edge_index: int,
    base_z: float,
    projection: _Projection,
) -> _Drawable:
    ring = footprint.ring
    start = ring[edge_index]
    end = ring[(edge_index + 1) % len(ring)]
    top_z = base_z + footprint.height

    wall = (
        (start[0], start[1], base_z),
        (end[0], end[1], base_z),
        (end[0], end[1], top_z),
        (start[0], start[1], top_z),
    )
    drawable = _Drawable(depth=projection.depth(wall))
    drawable.add_polygon(wall, "wall")

    # Storey separators make the report preview easier to read.
    for level in range(1, footprint.storeys):
        z = base_z + level * footprint.storey_height
        drawable.add_polyline(((start[0], start[1], z), (end[0], end[1], z)), "storey-line")

    ratio = footprint.envelope.window_to_wall_ratio
    if ratio:
        orientations = footprint.envelope.glazing_orientations
        orientation = cardinal_point(_wall_azimuth(start, end))
        if not orientations or orientation in orientations:
            for level in range(footprint.storeys):
                z = base_z + level * footprint.storey_height
                drawable.add_polygon(
                    _window_polygon(start, end, z, footprint.storey_height, ratio), "window"
                )

    return drawable


def _roof_drawables(
    footprint: Footprint,
    base_z: float,
    projection: _Projection,
    max_roof_height: float,
) -> list[_Drawable]:
    top_z = base_z + footprint.height
    faces = roof_faces(footprint, max_roof_height)
    pitched = bool(faces)
    if not faces:
        faces = [
            tuple((x, y, 0) for x, y in exterior_ring(patch, 0))
            for patch in floor_patches(footprint)
        ]
    drawables = []
    for face in faces:
        points = tuple((x, y, top_z + z) for x, y, z in face)
        drawable = _Drawable(depth=projection.depth(points))
        drawable.add_polygon(points, "roof pitched-roof" if pitched else "roof flat-roof")
        drawables.append(drawable)
    return drawables


def _build_scene(
    footprints: list[Footprint],
    projection: _Projection,
    max_roof_height: float,
) -> list[_Drawable]:
    reference_elevation = _reference_elevation(footprints)
    drawables: list[_Drawable] = []

    for footprint in footprints:
        base_z = _base_elevation(footprint, reference_elevation)
        for ring in (footprint.ring, *footprint.holes):
            wall_footprint = replace(footprint, ring=ring)
            for edge_index in range(len(ring)):
                drawables.append(_wall_drawable(wall_footprint, edge_index, base_z, projection))
        drawables.extend(_roof_drawables(footprint, base_z, projection, max_roof_height))

    return _sort_drawables(drawables, projection)


@dataclass(frozen=True)
class _ProjectedFace:
    polygon: Polygon
    # Depth = a*x + b*y + c in orthographic screen coordinates (Y up).
    plane: tuple[float, float, float]


def _projected_face(drawable: _Drawable, projection: _Projection) -> _ProjectedFace | None:
    points = [projection.camera(p) for p in drawable.primitives[0][1]]
    polygon = Polygon([(x, y) for x, y, depth in points])
    if polygon.is_empty or polygon.area < 1e-10:
        return None  # a face seen exactly edge-on has no opaque area
    x0, y0, d0 = points[0]
    for i in range(1, len(points) - 1):
        x1, y1, d1 = points[i]
        x2, y2, d2 = points[i + 1]
        determinant = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(determinant) < 1e-10:
            continue
        a = ((d1 - d0) * (y2 - y0) - (d2 - d0) * (y1 - y0)) / determinant
        b = ((x1 - x0) * (d2 - d0) - (x2 - x0) * (d1 - d0)) / determinant
        return _ProjectedFace(polygon, (a, b, d0 - a * x0 - b * y0))
    return None


def _plane_extrema(geometry, plane: tuple[float, float, float]) -> tuple[float, float]:
    """Return the extrema of an affine plane over polygonal geometry."""
    a, b, c = plane
    polygons = [geometry] if isinstance(geometry, Polygon) else geometry.geoms
    values = [
        a * x + b * y + c
        for polygon in polygons
        if isinstance(polygon, Polygon)
        for ring in (polygon.exterior, *polygon.interiors)
        for x, y in ring.coords[:-1]
    ]
    return min(values), max(values)


def _sort_drawables(drawables: list[_Drawable], projection: _Projection) -> list[_Drawable]:
    """Sort opaque faces far-to-near using depth over their actual overlap.

    Mean face depth is not sufficient for a tall facade whose visible overlap is behind a
    shorter roof. Pairwise constraints fix those cases, including faces meeting at an edge.
    Exact visibility clipping handles depth reversals and cycles.
    """
    faces = [_projected_face(drawable, projection) for drawable in drawables]
    polygons = [face.polygon if face else Polygon() for face in faces]
    tree = STRtree(polygons)
    successors: list[set[int]] = [set() for _ in drawables]
    indegree = [0] * len(drawables)

    for face_index, face in enumerate(faces):
        if face is None:
            continue
        for candidate in tree.query(face.polygon):
            other_index = int(candidate)
            if other_index <= face_index or faces[other_index] is None:
                continue
            other = faces[other_index]
            overlap = face.polygon.intersection(other.polygon)
            if overlap.is_empty or overlap.area < 1e-10:
                continue
            difference = (
                other.plane[0] - face.plane[0],
                other.plane[1] - face.plane[1],
                other.plane[2] - face.plane[2],
            )
            minimum, maximum = _plane_extrema(overlap, difference)
            if minimum >= -1e-7 and maximum > 1e-7:
                far, near = face_index, other_index
            elif maximum <= 1e-7 and minimum < -1e-7:
                far, near = other_index, face_index
            else:
                continue
            successors[far].add(near)
            indegree[near] += 1

    remaining = set(range(len(drawables)))
    ready: list[tuple[float, int]] = []
    for drawable_index, degree in enumerate(indegree):
        if degree == 0:
            heappush(ready, (drawables[drawable_index].depth, drawable_index))

    ordered: list[_Drawable] = []
    while remaining:
        if ready:
            _, drawable_index = heappop(ready)
            if drawable_index not in remaining:
                continue
        else:
            # Interlocking surfaces can form a painter-order cycle. Visibility
            # clipping resolves it; break the cycle deterministically by mean depth.
            drawable_index = min(remaining, key=lambda index: (drawables[index].depth, index))
        remaining.remove(drawable_index)
        ordered.append(drawables[drawable_index])
        for successor in successors[drawable_index]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heappush(ready, (drawables[successor].depth, successor))

    return ordered


def _nearer_region(overlap, plane: tuple[float, float, float]):
    """Clip overlap to the half-plane where the occluder is in front.

    A single mean depth cannot order long roofs or intersecting projected faces. Comparing
    their affine depth functions gives the correct visible portion.
    """
    a, b, c = plane
    xmin, ymin, xmax, ymax = overlap.bounds
    corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

    def distance(p):
        return a * p[0] + b * p[1] + c

    # Use tolerance only to reject coplanar faces, not to shift the cut: shifting
    # both cuts leaves a strip where both faces are visible and order matters.
    values = [distance(p) for p in corners]
    if max(values) <= 1e-7:
        return None
    if min(values) >= 0:
        return overlap
    clipped = []
    for start, end in zip(corners, corners[1:] + corners[:1]):
        ds, de = distance(start), distance(end)
        if ds >= 0:
            clipped.append(start)
        if (ds >= 0) != (de >= 0):
            t = ds / (ds - de)
            clipped.append((start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1])))
    if len(clipped) < 3:
        return None
    return overlap.intersection(Polygon(clipped))


def _visible_regions(drawables: list[_Drawable], projection: _Projection) -> list:
    """Exact planar visible regions for opaque faces, kept as vector geometry.

    Wall decorations are intersected with their wall's visible region. STRtree limits
    comparisons to overlapping screen bounds. Courtyards remain open.
    """
    faces = [_projected_face(d, projection) for d in drawables]
    polygons = [f.polygon if f else Polygon() for f in faces]
    tree = STRtree(polygons)
    visible = []
    for i, face in enumerate(faces):
        if face is None:
            visible.append(Polygon())
            continue
        occluded = []
        for j in tree.query(face.polygon):
            if i == j or faces[j] is None:
                continue
            other = faces[j]
            overlap = face.polygon.intersection(other.polygon)
            if overlap.is_empty or overlap.area < 1e-10:
                continue
            difference = (
                other.plane[0] - face.plane[0],
                other.plane[1] - face.plane[1],
                other.plane[2] - face.plane[2],
            )
            nearer = _nearer_region(overlap, difference)
            if nearer is not None and not nearer.is_empty:
                occluded.append(nearer)
        visible.append(face.polygon.difference(unary_union(occluded)) if occluded else face.polygon)
    return visible


def _clip_path_data(geometry, canvas) -> str:
    """Encode all visible polygon parts and holes using the even-odd fill rule."""
    parts = [geometry] if isinstance(geometry, Polygon) else list(getattr(geometry, "geoms", []))
    commands = []
    for part in parts:
        if not isinstance(part, Polygon):
            continue
        for ring in (part.exterior, *part.interiors):
            points = [canvas(p) for p in ring.coords[:-1]]
            if not points:
                continue
            commands.append(
                f"M {points[0][0]:.6f},{points[0][1]:.6f} "
                + " ".join(f"L {x:.6f},{y:.6f}" for x, y in points[1:])
                + " Z"
            )
    return " ".join(commands)


def _line_path_data(geometry, canvas) -> str:
    """Encode visible original edge fragments without connecting separate parts."""
    if isinstance(geometry, LineString):
        points = [canvas(point) for point in geometry.coords]
        if len(points) < 2:
            return ""
        return f"M {points[0][0]:.6f},{points[0][1]:.6f} " + " ".join(
            f"L {x:.6f},{y:.6f}" for x, y in points[1:]
        )
    return " ".join(
        data for part in getattr(geometry, "geoms", []) if (data := _line_path_data(part, canvas))
    )


def _all_projected_points(drawables: Iterable[_Drawable], projection: _Projection) -> list[Point2D]:
    return [
        projection.project(point)
        for drawable in drawables
        for _, points, _ in drawable.primitives
        for point in points
    ]


def _canvas_transform(points: list[Point2D], width: int, height: int, margin: float):
    if not points:
        raise SvgError("The SVG scene contains no points")

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    available_width = max(width - 2.0 * margin, 1.0)
    available_height = max(height - 2.0 * margin, 1.0)
    scale = min(available_width / span_x, available_height / span_y)

    rendered_width = span_x * scale
    rendered_height = span_y * scale
    offset_x = (width - rendered_width) / 2.0
    offset_y = (height - rendered_height) / 2.0

    def transform(point: Point2D) -> Point2D:
        x, y = point
        # SVG Y grows downwards, while our projected Y grows upwards.
        return (
            offset_x + (x - min_x) * scale,
            offset_y + (max_y - y) * scale,
        )

    return transform


def _points_attribute(points: Iterable[Point2D]) -> str:
    return " ".join(f"{x:.6f},{y:.6f}" for x, y in points)


def save_svg(
    footprints: Iterable[Footprint],
    path: str | Path,
    *,
    width: int = DEFAULT_SVG_WIDTH,
    height: int = DEFAULT_SVG_HEIGHT,
    margin: float = DEFAULT_SVG_MARGIN,
    azimuth: float = DEFAULT_CAMERA_AZIMUTH,
    elevation: float = DEFAULT_CAMERA_ELEVATION,
    max_roof_height: float = DEFAULT_MAX_ROOF_HEIGHT,
    background: str = "white",
) -> Path:
    """Write an orthographic SVG preview of ``footprints`` to ``path``.

    The returned SVG is a presentation asset, not a replacement for the OSM:
    simulation geometry continues to be generated by :func:`geogen.osm.build_model`.
    Fills and original edges are clipped geometrically before writing SVG paths,
    so occlusion does not depend on clipPath support or shared document IDs.
    """
    footprints = list(footprints)
    if not footprints:
        raise SvgError("At least one footprint is required to generate an SVG")
    if width <= 0 or height <= 0:
        raise SvgError("SVG width and height must be positive")
    if margin < 0 or 2 * margin >= min(width, height):
        raise SvgError("SVG margin is too large for the requested canvas")
    if not 0.0 <= elevation < 90.0:
        raise SvgError("SVG camera elevation must be between 0 and 90 degrees")

    projection = _Projection(
        origin=bounding_center(footprints), azimuth=azimuth, elevation=elevation
    )
    drawables = _build_scene(footprints, projection, max_roof_height)
    projected_points = _all_projected_points(drawables, projection)
    canvas = _canvas_transform(projected_points, width, height, margin)

    svg = ET.Element(
        f"{{{_SVG_NS}}}svg",
        {
            "width": str(width),
            "height": str(height),
            "viewBox": f"0 0 {width} {height}",
            "role": "img",
            "aria-label": "3D building geometry preview",
        },
    )
    ET.SubElement(svg, f"{{{_SVG_NS}}}title").text = "Geogen building preview"
    ET.SubElement(svg, f"{{{_SVG_NS}}}metadata").text = (
        "Generated by geogen from provider-derived building footprints"
    )

    style = ET.SubElement(svg, f"{{{_SVG_NS}}}style")
    style.text = """
.wall { fill: #aa9859; stroke: #252525; stroke-width: 1; stroke-linejoin: round; }
.roof { fill: #7b3d3d; stroke: #252525; stroke-width: 1; stroke-linejoin: round; }
.window { fill: #7d9faa; stroke: #35515a; stroke-width: 0.8; }
.storey-line { fill: none; stroke: #4b4637; stroke-width: 0.7; }
""".strip()

    ET.SubElement(
        svg,
        f"{{{_SVG_NS}}}rect",
        {
            "class": "background",
            "x": "0",
            "y": "0",
            "width": "100%",
            "height": "100%",
            "fill": background,
        },
    )

    scene = ET.SubElement(svg, f"{{{_SVG_NS}}}g", {"id": "scene"})
    visible_regions = _visible_regions(drawables, projection)
    for index, (drawable, visible) in enumerate(zip(drawables, visible_regions)):
        if visible.is_empty or visible.area < 1e-10:
            continue
        group = ET.SubElement(scene, f"{{{_SVG_NS}}}g", {"id": f"face-{index}"})
        for primitive, points_3d, css_class in drawable.primitives:
            projected = [projection.project(point) for point in points_3d]
            if primitive == "polygon":
                polygon = Polygon(projected)
                fill = polygon.intersection(visible)
                if not fill.is_empty and fill.area > 1e-10:
                    ET.SubElement(
                        group,
                        f"{{{_SVG_NS}}}path",
                        {
                            "class": css_class,
                            "d": _clip_path_data(fill, canvas),
                            "fill-rule": "evenodd",
                            "style": "stroke: none",
                        },
                    )
                # Stroke only original edges. Stroking the clipped polygon would
                # invent outlines along an occluder's boundary or a courtyard cut.
                edges = polygon.boundary
            else:
                edges = LineString(projected)
            stroke = edges.intersection(visible)
            data = _line_path_data(stroke, canvas)
            if data:
                ET.SubElement(
                    group,
                    f"{{{_SVG_NS}}}path",
                    {"class": css_class, "d": data, "style": "fill: none"},
                )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(svg)
    try:
        ET.indent(tree, space="  ")
        tree.write(destination, encoding="utf-8", xml_declaration=True)
    except OSError as error:
        raise SvgError(f"Could not write SVG preview to {destination}: {error}") from error

    LOGGER.info("Wrote SVG preview to %s", destination)
    return destination
