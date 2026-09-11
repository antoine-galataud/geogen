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
from dataclasses import dataclass, field
from pathlib import Path

from geogen.envelope import DEFAULT_MAX_ROOF_HEIGHT, cardinal_point
from geogen.geometry import Footprint, bounding_center, roof_apex

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
    ring = footprint.ring
    pitch = footprint.envelope.roof_pitch
    apex = roof_apex(ring, pitch, max_roof_height) if pitch else None

    if apex is None:
        top = tuple((x, y, top_z) for x, y in ring)
        drawable = _Drawable(depth=projection.depth(top))
        drawable.add_polygon(top, "roof flat-roof")
        return [drawable]

    apex_point = (apex[0], apex[1], top_z + apex[2])
    drawables: list[_Drawable] = []
    for index, start in enumerate(ring):
        end = ring[(index + 1) % len(ring)]
        triangle = (
            (start[0], start[1], top_z),
            (end[0], end[1], top_z),
            apex_point,
        )
        drawable = _Drawable(depth=projection.depth(triangle))
        drawable.add_polygon(triangle, "roof pitched-roof")
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
        for edge_index in range(len(footprint.ring)):
            drawables.append(_wall_drawable(footprint, edge_index, base_z, projection))
        drawables.extend(_roof_drawables(footprint, base_z, projection, max_roof_height))

    # Orthographic painter's algorithm: far groups first, near groups last.
    drawables.sort(key=lambda drawable: drawable.depth)
    return drawables


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
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


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
        "Generated by geogen from BDNB-derived building footprints"
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
    for drawable in drawables:
        group = ET.SubElement(scene, f"{{{_SVG_NS}}}g")
        for primitive, points_3d, css_class in drawable.primitives:
            projected = [canvas(projection.project(point)) for point in points_3d]
            if primitive == "polygon":
                ET.SubElement(
                    group,
                    f"{{{_SVG_NS}}}polygon",
                    {"class": css_class, "points": _points_attribute(projected)},
                )
            else:
                ET.SubElement(
                    group,
                    f"{{{_SVG_NS}}}polyline",
                    {"class": css_class, "points": _points_attribute(projected)},
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
