"""Conservative agreement checks, not independent surveyed-fidelity benchmarks.

NumPy is a dependency of the existing laspy dependency. Coordinates are only
translated into the scene frame; neither roofs nor supplied outlines are fitted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import laspy
import numpy as np
import shapely
from pyproj import CRS
from pyproj.exceptions import CRSError
from shapely import affinity, wkt
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from .types import GeometryScene, ReconstructionUnit, Surface

CELL_SIZE = 0.5
MAX_RESIDUAL_POINTS = 20_000
MAX_CELLS = 100_000
# Roof/ground planes evaluated per unit; cost is otherwise bounded by cells and samples.
MAX_SURFACES = 4096
CHUNK_SIZE = 100_000


@dataclass
class _Plane:
    origin: np.ndarray
    normal: np.ndarray
    u: np.ndarray
    v: np.ndarray
    polygon: Polygon
    horizontal: Polygon
    face_id: str


@dataclass
class _Evidence:
    outline: object
    roof_domain: object
    ground_domain: object
    roofs: list[_Plane]
    grounds: list[_Plane]
    grid: dict
    roof_grids: dict
    cell_keys: set = field(default_factory=set)
    occupied: set = field(default_factory=set)
    counts: dict = field(default_factory=lambda: {2: 0, 6: 0})
    seen: dict = field(default_factory=lambda: {2: 0, 6: 0})
    samples: dict = field(default_factory=lambda: {2: [], 6: []})
    outline_points: int = 0


def _plane(surface: Surface) -> _Plane:
    vertices = np.asarray(surface.exterior, dtype=float)
    if len(vertices) < 3 or not np.isfinite(vertices).all():
        raise ValueError("invalid surface vertices")
    origin = vertices[0]
    edges = vertices - origin
    u = edges[np.argmax(np.linalg.norm(edges, axis=1))]
    if np.linalg.norm(u) < 1e-8:
        raise ValueError("degenerate surface")
    u = u / np.linalg.norm(u)
    normals = np.cross(u, edges)
    normal = normals[np.argmax(np.linalg.norm(normals, axis=1))]
    if np.linalg.norm(normal) < 1e-8:
        raise ValueError("degenerate surface")
    normal = normal / np.linalg.norm(normal)
    if abs(normal[2]) < 1e-6:
        raise ValueError("vertical roof or ground")
    v = np.cross(normal, u)
    rings = [vertices, *(np.asarray(r, dtype=float) for r in surface.interiors)]
    projected = []
    for ring in rings:
        if len(ring) < 3 or not np.isfinite(ring).all():
            raise ValueError("invalid surface ring")
        delta = ring - origin
        if np.max(np.abs(delta @ normal)) > 0.02:
            raise ValueError("nonplanar surface")
        projected.append(np.column_stack((delta @ u, delta @ v)))
    polygon = Polygon(projected[0], projected[1:])
    horizontal = Polygon(vertices[:, :2], [r[:, :2] for r in rings[1:]])
    if not polygon.is_valid or not horizontal.is_valid or horizontal.area <= 1e-8:
        raise ValueError("invalid projected surface")
    return _Plane(origin, normal, u, v, polygon, horizontal, surface.face_id)


def _cells(polygon) -> dict[tuple[int, int], float]:
    xmin, ymin, xmax, ymax = polygon.bounds
    ix0, iy0 = math.floor(xmin / CELL_SIZE), math.floor(ymin / CELL_SIZE)
    ix1, iy1 = math.ceil(xmax / CELL_SIZE), math.ceil(ymax / CELL_SIZE)
    if (ix1 - ix0) * (iy1 - iy0) > MAX_CELLS:
        raise ValueError("coverage cell limit")
    grid = {}
    for ix in range(ix0, ix1):
        for iy in range(iy0, iy1):
            area = polygon.intersection(
                box(ix * CELL_SIZE, iy * CELL_SIZE,
                    (ix + 1) * CELL_SIZE, (iy + 1) * CELL_SIZE)
            ).area
            if area > 1e-10:
                grid[ix, iy] = area
    return grid


def _coverage(grid, occupied) -> float | None:
    area = sum(grid.values())
    return sum(a for cell, a in grid.items() if cell in occupied) / area if area else None


def _largest_gap(grid, occupied) -> float:
    remaining = set(grid).difference(occupied)
    largest = 0.0
    while remaining:
        stack = [remaining.pop()]
        area = 0.0
        while stack:
            ix, iy = stack.pop()
            area += grid[ix, iy]
            for cell in ((ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)):
                if cell in remaining:
                    remaining.remove(cell)
                    stack.append(cell)
        largest = max(largest, area)
    return largest


def _parts(polygon):
    return list(polygon.geoms) if polygon.geom_type == "MultiPolygon" else [polygon]


def _agreement(outline, model, prefix, metrics, reasons):
    intersection = outline.intersection(model).area
    metrics[f"{prefix}_iou"] = intersection / outline.union(model).area
    metrics[f"{prefix}_outline_coverage"] = intersection / outline.area
    metrics[f"{prefix}_outline_precision"] = intersection / model.area
    metrics[f"{prefix}_hausdorff_m"] = outline.hausdorff_distance(model)
    component_coverage = [p.intersection(model).area / p.area for p in _parts(outline)]
    metrics[f"{prefix}_component_coverage"] = component_coverage
    holes = [Polygon(ring) for p in _parts(outline) for ring in p.interiors]
    filled = sum(hole.intersection(model).area for hole in holes)
    metrics[f"{prefix}_filled_courtyard_area_m2"] = filled
    if filled > 0.01:
        reasons.append(f"{prefix}_fills_courtyard")
    if (metrics[f"{prefix}_outline_coverage"] < 0.98
            or metrics[f"{prefix}_outline_precision"] < 0.98
            or metrics[f"{prefix}_hausdorff_m"] > 0.5
            or min(component_coverage) < 0.98):
        reasons.append(f"{prefix}_outline_mismatch")


def _residuals(points: np.ndarray, planes: list[_Plane]) -> np.ndarray:
    """Distance to bounded planar polygons in 3D, including their hole edges."""
    nearest = np.full(len(points), np.inf)
    for plane in planes:
        delta = points - plane.origin
        projected = shapely.points(np.column_stack((delta @ plane.u, delta @ plane.v)))
        lateral = shapely.distance(projected, plane.polygon)
        distance = np.hypot(delta @ plane.normal, lateral)
        nearest = np.minimum(nearest, distance)
    return nearest


def _ground_residuals(points: np.ndarray, planes: list[_Plane]) -> np.ndarray:
    # Class-2 returns often surround, rather than lie underneath, a building.
    # Compare z to the nearest bounded groundprint location, without extrapolating
    # the model's plane outside its polygon or using horizontal setback as error.
    xy = shapely.points(points[:, :2])
    nearest_xy = np.full(len(points), np.inf)
    residual = np.full(len(points), np.inf)
    for plane in planes:
        lines = shapely.shortest_line(xy, plane.horizontal)
        nearest = shapely.get_point(lines, -1)
        x, y = shapely.get_x(nearest), shapely.get_y(nearest)
        distance = shapely.distance(xy, plane.horizontal)
        z = plane.origin[2] - (
            plane.normal[0] * (x - plane.origin[0])
            + plane.normal[1] * (y - plane.origin[1])
        ) / plane.normal[2]
        update = distance < nearest_xy
        residual[update] = np.abs(points[update, 2] - z[update])
        nearest_xy[update] = distance[update]
    return residual


def _metrics() -> dict:
    names = [
        "outline_extent_uncertainty", "wall_extent_assumption",
        "original_lidar_point_count", "working_lidar_point_count",
        "working_class_2_count", "working_class_6_count", "invalid_lidar_point_count",
        "roof_point_count", "ground_point_count", "roof_outline_point_count",
        "roof_density_points_m2", "roof_horizontal_coverage", "roof_gap_area_m2",
        "roof_gap_cell_count", "roof_largest_gap_area_m2",
        "roof_surface_coverage", "minimum_roof_surface_coverage",
        "roof_residual_sample_count", "ground_residual_sample_count",
        "roof_residual_median_m", "roof_residual_p95_m",
        "ground_residual_median_m", "ground_residual_p95_m",
    ]
    for prefix in ("roof", "ground"):
        names.extend(f"{prefix}_{name}" for name in (
            "iou", "outline_coverage", "outline_precision", "hausdorff_m",
            "component_coverage", "filled_courtyard_area_m2",
        ))
    return {name: None for name in names}


def _header_crs_reasons(header) -> list[str]:
    try:
        crs = header.parse_crs()
        if crs is None:
            return ["missing_lidar_crs"]
        reasons = []
        if not crs.to_2d().equals(CRS.from_epsg(2154)):
            reasons.append("contradictory_lidar_horizontal_crs")
        for component in crs.sub_crs_list:
            if component.is_vertical and component.to_epsg() != 5720:
                reasons.append("contradictory_lidar_vertical_crs")
        return reasons
    except (CRSError, ValueError):
        return ["invalid_lidar_crs"]


def assess_evidence(
    scene: GeometryScene,
    units: tuple[ReconstructionUnit, ...],
    lidar_path: Path,
) -> dict[str, dict]:
    """Assess working classified LiDAR; a pass is not surveyed fidelity.

    Raw working-file counts and coverage use every finite classified return.
    Residuals use deterministic, evenly spaced input-order samples when capped;
    no residual-based outlier rejection is performed. Original acquisition
    counts are unknown because this API only receives the working LAS/LAZ.
    """
    results = {}
    contexts = {}
    for unit in units:
        metrics, reasons = _metrics(), []
        results[unit.reconstruction_id] = {
            "fidelity_status": "needs_review",
            "benchmark_reference": "not_evaluated",
            "metrics": metrics,
            "reasons": reasons,
            "method": {
                "coverage_cell_size_m": CELL_SIZE,
                "residual_point_cap_per_class": MAX_RESIDUAL_POINTS,
                "sampling": "deterministic_input_order_stride",
                "outlier_removal": False,
                "ground_context_buffer_m": 2.0,
                "residual_median_threshold_m": 0.2,
                "residual_p95_threshold_m": 0.5,
                "minimum_roof_points": 20,
                "minimum_ground_points": 10,
                "minimum_roof_density_points_m2": 1.0,
                "minimum_horizontal_coverage": 0.8,
                "minimum_surface_coverage": 0.7,
                "courtyard_area_epsilon_m2": 0.01,
            },
        }
        if scene.frame.projected_crs.upper() != "EPSG:2154" or unit.crs.upper() != "EPSG:2154":
            reasons.append("unverified_projected_crs")
        if scene.frame.vertical_reference.upper() not in {"IGN69", "NGF-IGN69", "EPSG:5720"}:
            reasons.append("unverified_vertical_reference")
        if not np.isfinite(scene.frame.origin).all():
            reasons.append("invalid_frame_origin")
            continue
        if unit.outline_type == "wall_footprint":
            metrics["outline_extent_uncertainty"] = "wall_footprint_may_differ_from_roofprint"
        assumed_outline = (
            unit.outline_type == "unknown"
            and "wall_extent_assumption=roofprint_extruded" in unit.decisions
        )
        if assumed_outline:
            metrics["outline_extent_uncertainty"] = "unknown_outline_with_assumed_extruded_roofprint"
            metrics["wall_extent_assumption"] = "roofprint_extruded"
        if unit.outline_type not in {"roofprint", "wall_footprint"} and not assumed_outline:
            reasons.append("undefined_outline_semantics")
        surfaces = [
            s for solid in scene.solids if solid.reconstruction_id == unit.reconstruction_id
            for s in solid.surfaces
        ]
        if not surfaces:
            reasons.append("missing_reconstructed_surfaces")
            continue
        if sum(s.semantic in {"RoofSurface", "GroundSurface"} for s in surfaces) > MAX_SURFACES:
            reasons.append("surface_computation_limit")
            continue
        known_semantics = {
            "RoofSurface", "GroundSurface", "WallSurface",
            "OuterFloorSurface", "OuterCeilingSurface",
        }
        if any(s.semantic not in known_semantics for s in surfaces):
            reasons.append("undefined_surface_semantics")
        if any(s.semantic in {"OuterFloorSurface", "OuterCeilingSurface"} for s in surfaces):
            reasons.append("unsupported_overhang_surface_evidence")
        roofs, grounds = [], []
        for surface in surfaces:
            if surface.semantic in {"RoofSurface", "GroundSurface"}:
                try:
                    plane = _plane(surface)
                except (ValueError, shapely.errors.GEOSException):
                    reasons.append("invalid_roof_or_ground_surface")
                    continue
                (roofs if surface.semantic == "RoofSurface" else grounds).append(plane)
        if not roofs:
            reasons.append("missing_roof_surfaces")
        if not grounds:
            reasons.append("missing_ground_surfaces")
        try:
            outline = wkt.loads(unit.working_wkt)
            if (outline.geom_type not in {"Polygon", "MultiPolygon"}
                    or outline.is_empty or not outline.is_valid or outline.area <= 0):
                raise ValueError("invalid outline")
            outline = affinity.translate(
                outline, xoff=-scene.frame.origin[0], yoff=-scene.frame.origin[1]
            )
            roofprint = unary_union([p.horizontal for p in roofs])
            groundprint = unary_union([p.horizontal for p in grounds])
            if roofs:
                _agreement(outline, roofprint, "roof", metrics, reasons)
            if grounds:
                _agreement(outline, groundprint, "ground", metrics, reasons)
            roof_domain = outline.union(roofprint)
            ground_domain = outline.union(groundprint).buffer(2.0)
        except (ValueError, shapely.errors.GEOSException):
            reasons.append("invalid_outline_or_projection")
            continue
        try:
            grid = _cells(outline)
            roof_grids = {}
            cell_count = len(grid)
            for i, plane in enumerate(roofs):
                roof_grid = _cells(plane.horizontal)
                cell_count += len(roof_grid)
                if cell_count > MAX_CELLS:
                    raise ValueError("coverage cell limit")
                roof_grids[f"{i}:{plane.face_id}"] = roof_grid
        except ValueError:
            reasons.append("coverage_computation_limit")
            grid, roof_grids = {}, {}
        contexts[unit.reconstruction_id] = _Evidence(
            outline, roof_domain, ground_domain, roofs, grounds, grid, roof_grids
        )
        contexts[unit.reconstruction_id].cell_keys = set(grid)
        for roof_grid in roof_grids.values():
            contexts[unit.reconstruction_id].cell_keys.update(roof_grid)

    counts = {2: 0, 6: 0}
    total, invalid = 0, 0
    try:
        # Two streaming passes keep exact counts/coverage independent of the
        # residual cap and avoid biased "first N returns" residual samples.
        for pass_index in range(2):
            with laspy.open(lidar_path) as reader:
                if pass_index == 0:
                    header_reasons = _header_crs_reasons(reader.header)
                    for result in results.values():
                        result["reasons"].extend(header_reasons)
                for chunk in reader.chunk_iterator(CHUNK_SIZE):
                    classes = np.asarray(chunk.classification)
                    points = np.column_stack((chunk.x, chunk.y, chunk.z))
                    finite = np.isfinite(points).all(axis=1)
                    points -= np.asarray(scene.frame.origin)
                    if pass_index == 0:
                        total += len(chunk)
                        invalid += int(np.count_nonzero(~finite))
                        for cls in counts:
                            counts[cls] += int(np.count_nonzero(classes == cls))
                    for context in contexts.values():
                        for cls, domain in ((6, context.roof_domain), (2, context.ground_domain)):
                            classified = points[(classes == cls) & finite]
                            mask = shapely.intersects_xy(domain, classified[:, 0], classified[:, 1])
                            selected = classified[mask]
                            if pass_index == 0:
                                context.counts[cls] += len(selected)
                                if cls == 6:
                                    in_outline = shapely.intersects_xy(
                                        context.outline, selected[:, 0], selected[:, 1]
                                    )
                                    context.outline_points += int(np.count_nonzero(in_outline))
                                    if context.grid:
                                        cells = np.floor(selected[:, :2] / CELL_SIZE).astype(np.int64)
                                        context.occupied.update(
                                            cell for cell in map(tuple, cells) if cell in context.cell_keys
                                        )
                            else:
                                stride = max(1, math.ceil(context.counts[cls] / MAX_RESIDUAL_POINTS))
                                start = (-context.seen[cls]) % stride
                                sample = selected[start::stride]
                                if len(sample):
                                    context.samples[cls].append(sample.copy())
                                context.seen[cls] += len(selected)
    except (OSError, ValueError, laspy.errors.LaspyException):
        for result in results.values():
            result["reasons"].append("lidar_read_failed")
        return results

    for reconstruction_id, result in results.items():
        metrics, reasons = result["metrics"], result["reasons"]
        metrics.update(
            working_lidar_point_count=total,
            working_class_2_count=counts[2],
            working_class_6_count=counts[6],
            invalid_lidar_point_count=invalid,
        )
        if invalid:
            reasons.append("nonfinite_lidar_coordinates")
        context = contexts.get(reconstruction_id)
        if context is None:
            continue
        metrics.update(
            roof_point_count=context.counts[6],
            ground_point_count=context.counts[2],
            roof_outline_point_count=context.outline_points,
            roof_density_points_m2=context.outline_points / context.outline.area,
        )
        if context.counts[6] < 20:
            reasons.append("insufficient_roof_points")
        if context.counts[2] < 10:
            reasons.append("insufficient_ground_points")
        if metrics["roof_density_points_m2"] < 1.0:
            reasons.append("insufficient_roof_density")
        if context.grid:
            coverage = _coverage(context.grid, context.occupied)
            surface_coverage = {
                key: _coverage(grid, context.occupied) for key, grid in context.roof_grids.items()
            }
            metrics.update(
                roof_horizontal_coverage=coverage,
                roof_gap_area_m2=sum(a for c, a in context.grid.items() if c not in context.occupied),
                roof_gap_cell_count=len(set(context.grid).difference(context.occupied)),
                roof_largest_gap_area_m2=_largest_gap(context.grid, context.occupied),
                roof_surface_coverage=surface_coverage,
                minimum_roof_surface_coverage=min(surface_coverage.values()) if surface_coverage else None,
            )
            if coverage < 0.8:
                reasons.append("insufficient_roof_coverage")
            if surface_coverage and min(surface_coverage.values()) < 0.7:
                reasons.append("insufficient_roof_surface_coverage")
            if metrics["roof_largest_gap_area_m2"] > max(1.0, 0.1 * context.outline.area):
                reasons.append("large_roof_evidence_gap")
        for cls, label, planes, measure in (
            (6, "roof", context.roofs, _residuals),
            (2, "ground", context.grounds, _ground_residuals),
        ):
            samples = context.samples[cls]
            points = np.concatenate(samples) if samples else np.empty((0, 3))
            metrics[f"{label}_residual_sample_count"] = len(points) if planes else None
            if not len(points) or not planes:
                continue
            residuals = measure(points, planes)
            median, p95 = (float(v) for v in np.percentile(residuals, [50, 95]))
            metrics[f"{label}_residual_median_m"] = median
            metrics[f"{label}_residual_p95_m"] = p95
            if median > 0.2 or p95 > 0.5:
                reasons.append(f"{label}_residual_threshold_exceeded")
        result["reasons"] = list(dict.fromkeys(reasons))
        if not reasons:
            result["fidelity_status"] = "evidence_passed"
    return results
