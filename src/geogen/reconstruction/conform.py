"""Rebuild Roofer roofs as 2.5D solids over the exact source footprint.

Roofer re-quantizes and snaps its input outline, so independently reconstructed
neighbours never share exactly coincident party walls. Here only the LiDAR roof
evidence is kept: the roof partition and one fitted plane per roof part. Walls
and the ground are regenerated from the source footprint, which adjacent units
share exactly, so neighbouring solids touch without intersecting.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import polygonize, unary_union

from .types import GeometryScene, ReconstructionError, Solid, Surface

SNAP_TOLERANCE = 0.05
DEFAULT_TOLERANCE = 0.05
MIN_WALL_HEIGHT = 0.1
MIN_ROOF_NORMAL_Z = 0.05
# Narrower roof parts, steps or wall pieces are merged (OpenStudio needs > 1 cm).
MIN_FEATURE = 0.02
ON_EDGE = 1e-9
MAX_REGULARIZE_ROUNDS = 100


def _fail(message, code="geometry_invalid"):
    raise ReconstructionError(code, message)


def _key(point):
    return (round(point[0], 6), round(point[1], 6))


def _fit_plane(surface, tolerance):
    points = np.array([p for ring in (surface.exterior, *surface.interiors) for p in ring], float)
    centre = points.mean(axis=0)
    normal = np.linalg.svd(points - centre)[2][-1]
    if normal[2] < 0:
        normal = -normal
    if normal[2] < MIN_ROOF_NORMAL_Z:
        _fail(f"Near-vertical roof surface {surface.face_id}", "unsupported_geometry")
    residual = float(np.abs((points - centre) @ normal).max())
    if residual > tolerance:
        _fail(
            f"Non-planar source face {surface.face_id}: deviates {residual:.3g} m from its "
            f"plane, exceeds {tolerance:g} m input tolerance",
            "planarity_tolerance_exceeded",
        )
    return tuple(map(float, centre)), tuple(map(float, normal)), residual


def _height(plane, x, y):
    (cx, cy, cz), (nx, ny, nz) = plane
    return cz - (nx * (x - cx) + ny * (y - cy)) / nz


def _rings(polygon):
    return [list(polygon.exterior.coords)[:-1], *(list(r.coords)[:-1] for r in polygon.interiors)]


class _Snapper:
    """Move outline-adjacent Roofer vertices onto the source footprint boundary."""

    def __init__(self, footprint, tolerance):
        self.tolerance = tolerance
        self.rings = [list(ring) for ring in _rings(footprint)]
        self.vertices = [p for ring in self.rings for p in ring]
        self.boundary = footprint.boundary
        self.inserted = defaultdict(set)
        self.cache = {}

    def __call__(self, point):
        key = _key(point)
        if key in self.cache:
            return self.cache[key]
        x, y = point
        result = (x, y)
        if self.boundary.distance(Point(x, y)) <= self.tolerance:
            nearest = min(self.vertices, key=lambda v: math.dist(v, (x, y)))
            if math.dist(nearest, (x, y)) <= 2 * self.tolerance:
                result = tuple(nearest)
            else:
                best = None
                for r, ring in enumerate(self.rings):
                    for i, a in enumerate(ring):
                        b = ring[(i + 1) % len(ring)]
                        dx, dy = b[0] - a[0], b[1] - a[1]
                        t = ((x - a[0]) * dx + (y - a[1]) * dy) / (dx * dx + dy * dy)
                        t = min(1.0, max(0.0, t))
                        q = (a[0] + t * dx, a[1] + t * dy)
                        d = math.dist(q, (x, y))
                        if best is None or d < best[0]:
                            best = (d, r, i, t, q)
                _, r, i, t, q = best
                if 0 < t < 1:
                    self.inserted[(r, i)].add((t, q))
                    result = q
                else:
                    result = tuple(self.rings[r][i if t == 0 else (i + 1) % len(self.rings[r])])
        self.cache[key] = result
        return result

    def lines(self):
        lines = []
        for r, ring in enumerate(self.rings):
            points = []
            for i, a in enumerate(ring):
                points.append(a)
                points.extend(q for _, q in sorted(self.inserted.get((r, i), ())))
            lines.append(LineString([*points, points[0]]))
        return lines


def _edges(lines):
    """Node linework and return its undirected straight edges."""
    noded = unary_union(lines)
    result = set()
    for line in getattr(noded, "geoms", [noded]):
        coords = [tuple(c) for c in line.coords]
        for a, b in zip(coords, coords[1:]):
            if a != b:
                result.add((min(a, b), max(a, b)))
    return result


def _regularize(footprint, lines, min_feature):
    """Collapse arrangement features narrower than ``min_feature``.

    Nodes closer than ``min_feature`` merge and nodes nearly touching an edge
    move onto it, so no roof part, step or wall piece is thinner than that.
    Source footprint vertices never move; boundary nodes stay on the boundary.
    """
    rings = [LineString([*ring, ring[0]]) for ring in _rings(footprint)]
    corners = {tuple(p) for ring in _rings(footprint) for p in ring}
    boundary = footprint.boundary
    edges = _edges(lines)
    merged = moved = 0
    for _ in range(MAX_REGULARIZE_ROUNDS):
        nodes = sorted({p for edge in edges for p in edge})
        points = shapely.points(nodes)
        position = {}
        for r, ring in enumerate(rings):
            for index in np.flatnonzero(shapely.distance(ring, points) <= ON_EDGE):
                position.setdefault(nodes[index], (r, ring.project(points[index])))

        def rank(p):
            return 0 if p in corners else (1 if p in position else 2)

        def along(p, q):
            (rp, sp), (rq, sq) = position[p], position[q]
            gap = abs(sp - sq)
            return rp == rq and min(gap, rings[rp].length - gap) < min_feature

        parent = {p: p for p in nodes}

        def find(p):
            while parent[p] != p:
                parent[p] = parent[parent[p]]
                p = parent[p]
            return p

        # Close node pairs merge; two boundary nodes only if close along the boundary.
        left, right = shapely.STRtree(points).query(points, "dwithin", min_feature)
        for i, j in sorted(zip(left.tolist(), right.tolist())):
            p, q = nodes[i], nodes[j]
            if i >= j or (rank(p) < 2 and rank(q) < 2 and not along(p, q)):
                continue
            root_p, root_q = find(p), find(q)
            if root_p == root_q or (rank(root_p) == 0 and rank(root_q) == 0):
                continue
            keep, drop = sorted((root_p, root_q), key=lambda n: (rank(n), n))
            parent[drop] = keep
        mapping = {p: find(p) for p in nodes if find(p) != p}
        if not mapping:
            # Nodes nearly touching an edge: move them onto it (or bend the edge).
            ordered = sorted(edges)
            segments = [LineString(e) for e in ordered]
            middles = shapely.points([((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in ordered])
            on_boundary = shapely.distance(boundary, middles) <= ON_EDGE
            found, near = shapely.STRtree(segments).query(points, "dwithin", min_feature)
            used_nodes, used_edges, splits = set(), set(), {}
            for i, k in sorted(zip(found.tolist(), near.tolist())):
                p, (a, b) = nodes[i], ordered[k]
                if p in (a, b) or i in used_nodes or k in used_edges:
                    continue
                if on_boundary[k] and rank(p) < 2:
                    continue
                dx, dy = b[0] - a[0], b[1] - a[1]
                t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)
                if not 0 < t < 1:
                    continue
                q = p if rank(p) < 2 else (a[0] + t * dx, a[1] + t * dy)
                splits[a, b] = q
                if q != p:
                    mapping[p] = q
                used_nodes.add(i)
                used_edges.add(k)
            if not splits:
                return edges, {"merged_nodes": merged, "moved_nodes": moved}
            for (a, b), q in splits.items():
                edges.discard((a, b))
                edges.update({(min(a, q), max(a, q)), (min(q, b), max(q, b))})
            moved += len(splits)
        else:
            merged += len(mapping)
        remapped = {(mapping.get(a, a), mapping.get(b, b)) for a, b in edges}
        edges = _edges([LineString(e) for e in sorted(remapped) if e[0] != e[1]])
    _fail(
        f"Roof arrangement could not be regularized to a {min_feature:g} m minimum feature",
        "unsupported_geometry",
    )


def _node_rings(regions):
    """Insert every arrangement vertex lying on a ring edge (removes T-junctions)."""
    coordinates = sorted(
        {p for rings in regions for ring in rings for p in ring}
    )
    tree = shapely.STRtree([Point(p) for p in coordinates])
    result = []
    for rings in regions:
        noded_rings = []
        for ring in rings:
            noded = []
            for a, b in zip(ring, (*ring[1:], ring[0])):
                noded.append(a)
                dx, dy = b[0] - a[0], b[1] - a[1]
                squared = dx * dx + dy * dy
                hits = []
                for index in tree.query(LineString([a, b]).buffer(ON_EDGE * 10)):
                    p = coordinates[index]
                    if p == a or p == b:
                        continue
                    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / squared
                    if 0 < t < 1 and math.dist(p, (a[0] + t * dx, a[1] + t * dy)) <= ON_EDGE:
                        hits.append((t, p))
                noded.extend(p for _, p in sorted(hits))
            noded_rings.append(noded)
        result.append(noded_rings)
    return result


def _collinear(a, b, c):
    ux, uy, vx, vy = b[0] - a[0], b[1] - a[1], c[0] - b[0], c[1] - b[1]
    return ux * vx + uy * vy > 0 and abs(ux * vy - uy * vx) <= 1e-9 * math.hypot(
        ux, uy
    ) * math.hypot(vx, vy)


def _runs(ring, joinable):
    """Split a closed ring's edges into maximal straight runs (edge ``i`` is ring[i]->ring[i+1])."""
    n = len(ring)
    breaks = [
        i
        for i in range(n)
        if not (_collinear(ring[i - 1], ring[i], ring[(i + 1) % n]) and joinable(i))
    ]
    if not breaks:
        _fail("Degenerate collinear ring", "unsupported_geometry")
    runs = []
    for k, start in enumerate(breaks):
        end = breaks[(k + 1) % len(breaks)]
        runs.append(list(range(start, end if end > start else end + n)))
    return runs


def _cluster(values, tolerance):
    """Group (label, z) pairs into levels; levels differ by more than ``tolerance``."""
    ordered = sorted(values, key=lambda item: item[1])
    clusters, current = [], [ordered[0]]
    for item in ordered[1:]:
        if item[1] - current[0][1] <= tolerance:
            current.append(item)
        else:
            clusters.append(current)
            current = [item]
    clusters.append(current)
    result = {}
    for cluster in clusters:
        level = math.fsum(z for _, z in cluster) / len(cluster)
        for label, _ in cluster:
            result[label] = level
    return result


def conform_scene(
    scene: GeometryScene,
    footprint: Polygon,
    ground_z: float,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[GeometryScene, dict]:
    """Return one closed 2.5D solid over ``footprint`` carrying the Roofer roof planes.

    ``footprint`` and ``ground_z`` use the scene's local frame. Roof parts are
    snapped to the footprint within ``SNAP_TOLERANCE``; areas Roofer did not
    cover inherit the nearest roof part. Each Roofer roof face must lie within
    ``tolerance`` of its fitted plane; roof heights within ``tolerance`` at a
    shared vertex become one level, larger differences become vertical step walls.
    Roof-part details narrower than ``MIN_FEATURE`` are collapsed.
    """
    if footprint.geom_type != "Polygon" or not footprint.is_valid or footprint.is_empty:
        _fail("Source footprint must be one valid polygon")
    if not scene.solids or any(solid.interior_shells for solid in scene.solids):
        _fail("Conformation requires solids without interior shells", "unsupported_geometry")
    footprint = orient(footprint, 1.0)
    roofs = []
    for solid in scene.solids:
        for surface in solid.surfaces:
            if surface.semantic == "RoofSurface":
                roofs.append(surface)
            elif surface.semantic not in {"WallSurface", "GroundSurface"}:
                _fail(
                    f"Non-2.5D surface {surface.semantic} cannot be conformed",
                    "unsupported_geometry",
                )
    # Collapsed Roofer roof rings carry no roof evidence; walls are regenerated anyway.
    kept, parts, skipped = [], [], 0
    for surface in roofs:
        part = Polygon(
            [p[:2] for p in surface.exterior], [[p[:2] for p in r] for r in surface.interiors]
        )
        part = part if part.is_valid else part.buffer(0)
        if part.is_empty or part.area <= 1e-9:
            skipped += 1
            continue
        kept.append(surface)
        parts.append(part)
    roofs = kept
    if not roofs:
        _fail("Roofer output contains no usable roof surface")
    planes, residuals = [], []
    for surface in roofs:
        centre, normal, residual = _fit_plane(surface, tolerance)
        planes.append((centre, normal))
        residuals.append(residual)

    # Arrangement: exact footprint boundary plus snapped Roofer roof-part boundaries.
    snapper = _Snapper(footprint, SNAP_TOLERANCE)
    segments = set()
    for surface in roofs:
        for ring in (surface.exterior, *surface.interiors):
            snapped = [snapper(p[:2]) for p in ring]
            for a, b in zip(snapped, (*snapped[1:], snapped[0])):
                if a != b:
                    segments.add((min(a, b), max(a, b)))
    arrangement, regularization = _regularize(
        footprint, [*snapper.lines(), *(LineString(s) for s in sorted(segments))], MIN_FEATURE
    )
    faces = [
        face
        for face in polygonize([LineString(e) for e in sorted(arrangement)])
        if face.area > 0 and footprint.contains(face.representative_point())
    ]
    if abs(math.fsum(face.area for face in faces) - footprint.area) > 1e-6 * max(
        1.0, footprint.area
    ):
        _fail("Roof arrangement does not tile the source footprint", "unsupported_geometry")
    tree = shapely.STRtree(parts)
    labels = defaultdict(list)
    unassigned = 0.0
    for face in faces:
        overlaps = [(face.intersection(parts[i]).area, -i) for i in tree.query(face)]
        best = max(overlaps, default=(0.0, 0))
        if best[0] > 1e-9:
            label = -best[1]
        else:
            point = face.representative_point()
            label = min(range(len(parts)), key=lambda i: (parts[i].distance(point), i))
            unassigned += face.area
        labels[label].append(face)

    region_labels, region_rings = [], []
    for label in sorted(labels):
        merged = shapely.coverage_union_all(labels[label])
        for polygon in (merged.geoms if hasattr(merged, "geoms") else [merged]):
            if polygon.geom_type != "Polygon" or polygon.area <= 0:
                continue
            region_labels.append(label)
            region_rings.append([[tuple(p) for p in ring] for ring in _rings(orient(polygon))])
    ground = orient(shapely.coverage_union_all(faces), 1.0)
    if ground.geom_type != "Polygon":
        _fail("Conformed footprint is not one polygon", "unsupported_geometry")
    ground_rings = [[tuple(p) for p in ring] for ring in _rings(ground)]
    *region_rings, ground_rings = _node_rings([*region_rings, ground_rings])

    def levels():
        incident = defaultdict(set)
        for label, rings in zip(region_labels, region_rings):
            for ring in rings:
                for p in ring:
                    incident[p].add(label)
        return {
            p: _cluster(
                [(label, _height(planes[label], *p)) for label in sorted(found)], tolerance
            )
            for p, found in incident.items()
        }

    def owners():
        result = {}
        for index, rings in enumerate(region_rings):
            for ring in rings:
                for a, b in zip(ring, (*ring[1:], ring[0])):
                    result[a, b] = index
        return result

    # Split internal edges where neighbouring roof parts cross; walls stay one-sided.
    z = levels()
    edges = owners()
    splits = {}
    for (a, b), r in edges.items():
        s = edges.get((b, a))
        if s is None or (b, a) in splits:
            continue
        i, j = region_labels[r], region_labels[s]
        da, db = z[a][i] - z[a][j], z[b][i] - z[b][j]
        if da * db < 0:
            t = da / (da - db)
            splits[a, b] = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
    if splits:
        for rings in region_rings:
            for k, ring in enumerate(rings):
                updated = []
                for a, b in zip(ring, (*ring[1:], ring[0])):
                    updated.append(a)
                    if (a, b) in splits:
                        updated.append(splits[a, b])
                    elif (b, a) in splits:
                        updated.append(splits[b, a])
                rings[k] = updated
        z = levels()
        edges = owners()

    solid_id = scene.solids[0].solid_id
    reconstruction_id = scene.solids[0].reconstruction_id
    source_object = roofs[0].source_object_id
    provenance = (*roofs[0].provenance, "conformed_to_source_footprint")
    surfaces = []

    def add(kind, semantic, ring, holes=(), extra=()):
        face_id = f"{solid_id}:conformed:{kind}:{len(surfaces)}"
        surfaces.append(
            Surface(
                face_id,
                tuple(ring),
                tuple(tuple(h) for h in holes),
                semantic=semantic,
                source_object_id=source_object,
                source_face_id=face_id,
                provenance=(*provenance, *extra),
            )
        )

    for label, rings in zip(region_labels, region_rings):
        lifted = [[(x, y, z[x, y][label]) for x, y in ring] for ring in rings]
        add("roof", "RoofSurface", lifted[0], lifted[1:], (roofs[label].source_face_id,))
    add(
        "ground",
        "GroundSurface",
        [(x, y, ground_z) for x, y in reversed(ground_rings[0])],
        [[(x, y, ground_z) for x, y in reversed(ring)] for ring in ground_rings[1:]],
    )

    def height(region, point):
        return z[point][region_labels[region]]

    if any(
        height(r, p) - ground_z < MIN_WALL_HEIGHT
        for r, rings in enumerate(region_rings)
        for ring in rings
        for p in ring
    ):
        _fail("Roof is not above the ground reference", "unsupported_geometry")

    # Exterior walls: one vertical polygon per straight run of the footprint boundary.
    for ring in ground_rings:
        n = len(ring)
        owners_along = [edges.get((a, b)) for a, b in zip(ring, (*ring[1:], ring[0]))]
        if None in owners_along:
            _fail("Roof partition does not reach the footprint boundary", "unsupported_geometry")
        for run in _runs(ring, lambda i: True):
            owners_run = [owners_along[i % n] for i in run]
            points = [ring[i % n] for i in range(run[0], run[-1] + 2)]
            top = [(*points[-1], height(owners_run[-1], points[-1]))]
            for k in range(len(points) - 2, 0, -1):
                after = height(owners_run[k], points[k])
                before = height(owners_run[k - 1], points[k])
                top.append((*points[k], after))
                if before != after:
                    top.append((*points[k], before))
            top.append((*points[0], height(owners_run[0], points[0])))
            add("wall", "WallSurface", [(*p, ground_z) for p in points] + top)

    # Step walls between roof regions, emitted once from the higher side.
    step_walls = 0
    for r, rings in enumerate(region_rings):
        for ring in rings:
            n = len(ring)

            def step(i, ring=ring, n=n):
                a, b = ring[i % n], ring[(i + 1) % n]
                s = edges.get((b, a))
                if s is None:
                    return None
                high, low = (height(r, a), height(r, b)), (height(s, a), height(s, b))
                if high[0] < low[0] or high[1] < low[1] or high == low:
                    return None
                return s

            def joinable(i, ring=ring, n=n):
                s = step(i - 1)
                p = ring[i % n]
                return s is not None and s == step(i) and height(r, p) != height(s, p)

            for run in _runs(ring, joinable):
                s = step(run[0])
                if s is None:
                    continue
                step_walls += 1
                points = [ring[i % n] for i in range(run[0], run[-1] + 2)]
                wall = [(*p, height(s, p)) for p in points]
                wall += [(*p, height(r, p)) for p in reversed(points)]
                wall = [p for k, p in enumerate(wall) if p != wall[k - 1]]
                add("wall", "WallSurface", wall)

    solid = Solid(solid_id, reconstruction_id, tuple(surfaces), links=scene.solids[0].links)
    audit = {
        "method": "roof_planes_over_source_footprint",
        "snap_tolerance_m": SNAP_TOLERANCE,
        "step_tolerance_m": tolerance,
        "min_feature_m": MIN_FEATURE,
        **regularization,
        "input_roof_surfaces": len(roofs),
        "skipped_degenerate_roof_surfaces": skipped,
        "output_roof_surfaces": len(region_labels),
        "output_wall_surfaces": len(surfaces) - len(region_labels) - 1,
        "step_wall_surfaces": step_walls,
        "max_roof_plane_residual_m": max(residuals),
        "footprint_area_m2": footprint.area,
        "roofer_roof_area_m2": float(unary_union(parts).area),
        "unassigned_area_m2": unassigned,
        "max_level_merge_m": max(
            (
                abs(level - _height(planes[label], *p))
                for p, found in z.items()
                for label, level in found.items()
            ),
            default=0.0,
        ),
    }
    return GeometryScene(scene.frame, (solid,)), audit

