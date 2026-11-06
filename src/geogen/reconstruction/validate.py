"""Conservative metric solid checks, not an independent fidelity assessment.

The independent val3dity gate belongs to the runtime adapter. These checks also
guard generated thermal partitions and exporter face decompositions.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace
from functools import lru_cache

import numpy as np
import shapely
from shapely.errors import GEOSException, UnsupportedGEOSVersionError
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import triangulate, unary_union

from .types import GeometryScene, ReconstructionError

TOLERANCE = 1e-6
PLANARITY_TOLERANCE = 1e-3
VERTEX_EQUIVALENCE_TOLERANCE = 1e-3


def fail(message, code="geometry_invalid"):
    raise ReconstructionError(code, message)


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def mul(a, scalar):
    return tuple(x * scalar for x in a)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def norm(a):
    return math.sqrt(dot(a, a))


def key(point):
    return tuple(round(v, 8) for v in point)


def _bounds_disjoint(a, b):
    return any(
        max(p[i] for p in a) < min(p[i] for p in b) - TOLERANCE
        or max(p[i] for p in b) < min(p[i] for p in a) - TOLERANCE
        for i in range(3)
    )


def face_plane(surface, *, planarity_tolerance=PLANARITY_TOLERANCE):
    """Return origin, orthonormal basis, outward normal and the planar polygon."""
    return _face_plane(surface, planarity_tolerance)


@lru_cache(maxsize=1 << 16)
def _face_plane(surface, planarity_tolerance):
    rings = (surface.exterior, *surface.interiors)
    for ring in rings:
        if len(ring) < 3 or any(len(p) != 3 or any(not math.isfinite(v) for v in p) for p in ring):
            fail(f"Invalid coordinates on {surface.face_id}")
        if len({key(p) for p in ring}) != len(ring):
            fail(f"Repeated/closing vertices on {surface.face_id}")
    origin = surface.exterior[0]
    area_vector = (0.0, 0.0, 0.0)
    for a, b in zip(surface.exterior, (*surface.exterior[1:], origin)):
        area_vector = add(area_vector, cross(sub(a, origin), sub(b, origin)))
    if norm(area_vector) < TOLERANCE**2:
        fail(f"Degenerate face {surface.face_id}")
    normal = mul(area_vector, 1 / norm(area_vector))
    first = sub(surface.exterior[1], origin)
    first = sub(first, mul(normal, dot(first, normal)))
    if norm(first) < TOLERANCE:
        fail(f"Zero-length edge on {surface.face_id}")
    u = mul(first, 1 / norm(first))
    v = cross(normal, u)
    for ring in rings:
        for point in ring:
            if abs(dot(sub(point, origin), normal)) > planarity_tolerance:
                fail(
                    f"Non-planar face {surface.face_id}: exceeds {planarity_tolerance:g} m input tolerance",
                    "planarity_tolerance_exceeded",
                )
    projected = [[(dot(sub(p, origin), u), dot(sub(p, origin), v)) for p in ring] for ring in rings]
    polygon = Polygon(projected[0], projected[1:])
    if not polygon.is_valid or polygon.area <= TOLERANCE**2:
        fail(f"Invalid face or hole topology on {surface.face_id}")
    if any(Polygon(ring).exterior.is_ccw for ring in projected[1:]):
        fail(f"Hole orientation is not opposite the exterior on {surface.face_id}")
    return origin, u, v, normal, polygon


def lift(point, plane):
    return add(plane[0], add(mul(plane[1], point[0]), mul(plane[2], point[1])))


def project(point, plane):
    relative = sub(point, plane[0])
    return dot(relative, plane[1]), dot(relative, plane[2])


def polygons(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [p for part in geometry.geoms for p in polygons(part)]
    return []


def surface_from_polygon(surface, polygon, plane, suffix):
    polygon = orient(polygon, sign=1.0)
    return replace(
        surface,
        face_id=f"{surface.face_id}{suffix}",
        exterior=tuple(lift(p, plane) for p in list(polygon.exterior.coords)[:-1]),
        interiors=tuple(
            tuple(lift(p, plane) for p in list(ring.coords)[:-1]) for ring in polygon.interiors
        ),
    )


def _face_triangulation(surface, plane):
    """Use source vertices only, checking coverage before accepting either triangulator."""
    polygon = plane[4]
    vertices = {
        project(point, plane): point
        for ring in (surface.exterior, *surface.interiors)
        for point in ring
    }

    def complete(parts):
        if not parts or any(
            part.geom_type != "Polygon"
            or part.interiors
            or len(part.exterior.coords) != 4
            or part.area <= 0
            or not polygon.covers(part)
            or any(tuple(p) not in vertices for p in list(part.exterior.coords)[:-1])
            for part in parts
        ):
            return False
        coverage = unary_union(parts)
        # Area arithmetic scales with face size and triangle count, not a geometry concession.
        epsilon = max(TOLERANCE**2, 8 * len(parts) * math.ulp(polygon.area))
        return (
            polygon.symmetric_difference(coverage).area <= epsilon
            and abs(math.fsum(part.area for part in parts) - polygon.area) <= epsilon
        )

    pieces = [part for part in triangulate(polygon) if polygon.covers(part)]
    if not complete(pieces):
        constrained = getattr(shapely, "constrained_delaunay_triangles", None)
        if constrained is None:
            fail(
                f"Unsupported triangulation for {surface.face_id}; "
                "constrained triangulation requires Shapely >= 2.1 and GEOS >= 3.10",
                "unsupported_geometry",
            )
        try:
            pieces = list(constrained(polygon).geoms)
        except (GEOSException, UnsupportedGEOSVersionError) as error:
            raise ReconstructionError(
                "unsupported_geometry", f"Unsupported triangulation for {surface.face_id}: {error}"
            ) from error
        if not complete(pieces):
            fail(f"Unsupported triangulation for {surface.face_id}", "unsupported_geometry")
    return tuple(
        tuple(vertices[tuple(p)] for p in list(orient(part).exterior.coords)[:-1])
        for part in pieces
    )


def surface_pieces(surface, *, planarity_tolerance=PLANARITY_TOLERANCE):
    """Hole-free oriented pieces, with exact area and coverage checks."""
    plane = face_plane(surface, planarity_tolerance=planarity_tolerance)
    # OpenStudio accepts concave planar polygons, but not interior rings.
    if not surface.interiors and plane_deviation(surface, plane) <= TOLERANCE:
        return (surface,)
    return tuple(
        replace(
            surface,
            face_id=f"{surface.face_id}:tile:{i}",
            interiors=(),
            exterior=triangle,
        )
        for i, triangle in enumerate(_face_triangulation(surface, plane))
    )


def plane_deviation(surface, plane=None):
    plane = plane if plane is not None else face_plane(surface)
    return max(
        abs(dot(sub(point, plane[0]), plane[3]))
        for ring in (surface.exterior, *surface.interiors)
        for point in ring
    )


def planar_pieces(surface):
    """Triangulate quantization-scale warping without relocating source vertices."""
    if plane_deviation(surface) <= TOLERANCE:
        return (surface,)
    return tuple(
        replace(
            piece,
            provenance=(*surface.provenance, "quantization_planarity:triangulated_within_0.001m"),
        )
        for piece in surface_pieces(surface)
    )


def face_triangles(surface):
    return list(_face_triangulation(surface, face_plane(surface)))


def canonical_vertices(surfaces):
    """Return deterministic existing representatives within the 1 mm snapping bound."""
    source_points = {
        key(p): p for s in surfaces for ring in (s.exterior, *s.interiors) for p in ring
    }
    canonical = {}
    # Representatives bucketed by 1 mm cells; only neighbouring cells can be equivalent.
    cells = defaultdict(dict)

    def cell(point):
        return tuple(math.floor(v / VERTEX_EQUIVALENCE_TOLERANCE) for v in point)

    for coordinate, point in sorted(source_points.items()):
        cx, cy, cz = cell(point)
        equivalent = [
            candidate
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            for candidate in cells.get((cx + dx, cy + dy, cz + dz), {}).values()
            if norm(sub(candidate, point)) <= VERTEX_EQUIVALENCE_TOLERANCE
        ]
        representative = (
            min(equivalent, key=lambda p: (norm(sub(p, point)), p)) if equivalent else point
        )
        canonical[coordinate] = representative
        cells[cell(representative)][key(representative)] = representative
    return {
        point: canonical[key(point)]
        for surface in surfaces
        for ring in (surface.exterior, *surface.interiors)
        for point in ring
    }


def node_surfaces(surfaces, *, vertex_mapping=None):
    """Merge within 1 mm and node edges, optionally recording exact representatives."""
    canonical = canonical_vertices(surfaces)
    points = list(dict.fromkeys(canonical.values()))
    coordinates = np.array(points, dtype=float).reshape(-1, 3)
    if vertex_mapping is not None:
        vertex_mapping.update(canonical)
    result = []
    for surface in surfaces:
        rings = []
        for ring in (surface.exterior, *surface.interiors):
            ring = tuple(canonical[point] for point in ring)
            noded = []
            for a, b in zip(ring, (*ring[1:], ring[0])):
                edge = sub(b, a)
                squared = dot(edge, edge)
                if squared < TOLERANCE**2:
                    fail(
                        f"Edge collapsed within 0.001 m equivalence tolerance on {surface.face_id}",
                        "vertex_equivalence_collapsed_edge",
                    )
                nodes = [(0.0, a)]
                low = np.minimum(a, b) - 1e-8
                high = np.maximum(a, b) + 1e-8
                near = np.flatnonzero(np.all((coordinates >= low) & (coordinates <= high), axis=1))
                for index in near:
                    point = points[index]
                    t = dot(sub(point, a), edge) / squared
                    if 1e-9 < t < 1 - 1e-9 and norm(sub(point, add(a, mul(edge, t)))) < 1e-8:
                        nodes.append((t, point))
                noded.extend(point for _, point in sorted(nodes))
            rings.append(tuple(noded))
        result.append(replace(surface, exterior=rings[0], interiors=tuple(rings[1:])))
    return tuple(result)


def _line_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type in {"LineString", "Point"}:
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [part for child in geometry.geoms for part in _line_parts(child)]
    return []


def _face_intersection(a, b, *, separate=False):
    if _bounds_disjoint(a.exterior, b.exterior):
        return
    pa, pb = face_plane(a), face_plane(b)
    direction = cross(pa[3], pb[3])
    length = norm(direction)
    if length < 1e-9:
        if abs(dot(sub(pb[0], pa[0]), pa[3])) > TOLERANCE:
            return
        other = Polygon(
            [project(p, pa) for p in b.exterior],
            [[project(p, pa) for p in r] for r in b.interiors],
        )
        overlap = pa[4].intersection(other)
        if overlap.area > TOLERANCE**2 and (not separate or dot(pa[3], pb[3]) > 0):
            fail(f"Overlapping coplanar faces {a.face_id} and {b.face_id}")
        return
    ca, cb = dot(pa[3], pa[0]), dot(pb[3], pb[0])
    origin = mul(
        add(mul(cross(pb[3], direction), ca), mul(cross(direction, pa[3]), cb)), 1 / length**2
    )
    direction = mul(direction, 1 / length)
    limit = max(norm(sub(p, origin)) for s in (a, b) for p in s.exterior) + 1
    sections = []
    for plane in (pa, pb):
        line = LineString([project(add(origin, mul(direction, t)), plane) for t in (-limit, limit)])
        section = plane[4].intersection(line)
        intervals = []
        for part in _line_parts(section):
            values = [dot(sub(lift(p, plane), origin), direction) for p in part.coords]
            intervals.append((min(values), max(values)))
        sections.append(_merge_intervals(intervals))
    # Plain 1-D interval overlap: GEOS overlays of mixed point/line collections are unsupported.
    common = [
        (max(a0, b0), min(a1, b1))
        for a0, a1 in sections[0]
        for b0, b1 in sections[1]
        if max(a0, b0) <= min(a1, b1)
    ]
    for low, high in common:
        samples = (low,) if high == low else (low, high, (low + high) / 2)
        for t in samples:
            point = add(origin, mul(direction, t))
            interior = [
                plane[4].boundary.distance(Point(project(point, plane))) > TOLERANCE
                for plane in (pa, pb)
            ]
            if all(interior) if separate else any(interior):
                fail(f"Self/inter-solid intersection: {a.face_id} and {b.face_id}")


def _merge_intervals(intervals):
    merged = []
    for low, high in sorted(intervals):
        if merged and low <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return merged


def _boxes(surfaces):
    """Exterior-ring bounds (min xyz, max xyz) per surface, as used by ``_bounds_disjoint``."""
    if not surfaces:
        return np.zeros((0, 6))
    return np.array(
        [
            [min(p[i] for p in s.exterior) for i in range(3)]
            + [max(p[i] for p in s.exterior) for i in range(3)]
            for s in surfaces
        ],
        dtype=float,
    )


def _overlapping(box, boxes):
    """Indices of ``boxes`` not disjoint from ``box`` within the arithmetic tolerance."""
    return np.flatnonzero(
        np.all(
            (boxes[:, 3:] >= box[:3] - TOLERANCE) & (box[3:] >= boxes[:, :3] - TOLERANCE), axis=1
        )
    )


class _SolidIndex:
    """Precomputed boundary pieces and triangles for repeated containment queries."""

    def __init__(self, surfaces):
        self.pieces = tuple(piece for surface in surfaces for piece in planar_pieces(surface))
        self.planes = [face_plane(piece) for piece in self.pieces]
        self.origins = np.array([plane[0] for plane in self.planes], dtype=float)
        self.normals = np.array([plane[3] for plane in self.planes], dtype=float)
        triangles = [
            triangle for piece in self.pieces for triangle in face_triangles(piece)
        ]
        self.triangles = np.array(triangles, dtype=float).reshape(-1, 3, 3)
        self.box = _boxes(self.pieces)
        self.box = np.concatenate([self.box[:, :3].min(axis=0), self.box[:, 3:].max(axis=0)])

    def contains(self, point):
        """Return None on the boundary, otherwise strict interior membership."""
        p = np.array(point, dtype=float)
        if np.any(p < self.box[:3] - TOLERANCE) or np.any(p > self.box[3:] + TOLERANCE):
            return False
        distances = np.abs(np.einsum("ij,ij->i", p - self.origins, self.normals))
        for index in np.flatnonzero(distances <= TOLERANCE):
            plane = self.planes[index]
            if plane[4].distance(Point(project(point, plane))) <= TOLERANCE:
                return None
        a, b, c = self.triangles[:, 0], self.triangles[:, 1], self.triangles[:, 2]
        e1, e2 = b - a, c - a
        s = p - a
        q = np.cross(s, e1)
        for direction in ((1.0, 0.371, 0.529), (0.219, 1.0, 0.713), (0.413, 0.293, 1.0)):
            d = np.array(direction)
            h = np.cross(d, e2)
            determinant = np.einsum("ij,ij->i", e1, h)
            valid = np.abs(determinant) >= 1e-12
            safe = np.where(valid, determinant, 1.0)
            u = np.einsum("ij,ij->i", s, h) / safe
            v = (q @ d) / safe
            t = np.einsum("ij,ij->i", e2, q) / safe
            hit = valid & (t > TOLERANCE) & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
            if np.any(np.minimum(np.minimum(u, v), 1 - u - v)[hit] < 1e-9):
                continue
            return bool(np.count_nonzero(hit) % 2 == 1)
        fail("Ambiguous containment test", "unsupported_geometry")


def _max_adjustment(before, after):
    """Largest distance from an input vertex to its nearest output vertex."""

    def array(surfaces):
        points = {p for s in surfaces for ring in (s.exterior, *s.interiors) for p in ring}
        return np.array(sorted(points), dtype=float)

    source, target = array(before), array(after)
    result = 0.0
    for start in range(0, len(source), 256):
        chunk = source[start : start + 256]
        distances = np.sqrt(((chunk[:, None, :] - target[None, :, :]) ** 2).sum(axis=2))
        result = max(result, float(distances.min(axis=1).max()))
    return result


def _validate_solid(solid):
    if solid.interior_shells:
        fail(
            "Enclosed interior shells require unsupported containment validation",
            "unsupported_geometry",
        )
    if len(solid.surfaces) < 4:
        fail(f"Too few faces in {solid.solid_id}")
    if len({s.face_id for s in solid.surfaces}) != len(solid.surfaces):
        fail("Duplicate face identifiers")
    for surface in solid.surfaces:
        face_plane(surface)
    surfaces = node_surfaces(solid.surfaces)
    deviations = [plane_deviation(surface) for surface in surfaces]
    check_surfaces = tuple(piece for surface in surfaces for piece in planar_pieces(surface))
    edges = defaultdict(list)
    incident = defaultdict(set)
    for i, surface in enumerate(surfaces):
        for ring in (surface.exterior, *surface.interiors):
            for a, b in zip(ring, (*ring[1:], ring[0])):
                a, b = key(a), key(b)
                edges[tuple(sorted((a, b)))].append((i, a, b))
                incident[a].add(i)
    neighbours = defaultdict(set)
    vertex_neighbours = defaultdict(lambda: defaultdict(set))
    for edge, uses in edges.items():
        if len(uses) != 2 or uses[0][1:] != uses[1][1:][::-1]:
            fail(
                f"Unpaired/non-manifold or inconsistently oriented edge in {solid.solid_id}: {edge}"
            )
        a, b = uses[0][0], uses[1][0]
        neighbours[a].add(b)
        neighbours[b].add(a)
        for vertex in edge:
            vertex_neighbours[vertex][a].add(b)
            vertex_neighbours[vertex][b].add(a)

    def connected(items, adjacency):
        visited, queue = set(), [next(iter(items))]
        while queue:
            item = queue.pop()
            if item not in visited:
                visited.add(item)
                queue.extend(adjacency[item] - visited)
        return visited == set(items)

    if not connected(range(len(surfaces)), neighbours):
        fail("A Solid exterior shell must be connected")
    for vertex, faces in incident.items():
        if not connected(faces, vertex_neighbours[vertex]):
            fail(f"Non-manifold vertex {vertex}")
    boxes = _boxes(check_surfaces)
    for i, a in enumerate(check_surfaces):
        for j in _overlapping(boxes[i], boxes[i + 1 :]):
            _face_intersection(a, check_surfaces[i + 1 + j])
    origin = surfaces[0].exterior[0]
    volume = 0.0
    for surface in check_surfaces:
        for ring in (surface.exterior, *surface.interiors):
            a = sub(ring[0], origin)
            for i in range(1, len(ring) - 1):
                volume += dot(a, cross(sub(ring[i], origin), sub(ring[i + 1], origin))) / 6
    if not math.isfinite(volume) or volume <= TOLERANCE**3:
        fail(f"Non-positive volume/outward orientation in {solid.solid_id}")
    return replace(
        solid,
        surfaces=surfaces,
        volume=volume,
        validation={
            "status": "passed",
            "closed": True,
            "manifold": True,
            "planar": True,
            "self_intersections": False,
            "noded_edges": len(edges),
            "volume_m3": volume,
            "tolerance_m": PLANARITY_TOLERANCE,
            "arithmetic_tolerance_m": TOLERANCE,
            "planarity_tolerance_m": PLANARITY_TOLERANCE,
            "vertex_equivalence_tolerance_m": VERTEX_EQUIVALENCE_TOLERANCE,
            "max_face_deviation_m": max(deviations),
            "quantized_faces_triangulated_for_checks": sum(d > TOLERANCE for d in deviations),
            "max_vertex_adjustment_m": _max_adjustment(solid.surfaces, surfaces),
            "fidelity_status": "unverified",
            "independent_validation": "not_run",
        },
    )


def validate_scene(scene: GeometryScene) -> GeometryScene:
    """Validate noded topology, winding, holes, intersections and positive volume."""
    if not scene.solids or len({s.solid_id for s in scene.solids}) != len(scene.solids):
        fail("Empty scene or duplicate solid identifiers")
    solids = tuple(_validate_solid(solid) for solid in scene.solids)
    indexes = {}

    def index(i):
        if i not in indexes:
            surfaces = solids[i].surfaces
            pieces = tuple(p for s in surfaces for p in planar_pieces(s))
            indexes[i] = (pieces, _boxes(pieces), _SolidIndex(surfaces))
        return indexes[i]

    for i, a in enumerate(solids):
        for j in range(i + 1, len(solids)):
            b = solids[j]
            if _bounds_disjoint(
                [p for s in a.surfaces for p in s.exterior],
                [p for s in b.surfaces for p in s.exterior],
            ):
                continue
            a_surfaces, a_boxes, a_index = index(i)
            b_surfaces, b_boxes, b_index = index(j)
            for k, fa in enumerate(a_surfaces):
                for m in _overlapping(a_boxes[k], b_boxes):
                    _face_intersection(fa, b_surfaces[m], separate=True)
            for first, second in ((a_surfaces, b_index), (b_surfaces, a_index)):
                for surface in first:
                    plane = face_plane(surface)
                    samples = (
                        *surface.exterior,
                        lift(plane[4].representative_point().coords[0], plane),
                    )
                    if any(second.contains(point) is True for point in samples):
                        fail(f"Overlapping solid interiors: {a.solid_id}, {b.solid_id}")
    return replace(scene, solids=solids)
