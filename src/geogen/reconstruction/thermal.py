"""Thermal partitions of reconstructed solids, without roof approximation."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
from dataclasses import asdict, replace

from shapely import make_valid
from shapely.errors import GEOSException
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, split, unary_union

from .types import (
    GeometryScene,
    ReconstructionError,
    Solid,
    Surface,
    ThermalScene,
    ThermalSpace,
)
from .validate import (
    TOLERANCE,
    _line_parts,
    _SolidIndex,
    dot,
    face_plane,
    face_triangles,
    fail,
    lift,
    node_surfaces,
    norm,
    planar_pieces,
    plane_deviation,
    polygons,
    project,
    sub,
    surface_from_polygon,
    validate_scene,
)


# OpenStudio's per-axis point equivalence (isAlmostEqual3dPt) used to match edges.
PARTY_WALL_TOLERANCE = 0.0127


def thermal_geometry_hash(frame, spaces):
    """Hash the canonical thermal geometry shared by both exporters."""

    def ring_key(ring):
        ring = tuple(tuple(round(v, 8) for v in point) for point in ring)
        return min(ring[i:] + ring[:i] for i in range(len(ring)))

    data = {
        "frame": asdict(frame),
        "spaces": [
            {
                "id": space.space_id,
                "solid_id": space.solid_id,
                "surfaces": sorted(
                    [
                        {
                            "id": s.face_id,
                            "outer": ring_key(s.exterior),
                            "holes": sorted(ring_key(r) for r in s.interiors),
                            "semantic": s.semantic,
                            "boundary": s.boundary,
                            "adjacent": s.adjacent_space_id,
                            "interface": s.interface_id,
                        }
                        for s in space.surfaces
                    ],
                    key=lambda s: s["id"],
                ),
            }
            for space in sorted(spaces, key=lambda s: s.space_id)
        ],
    }
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reverse(surface, **changes):
    return replace(
        surface,
        exterior=surface.exterior[::-1],
        interiors=tuple(r[::-1] for r in surface.interiors),
        **changes,
    )


def _face_section(surface, z):
    plane = face_plane(surface)
    gradient = (plane[1][2], plane[2][2])
    length = math.hypot(*gradient)
    if length < 1e-10:
        return None
    offset = z - plane[0][2]
    center = (gradient[0] * offset / length**2, gradient[1] * offset / length**2)
    direction = (-gradient[1] / length, gradient[0] / length)
    limit = max(math.dist(p[:2], center) for p in plane[4].exterior.coords) + 1
    line = LineString(
        [
            (center[0] + sign * limit * direction[0], center[1] + sign * limit * direction[1])
            for sign in (-1, 1)
        ]
    )
    return plane, line


def _clip_surface(surface, z, above):
    elevations = [p[2] for ring in (surface.exterior, *surface.interiors) for p in ring]
    if min(elevations) >= z - TOLERANCE if above else max(elevations) <= z + TOLERANCE:
        return [surface]
    if max(elevations) < z - TOLERANCE if above else min(elevations) > z + TOLERANCE:
        return []
    section = _face_section(surface, z)
    if section is None:
        return (
            [surface]
            if (
                surface.exterior[0][2] >= z - TOLERANCE
                if above
                else surface.exterior[0][2] <= z + TOLERANCE
            )
            else []
        )
    plane, line = section
    return [
        surface_from_polygon(surface, polygon, plane, f":{'above' if above else 'below'}:{z:g}:{i}")
        for i, polygon in enumerate(polygons(split(plane[4], line)))
        if (
            lift(polygon.representative_point().coords[0], plane)[2] >= z - TOLERANCE
            if above
            else lift(polygon.representative_point().coords[0], plane)[2] <= z + TOLERANCE
        )
        and polygon.area > TOLERANCE**2
    ]


def _cap(solid, z, containment):
    lines = []
    for surface in solid.surfaces:
        section = _face_section(surface, z)
        if section is None:
            if abs(surface.exterior[0][2] - z) <= TOLERANCE:
                fail(
                    "Storey cuts coincident with horizontal source faces are unsupported",
                    "unsupported_zoning",
                )
            continue
        plane, line = section
        for part in _line_parts(plane[4].intersection(line)):
            if part.geom_type == "LineString" and part.length > TOLERANCE:
                lines.append(
                    LineString(
                        [tuple(round(v, 8) for v in lift(p, plane)[:2]) for p in part.coords]
                    )
                )
    candidates = [
        polygon
        for polygon in polygonize(unary_union(lines))
        if containment.contains((*polygon.representative_point().coords[0], z)) is True
    ]
    section = unary_union(candidates)
    if section.is_empty or section.geom_type != "Polygon":
        fail("A storey cut must have one connected polygonal section", "unsupported_zoning")
    plane = ((0.0, 0.0, z), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    template = Surface(
        f"{solid.solid_id}:cut:{z:.17g}",
        (),
        semantic="RoofSurface",
        boundary="surface",
        provenance=("horizontal thermal partition",),
    )
    return surface_from_polygon(template, section, plane, "")


def _cuts(solid, levels):
    if not isinstance(levels, (list, tuple)) or any(
        isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(z)
        for z in levels
    ):
        fail("Storey cuts must be a list of finite local elevations", "unsupported_zoning")
    low = min(p[2] for s in solid.surfaces for p in s.exterior)
    high = max(p[2] for s in solid.surfaces for p in s.exterior)
    cuts = list(levels)
    if cuts != sorted(set(cuts)) or any(not low + TOLERANCE < z < high - TOLERANCE for z in cuts):
        fail("Storey cuts must be strictly increasing and inside the solid", "unsupported_zoning")
    return cuts


def _partition(solid, cuts):
    containment = _SolidIndex(solid.surfaces)
    caps = [_cap(solid, z, containment) for z in cuts]
    result = []
    for index in range(len(cuts) + 1):
        surfaces = list(solid.surfaces)
        if index:
            surfaces = [
                part
                for surface in surfaces
                for part in _clip_surface(surface, cuts[index - 1], True)
            ]
            surfaces.append(_reverse(caps[index - 1], semantic="GroundSurface"))
        if index < len(cuts):
            surfaces = [
                part for surface in surfaces for part in _clip_surface(surface, cuts[index], False)
            ]
            surfaces.append(caps[index])
        surfaces = tuple(replace(s, face_id=f"{s.face_id}:storey:{index + 1}") for s in surfaces)
        result.append(
            Solid(f"{solid.solid_id}:storey:{index + 1}", solid.reconstruction_id, surfaces)
        )
    return result


def _backtracks(p, a, b):
    """Whether ``p`` lies on segment ``a``-``b`` within the edge-noding tolerance."""
    ab = (b[0] - a[0], b[1] - a[1])
    squared = ab[0] ** 2 + ab[1] ** 2
    if squared == 0:
        return math.dist(p, a) < 1e-8
    t = ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / squared
    closest = (a[0] + ab[0] * min(1, max(0, t)), a[1] + ab[1] * min(1, max(0, t)))
    return math.dist(p, closest) < 1e-8


def _without_spikes(polygon):
    """Drop zero-width spikes left by overlaying near-collinear party-wall edges.

    A vertex whose ring doubles back onto the previous or next edge encloses no
    area beyond arithmetic noise, but edge noding would revisit its neighbour.
    """

    def clean(coords):
        ring = list(coords)[:-1]
        changed = True
        while changed and len(ring) >= 3:
            changed = False
            for i, v in enumerate(ring):
                u, w = ring[i - 1], ring[(i + 1) % len(ring)]
                if _backtracks(u, v, w) or _backtracks(w, u, v):
                    del ring[i]
                    changed = True
                    break
        return ring

    exterior = clean(polygon.exterior.coords)
    interiors = [clean(r.coords) for r in polygon.interiors]
    if len(exterior) < 3:
        return None
    cleaned = Polygon(exterior, [r for r in interiors if len(r) >= 3])
    if not cleaned.is_valid or abs(cleaned.area - polygon.area) > TOLERANCE:
        return polygon
    return cleaned


def _noded(polygon):
    """Insert ring vertices lying on other edges, as ``node_surfaces`` later would."""
    rings = [list(polygon.exterior.coords)[:-1], *(list(r.coords)[:-1] for r in polygon.interiors)]
    vertices = [p for ring in rings for p in ring]
    result = []
    for ring in rings:
        noded = []
        for a, b in zip(ring, (*ring[1:], ring[0])):
            noded.append(a)
            ab = (b[0] - a[0], b[1] - a[1])
            squared = ab[0] ** 2 + ab[1] ** 2
            hits = {
                ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / squared: p
                for p in vertices
                if p != a and p != b and squared > 0 and _backtracks(p, a, b)
            }
            noded.extend(hits[t] for t in sorted(hits) if 0 < t < 1)
        result.append(noded)
    return Polygon(result[0], result[1:])


def _split_polygons(geometry):
    """Polygons of an overlay result, split where a ring pinches onto another edge.

    Overlaying party walls whose roof lines partly coincide can leave one ring
    joined by a zero-width corridor; edge noding would then repeat vertices.
    """
    result = []
    for polygon in polygons(geometry):
        noded = _noded(polygon)
        if noded.is_valid:
            pieces = [polygon]
        else:
            pieces = polygons(make_valid(noded, method="structure", keep_collapsed=False))
            if abs(sum(p.area for p in pieces) - polygon.area) > TOLERANCE:
                pieces = [polygon]
        result.extend(_without_spikes(p) for p in pieces)
    return [p for p in result if p is not None and p.area > TOLERANCE**2]


def _restored(surface, originals):
    return replace(
        surface,
        exterior=tuple(originals.get(p, p) for p in surface.exterior),
        interiors=tuple(tuple(originals.get(p, p) for p in r) for r in surface.interiors),
    )


def _match_boundaries(spaces):
    """Split partial coplanar interfaces; preserve every exterior remainder."""
    surfaces = [list(space.surfaces) for space in spaces]
    for i in range(len(spaces)):
        for j in range(i + 1, len(spaces)):
            changed = True
            while changed:
                changed = False
                for ai, a in enumerate(surfaces[i]):
                    if a.adjacent_space_id:
                        continue
                    pa = face_plane(a)
                    for bi, b in enumerate(surfaces[j]):
                        if b.adjacent_space_id:
                            continue
                        pb = face_plane(b)
                        if (
                            dot(pa[3], pb[3]) > -1 + 1e-9
                            or abs(dot(sub(pb[0], pa[0]), pa[3])) > TOLERANCE
                        ):
                            continue
                        other = Polygon(
                            [project(p, pa) for p in b.exterior],
                            [[project(p, pa) for p in r] for r in b.interiors],
                        )
                        overlap = pa[4].intersection(other)
                        if overlap.area <= TOLERANCE**2:
                            continue
                        interface = hashlib.sha256(
                            f"{spaces[i].space_id}|{spaces[j].space_id}|{a.face_id}|{b.face_id}".encode()
                        ).hexdigest()[:20]
                        # Projecting into a's plane moves vertices that are only planar within
                        # tolerance; unchanged corners keep their exact source coordinates.
                        originals = {
                            lift(project(p, pa), pa): p
                            for s in (a, b)
                            for ring in (s.exterior, *s.interiors)
                            for p in ring
                        }

                        def lifted(surface, polygon, suffix):
                            return _restored(
                                surface_from_polygon(surface, polygon, pa, suffix), originals
                            )

                        shared_a, shared_b = [], []
                        for k, polygon in enumerate(_split_polygons(overlap)):
                            sa = lifted(a, polygon, f":interface:{interface}:{k}")
                            sb = _reverse(
                                sa,
                                face_id=f"{b.face_id}:interface:{interface}:{k}",
                                semantic=b.semantic,
                                source_face_id=b.source_face_id,
                                source_object_id=b.source_object_id,
                                provenance=b.provenance,
                            )
                            common = f"{interface}:{k}"
                            shared_a.append(
                                replace(
                                    sa,
                                    boundary="surface",
                                    adjacent_space_id=spaces[j].space_id,
                                    interface_id=common,
                                )
                            )
                            shared_b.append(
                                replace(
                                    sb,
                                    boundary="surface",
                                    adjacent_space_id=spaces[i].space_id,
                                    interface_id=common,
                                )
                            )
                        remainder_a = [
                            lifted(a, p, f":remainder:{interface}:{k}")
                            for k, p in enumerate(_split_polygons(pa[4].difference(overlap)))
                        ]
                        remainder_b = [
                            _reverse(lifted(b, p, f":remainder:{interface}:{k}"))
                            for k, p in enumerate(_split_polygons(other.difference(overlap)))
                        ]
                        surfaces[i][ai : ai + 1] = shared_a + remainder_a
                        surfaces[j][bi : bi + 1] = shared_b + remainder_b
                        changed = True
                        break
                    if changed:
                        break
    for index, items in enumerate(surfaces):
        for surface in items:
            if surface.boundary == "surface" and not surface.adjacent_space_id:
                fail(f"Unmatched thermal interface {surface.face_id}")
        surfaces[index] = node_surfaces(items)
    return tuple(replace(space, surfaces=tuple(items)) for space, items in zip(spaces, surfaces))


def _space(solid, origin, storey_number=None, nominal_z=None):
    surfaces = tuple(
        replace(
            surface,
            boundary=(
                surface.boundary
                if surface.boundary == "surface"
                else ("ground" if surface.semantic == "GroundSurface" else "outdoors")
            ),
        )
        for surface in solid.surfaces
    )
    return ThermalSpace(
        solid.solid_id,
        origin,
        surfaces,
        solid.volume,
        storey_number=storey_number,
        nominal_z=nominal_z,
    )


def _checked_spaces(frame, spaces):
    checked = validate_scene(
        GeometryScene(
            frame, tuple(Solid(space.space_id, space.solid_id, space.surfaces) for space in spaces)
        )
    )
    return tuple(
        replace(space, surfaces=solid.surfaces, volume=solid.volume)
        for space, solid in zip(spaces, checked.solids)
    )


def _check_conservation(original, children):
    if abs(sum(space.volume for space in children) - original.volume) > max(
        TOLERANCE, original.volume * 1e-8
    ):
        fail("Thermal partition did not conserve source volume")
    for surface in original.surfaces:
        plane = face_plane(surface)
        fragments = [
            Polygon(
                [project(p, plane) for p in s.exterior],
                [[project(p, plane) for p in r] for r in s.interiors],
            )
            for space in children
            for s in space.surfaces
            if s.source_face_id == surface.source_face_id
        ]
        if (
            plane[4].symmetric_difference(unary_union(fragments)).area > TOLERANCE
            or abs(sum(p.area for p in fragments) - plane[4].area) > TOLERANCE
        ):
            fail(f"Thermal partition changed source face {surface.face_id}")


def _unify_scene_vertices(scene):
    """Merge near-coincident vertices of different solids before interface matching.

    Adjacent solids are reconstructed independently, so a shared party-wall corner
    may differ by a few millimetres. Interface remainders would then be slivers that
    OpenStudio's 12.7 mm point equivalence cannot enclose. Vertices of different
    solids within that bound are moved to one existing representative, unless a
    solid owns two of them. Faces that would leave the planarity bound are first
    triangulated on their source vertices. Returns the scene and the largest move.
    """
    owners = defaultdict(set)
    for index, solid in enumerate(scene.solids):
        for surface in solid.surfaces:
            for ring in (surface.exterior, *surface.interiors):
                for point in ring:
                    owners[point].add(index)
    points = sorted(owners)
    parent = list(range(len(points)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def cell(point):
        return tuple(math.floor(v / PARTY_WALL_TOLERANCE) for v in point)

    cells = defaultdict(list)
    for i, point in enumerate(points):
        cells[cell(point)].append(i)
    for i, point in enumerate(points):
        cx, cy, cz = cell(point)
        for dx, dy, dz in itertools.product((-1, 0, 1), repeat=3):
            for j in cells.get((cx + dx, cy + dy, cz + dz), ()):
                if (
                    j > i
                    and owners[point].isdisjoint(owners[points[j]])
                    and max(abs(a - b) for a, b in zip(point, points[j])) < PARTY_WALL_TOLERANCE
                ):
                    parent[root(j)] = root(i)
    clusters = defaultdict(list)
    for i in range(len(points)):
        clusters[root(i)].append(points[i])
    mapping = {}
    for cluster in clusters.values():
        if len(cluster) < 2:
            continue
        solids = [owner for point in cluster for owner in owners[point]]
        if len(solids) != len(set(solids)):
            continue
        representative = min(cluster)
        mapping.update({point: representative for point in cluster if point != representative})
    if not mapping:
        return scene, 0.0

    def ring(points):
        return tuple(mapping.get(p, p) for p in points)

    def unified(surface):
        if not any(p in mapping for r in (surface.exterior, *surface.interiors) for p in r):
            return (surface,)
        moved = replace(
            surface, exterior=ring(surface.exterior), interiors=tuple(map(ring, surface.interiors))
        )
        try:
            if plane_deviation(moved) <= TOLERANCE:
                return (moved,)
        except ReconstructionError:
            pass
        pieces = []
        for k, triangle in enumerate(face_triangles(surface)):
            face_id = f"{surface.face_id}:unified:{k}"
            pieces.append(
                replace(
                    surface,
                    face_id=face_id,
                    source_face_id=face_id,
                    exterior=ring(triangle),
                    interiors=(),
                    provenance=(*surface.provenance, "party_wall_vertex_unification:triangulated"),
                )
            )
        return tuple(pieces)

    return (
        replace(
            scene,
            solids=tuple(
                replace(
                    solid,
                    surfaces=tuple(piece for s in solid.surfaces for piece in unified(s)),
                )
                for solid in scene.solids
            ),
        ),
        max(math.dist(p, q) for p, q in mapping.items()),
    )


def _rings(surface):
    return (surface.exterior, *surface.interiors)


def _edge_position(point, a, b):
    ab = sub(b, a)
    squared = dot(ab, ab)
    if squared <= TOLERANCE**2:
        return None, math.inf
    t = dot(sub(point, a), ab) / squared
    return t, norm(sub(point, tuple(a[i] + t * ab[i] for i in range(3))))


def _insert_junctions(scene):
    """Insert vertices lying near another solid's edge into that edge.

    Where a neighbour's roof line passes within OpenStudio's point equivalence of
    a corner without meeting one of its vertices, interface matching would create
    a crossing vertex next to that corner and a sliver OpenStudio cannot enclose.
    Faces that would leave the planarity bound are triangulated on their source
    vertices before the inserted vertex splits a boundary triangle. Returns the
    scene and the largest distance between an inserted vertex and its edge.
    """
    from shapely import STRtree, box, points as shapely_points

    owners = defaultdict(set)
    edges = {}
    for index, solid in enumerate(scene.solids):
        for surface in solid.surfaces:
            for ring in _rings(surface):
                for a, b in zip(ring, (*ring[1:], ring[0])):
                    owners[a].add(index)
                    edges[(index, min(a, b), max(a, b))] = None
    if len(scene.solids) < 2 or not edges:
        return scene, 0.0
    edges = list(edges)
    cells = defaultdict(list)

    def cell(point):
        return tuple(math.floor(v / PARTY_WALL_TOLERANCE) for v in point)

    for point in owners:
        cells[cell(point)].append(point)

    def near_vertex(point, index):
        cx, cy, cz = cell(point)
        return any(
            index in owners[other]
            and max(abs(p - q) for p, q in zip(point, other)) < PARTY_WALL_TOLERANCE
            for dx, dy, dz in itertools.product((-1, 0, 1), repeat=3)
            for other in cells.get((cx + dx, cy + dy, cz + dz), ())
        )

    tree = STRtree(
        [
            box(
                min(a[0], b[0]) - PARTY_WALL_TOLERANCE,
                min(a[1], b[1]) - PARTY_WALL_TOLERANCE,
                max(a[0], b[0]) + PARTY_WALL_TOLERANCE,
                max(a[1], b[1]) + PARTY_WALL_TOLERANCE,
            )
            for _, a, b in edges
        ]
    )
    vertices = sorted(owners)
    inserts = defaultdict(lambda: defaultdict(set))
    moved = 0.0
    for i, j in zip(*tree.query(shapely_points([p[:2] for p in vertices]), predicate="intersects")):
        point, (index, a, b) = vertices[i], edges[j]
        if index in owners[point]:
            continue
        t, distance = _edge_position(point, a, b)
        if t is None or not 0 < t < 1 or not 1e-8 < distance < PARTY_WALL_TOLERANCE:
            continue
        if near_vertex(point, index):
            continue
        inserts[index][(a, b)].add(point)
        moved = max(moved, distance)
    if not inserts:
        return scene, 0.0

    def on_edge(solid_inserts, a, b):
        points = solid_inserts.get((min(a, b), max(a, b)), ())
        return sorted(points, key=lambda p: _edge_position(p, a, b)[0])

    def junctioned(surface, solid_inserts):
        def ring(points):
            result = []
            for a, b in zip(points, (*points[1:], points[0])):
                result.append(a)
                result.extend(on_edge(solid_inserts, a, b))
            return tuple(result)

        rings = tuple(ring(r) for r in _rings(surface))
        if rings == _rings(surface):
            return (surface,)
        candidate = replace(surface, exterior=rings[0], interiors=rings[1:])
        try:
            if plane_deviation(candidate) <= TOLERANCE:
                return (candidate,)
        except ReconstructionError:
            pass
        triangles = [tuple(t) for t in face_triangles(surface)]
        for edge, points in solid_inserts.items():
            for k, triangle in enumerate(triangles):
                for m in range(3):
                    a, b, c = triangle[m], triangle[(m + 1) % 3], triangle[(m + 2) % 3]
                    if (min(a, b), max(a, b)) == edge:
                        chain = (a, *on_edge(solid_inserts, a, b), b)
                        triangles[k : k + 1] = [(p, q, c) for p, q in zip(chain, chain[1:])]
                        break
                else:
                    continue
                break
        return tuple(
            replace(
                surface,
                face_id=f"{surface.face_id}:junction:{k}",
                source_face_id=f"{surface.face_id}:junction:{k}",
                exterior=triangle,
                interiors=(),
                provenance=(*surface.provenance, "party_wall_junction:triangulated"),
            )
            for k, triangle in enumerate(triangles)
        )

    return (
        replace(
            scene,
            solids=tuple(
                replace(
                    solid,
                    surfaces=tuple(
                        piece for s in solid.surfaces for piece in junctioned(s, inserts[index])
                    ),
                )
                if index in inserts
                else solid
                for index, solid in enumerate(scene.solids)
            ),
        ),
        moved,
    )


def _assemble_thermal_scene(source, children_by_solid, report, unified_m=0.0):
    spaces = tuple(space for children in children_by_solid.values() for space in children)
    spaces = _checked_spaces(source.frame, spaces)
    spaces = _match_boundaries(spaces)
    spaces = _checked_spaces(source.frame, spaces)
    for original in source.solids:
        children = [space for space in spaces if space.solid_id == original.solid_id]
        _check_conservation(original, children)
    assumptions = (
        "experimental: no independently reviewed fidelity benchmark",
        (
            "horizontal storey partitions; exterior roof unchanged"
            if any(entry["status"] == "partitioned" for entry in report.values())
            else "single-volume zoning"
        ),
        "conditioning is unknown; no inferred attic exclusion",
        "GroundSurface treated as ground-contact; other unpaired faces outdoors",
        "faces warped within 0.001 m are triangulated without relocating their vertices",
    )
    if unified_m:
        assumptions += (
            f"vertices of adjacent solids within {PARTY_WALL_TOLERANCE:g} m are unified, "
            "or inserted into the nearby edge, before interface matching "
            f"(max movement {unified_m:.3g} m)",
        )
    return validate_thermal_scene(
        ThermalScene(source.frame, spaces, thermal_geometry_hash(source.frame, spaces), assumptions)
    )


def make_thermal_scene(
    scene: GeometryScene,
    *,
    levels: dict[str, list[float]] | None = None,
    partition_report: dict | None = None,
) -> ThermalScene:
    """Conserve the exterior, optionally cutting each solid into horizontal storeys.

    ``levels`` maps exact solid IDs to strictly increasing interior cut elevations
    in local metres. An empty list marks one storey; omitted solids retain the
    single-volume API. With ``partition_report``, an unsupported partition falls
    back atomically to the original solid. A combined-scene failure restores all
    successfully split solids before a strict retry. Source and final validation
    remain strict, and each solid's outcome is recorded under its ID in the report.
    Near-coincident vertices of different solids are unified first, and vertices
    near another solid's edge are inserted into it, so independently
    reconstructed neighbours share party-wall vertices.
    """
    scene = replace(
        scene,
        solids=tuple(
            replace(
                solid,
                surfaces=tuple(
                    replace(s, source_face_id=s.source_face_id or s.face_id) for s in solid.surfaces
                ),
            )
            for solid in scene.solids
        ),
    )
    scene, unified_m = _unify_scene_vertices(scene)
    scene, junction_m = _insert_junctions(scene)
    unified_m = max(unified_m, junction_m)
    source = validate_scene(scene)
    if any(
        s.semantic
        not in {
            "GroundSurface",
            "RoofSurface",
            "WallSurface",
            "OuterFloorSurface",
            "OuterCeilingSurface",
        }
        for solid in source.solids
        for s in solid.surfaces
    ):
        fail("Thermal conversion requires explicit supported surface semantics")
    if levels is not None and (
        not isinstance(levels, dict) or set(levels) - {solid.solid_id for solid in source.solids}
    ):
        fail("Storey levels must be keyed by exact source solid IDs", "unsupported_zoning")
    children_by_solid, single_volumes, report = {}, {}, {}
    for original in source.solids:
        solid = replace(
            original,
            surfaces=tuple(
                piece for surface in original.surfaces for piece in planar_pieces(surface)
            ),
        )
        children = (_space(solid, solid.solid_id),)
        single_volumes[solid.solid_id] = children
        requested = levels is not None and solid.solid_id in levels
        entry = {
            "status": "single_volume",
            "requested_levels_m": levels[solid.solid_id] if requested else None,
        }
        if requested:
            try:
                cuts = _cuts(solid, levels[solid.solid_id])
                parts = _partition(solid, cuts) if cuts else [solid]
                elevations = [
                    min(p[2] for s in solid.surfaces for p in s.exterior),
                    *cuts,
                ]
                candidates = tuple(
                    _space(part, solid.solid_id, i + 1, z)
                    for i, (part, z) in enumerate(zip(parts, elevations))
                )
                candidates = _checked_spaces(scene.frame, candidates)
                candidates = _checked_spaces(scene.frame, _match_boundaries(candidates))
                _check_conservation(original, candidates)
                _validate_storeys(candidates)
                children = candidates
                entry["status"] = "partitioned" if cuts else "single_volume"
            except (ReconstructionError, GEOSException) as exc:
                if partition_report is None:
                    raise
                entry["status"] = "fallback_single_volume"
                entry["failure"] = {
                    "code": (
                        exc.code if isinstance(exc, ReconstructionError) else "geometry_invalid"
                    ),
                    "message": str(exc),
                }
        entry["space_ids"] = [space.space_id for space in children]
        report[solid.solid_id] = entry
        children_by_solid[solid.solid_id] = children
    try:
        thermal = _assemble_thermal_scene(source, children_by_solid, report, unified_m)
    except (ReconstructionError, GEOSException) as exc:
        partitioned = [key for key, entry in report.items() if entry["status"] == "partitioned"]
        if partition_report is None or not partitioned:
            raise
        for solid_id in partitioned:
            children_by_solid[solid_id] = single_volumes[solid_id]
            report[solid_id].update(
                status="fallback_single_volume",
                space_ids=[space.space_id for space in single_volumes[solid_id]],
                failure={
                    "code": (
                        exc.code if isinstance(exc, ReconstructionError) else "geometry_invalid"
                    ),
                    "message": f"Combined thermal partition validation failed: {exc}",
                },
            )
        thermal = _assemble_thermal_scene(source, children_by_solid, report, unified_m)
    if partition_report is not None:
        partition_report.update(report)
    return thermal


def _validate_storeys(spaces):
    groups = {}
    for space in spaces:
        groups.setdefault(space.solid_id, []).append(space)
        if space.storey_number is None and space.nominal_z is None:
            continue
        if (
            isinstance(space.storey_number, bool)
            or not isinstance(space.storey_number, int)
            or space.storey_number < 1
            or isinstance(space.nominal_z, bool)
            or not isinstance(space.nominal_z, (int, float))
            or not math.isfinite(space.nominal_z)
        ):
            fail("Storey metadata requires a positive integer number and finite local elevation")
        low = min(p[2] for s in space.surfaces for p in s.exterior)
        if abs(low - space.nominal_z) > TOLERANCE:
            fail("Nominal storey elevation does not match the space geometry")
    for children in groups.values():
        if all(space.storey_number is None for space in children):
            continue
        if any(space.storey_number is None for space in children):
            fail("Storey metadata must cover every space of a source solid")
        children = sorted(children, key=lambda space: space.storey_number)
        if [space.storey_number for space in children] != list(range(1, len(children) + 1)):
            fail("Storey numbers must be consecutive and unique within each source solid")
        for lower, upper in zip(children, children[1:]):
            high = max(p[2] for s in lower.surfaces for p in s.exterior)
            if (
                upper.nominal_z <= lower.nominal_z + TOLERANCE
                or abs(high - upper.nominal_z) > TOLERANCE
            ):
                fail("Storey elevations must describe consecutive horizontal bands")


def validate_thermal_scene(scene):
    """Reject stale hashes or broken reciprocal boundaries before exporting."""
    if thermal_geometry_hash(scene.frame, scene.spaces) != scene.geometry_hash:
        fail("Thermal geometry hash does not match the supplied geometry")
    checked = validate_scene(
        GeometryScene(
            scene.frame, tuple(Solid(s.space_id, s.solid_id, s.surfaces) for s in scene.spaces)
        )
    )
    face_ids = [surface.face_id for space in scene.spaces for surface in space.surfaces]
    if len(face_ids) != len(set(face_ids)):
        fail("Thermal surface identifiers must be globally unique")
    _validate_storeys(scene.spaces)
    by_interface = {}
    for space, solid in zip(scene.spaces, checked.solids):
        if not math.isclose(space.volume, solid.volume, rel_tol=1e-8, abs_tol=TOLERANCE):
            fail("Thermal volume does not match the supplied geometry")
        for surface in space.surfaces:
            if surface.boundary not in {"surface", "outdoors", "ground", "adiabatic"}:
                fail(f"Unresolved thermal boundary {surface.face_id}")
            if surface.boundary == "surface":
                if not surface.interface_id or not surface.adjacent_space_id:
                    fail("Missing reciprocal interface identifiers")
                by_interface.setdefault(surface.interface_id, []).append((space, surface))
            elif surface.interface_id or surface.adjacent_space_id:
                fail("Non-interface surface has an adjacent space")
    for pairs in by_interface.values():
        if len(pairs) != 2:
            fail("Thermal interfaces must have exactly two sides")
        (a, fa), (b, fb) = pairs
        if fa.adjacent_space_id != b.space_id or fb.adjacent_space_id != a.space_id:
            fail("Non-reciprocal thermal adjacency")
        pa, pb = face_plane(fa), face_plane(fb)
        other = Polygon(
            [project(p, pa) for p in fb.exterior],
            [[project(p, pa) for p in r] for r in fb.interiors],
        )
        if (
            dot(pa[3], pb[3]) > -1 + 1e-9
            or norm(sub(pa[0], lift(project(pa[0], pb), pb))) > TOLERANCE
            or pa[4].symmetric_difference(other).area > TOLERANCE
        ):
            fail("Reciprocal thermal faces do not coincide")
    return scene
