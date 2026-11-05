"""Thermal partitions of reconstructed solids, without roof approximation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace

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
    fail,
    lift,
    node_surfaces,
    norm,
    planar_pieces,
    polygons,
    project,
    sub,
    surface_from_polygon,
    validate_scene,
)


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
                        shared_a, shared_b = [], []
                        for k, polygon in enumerate(polygons(overlap)):
                            sa = surface_from_polygon(a, polygon, pa, f":interface:{interface}:{k}")
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
                            surface_from_polygon(a, p, pa, f":remainder:{interface}:{k}")
                            for k, p in enumerate(polygons(pa[4].difference(overlap)))
                            if p.area > TOLERANCE**2
                        ]
                        remainder_b = [
                            _reverse(surface_from_polygon(b, p, pa, f":remainder:{interface}:{k}"))
                            for k, p in enumerate(polygons(other.difference(overlap)))
                            if p.area > TOLERANCE**2
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


def _assemble_thermal_scene(source, children_by_solid, report):
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
        thermal = _assemble_thermal_scene(source, children_by_solid, report)
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
        thermal = _assemble_thermal_scene(source, children_by_solid, report)
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
