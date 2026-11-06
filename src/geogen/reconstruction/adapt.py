"""Bounded, auditable geometry adaptation for valid closed envelopes."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

from pyproj import CRS

from .types import GeometryScene, ReconstructionError
from .validate import (
    TOLERANCE,
    VERTEX_EQUIVALENCE_TOLERANCE,
    canonical_vertices,
    face_plane,
    fail,
    node_surfaces,
    plane_deviation,
    surface_pieces,
    validate_scene,
)


def _rings(surface):
    return (surface.exterior, *surface.interiors)


def _points(surfaces):
    return {point for surface in surfaces for ring in _rings(surface) for point in ring}


def scene_geometry_hash(scene: GeometryScene) -> str:
    """Hash exact metric geometry and identities, excluding validation metadata."""

    def ring_key(ring):
        points = tuple(tuple(float(v) for v in point) for point in ring)
        return min(points[i:] + points[:i] for i in range(len(points)))

    def faces(surfaces):
        return sorted(
            [
                {
                    "face_id": surface.face_id,
                    "semantic": surface.semantic,
                    "exterior": ring_key(surface.exterior),
                    "interiors": sorted(ring_key(ring) for ring in surface.interiors),
                }
                for surface in surfaces
            ],
            key=lambda item: item["face_id"],
        )

    data = {
        "frame": asdict(scene.frame),
        "solids": [
            {
                "solid_id": solid.solid_id,
                "reconstruction_id": solid.reconstruction_id,
                "surfaces": faces(solid.surfaces),
                "interior_shells": [faces(shell) for shell in solid.interior_shells],
            }
            for solid in sorted(scene.solids, key=lambda item: item.solid_id)
        ],
    }
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _occurrences(surfaces):
    result = defaultdict(list)
    for surface in surfaces:
        for ring_index, ring in enumerate(_rings(surface)):
            for vertex_index, point in enumerate(ring):
                result[point].append(
                    {
                        "face_id": surface.face_id,
                        "ring_index": ring_index,
                        "vertex_index": vertex_index,
                    }
                )
    return result


def _clean_consecutive_vertices(surfaces):
    """Remove only cyclically adjacent occurrences sharing a bounded representative."""
    for surface in surfaces:
        for ring in _rings(surface):
            if len(ring) < 3 or any(
                len(point) != 3 or any(not math.isfinite(v) for v in point) for point in ring
            ):
                fail(f"Invalid coordinates on {surface.face_id}")
    canonical = canonical_vertices(surfaces)
    if not any(
        canonical[a] == canonical[b]
        for surface in surfaces
        for ring in _rings(surface)
        for a, b in zip(ring, (*ring[1:], ring[0]))
    ):
        return surfaces, {point: point for point in canonical}, []
    cleaned, removed = [], []
    for surface in surfaces:
        rings = []
        for ring_index, ring in enumerate(_rings(surface)):
            retained = []
            for index, point in enumerate(ring):
                if not retained or canonical[point] != canonical[ring[retained[-1]]]:
                    retained.append(index)
            if len(retained) > 1 and canonical[ring[retained[0]]] == canonical[ring[retained[-1]]]:
                retained.pop()
            reduced = tuple(canonical[ring[index]] for index in retained)
            if len(set(reduced)) < 3:
                fail(
                    f"Consecutive-vertex cleanup would collapse a ring on {surface.face_id}",
                    "vertex_cleanup_degenerate_ring",
                )
            # Nonconsecutive duplicates indicate pinched topology, not redundant edges.
            if len(set(reduced)) != len(reduced):
                fail(f"Nonconsecutive repeated vertices on {surface.face_id}")
            for index, point in enumerate(ring):
                if index not in retained:
                    removed.append(
                        {
                            "face_id": surface.face_id,
                            "ring_index": ring_index,
                            "vertex_index": index,
                            "source": list(point),
                            "representative": list(canonical[point]),
                        }
                    )
            rings.append(reduced)
        changed = tuple(rings) != _rings(surface)
        cleaned.append(
            replace(
                surface,
                exterior=rings[0],
                interiors=tuple(rings[1:]),
                provenance=surface.provenance
                + (
                    ("geometry_adaptation:consecutive_vertex_cleanup_within_0.001m",)
                    if changed
                    else ()
                ),
            )
        )
    return tuple(cleaned), canonical, removed


def _triangulate(surface, *, tolerance, used_ids, reason):
    pieces = surface_pieces(surface, planarity_tolerance=tolerance)
    if _points(pieces) != _points((surface,)):
        fail(f"Triangulation would lose source vertices on {surface.face_id}", "unsupported_geometry")
    result = []
    for i, piece in enumerate(pieces):
        identifier = f"{surface.face_id}:adapt:triangle:{i}"
        suffix = 0
        while identifier in used_ids:
            suffix += 1
            identifier = f"{surface.face_id}:adapt:triangle:{i}:{suffix}"
        used_ids.add(identifier)
        result.append(
            replace(
                piece,
                face_id=identifier,
                source_face_id=identifier,
                provenance=(
                    *surface.provenance,
                    f"source_face:{surface.source_face_id or surface.face_id}",
                    reason,
                ),
            )
        )
    return tuple(result)


def adapt_scene(
    scene: GeometryScene, *, max_deviation: float = 0.05
) -> tuple[GeometryScene, dict]:
    """Return a planar, validated scene and a JSON-serializable source/output ledger.

    Input warping up to ``max_deviation`` metres is triangulated, retaining every
    source vertex during decomposition, with the validator's hard topology checks
    and its separately reported 1 mm vertex equivalence. Consecutive vertices
    sharing a representative are cleaned before face checks; no face or hole may
    be collapsed or discarded.
    Each triangle becomes a distinct source face for downstream coverage checks;
    its raw source identity remains in the ledger and surface provenance.
    """
    try:
        if isinstance(max_deviation, bool) or not isinstance(max_deviation, (int, float)):
            raise ValueError
        max_deviation = float(max_deviation)
        if not math.isfinite(max_deviation) or max_deviation < 0:
            raise ValueError
    except (ValueError, OverflowError):
        fail("max_deviation must be a finite non-negative number of metres", "adaptation_invalid")
    if not scene.solids or len({s.solid_id for s in scene.solids}) != len(scene.solids):
        fail("Empty scene or duplicate solid identifiers")
    tolerance = max(TOLERANCE, max_deviation)
    mappings, normalized_solids, normalization = [], [], []
    correspondence_by_solid = {}
    owners, used_by_solid = {}, {}
    original_triangulations = 0
    removed_occurrences = []
    for solid in scene.solids:
        if solid.interior_shells:
            fail("Geometry adaptation does not support enclosed interior shells", "unsupported_geometry")
        used_ids = {surface.face_id for surface in solid.surfaces}
        if len(used_ids) != len(solid.surfaces):
            fail("Duplicate face identifiers")
        used_ids.update(surface.source_face_id for surface in solid.surfaces)
        used_by_solid[solid.solid_id] = used_ids
        source_surfaces, cleanup_mapping, removed = _clean_consecutive_vertices(solid.surfaces)
        removed_occurrences.extend({"solid_id": solid.solid_id, **item} for item in removed)
        surfaces = []
        for original_surface, surface in zip(solid.surfaces, source_surfaces):
            plane = face_plane(surface, planarity_tolerance=tolerance)
            deviation = plane_deviation(surface, plane)
            original_deviation = plane_deviation(original_surface, plane)
            if original_deviation > tolerance:
                fail(
                    f"Non-planar source face {surface.face_id}: exceeds {tolerance:g} m input tolerance",
                    "planarity_tolerance_exceeded",
                )
            pieces = (surface,)
            if deviation > TOLERANCE:
                pieces = _triangulate(
                    surface,
                    tolerance=tolerance,
                    used_ids=used_ids,
                    reason="geometry_adaptation:triangulated_without_vertex_relocation",
                )
                original_triangulations += 1
            mapping = {
                "solid_id": solid.solid_id,
                "reconstruction_id": solid.reconstruction_id,
                "source_face_id": surface.source_face_id,
                "source_object_id": surface.source_object_id,
                "input_face_id": surface.face_id,
                "semantic": surface.semantic,
                "source_provenance": list(original_surface.provenance),
                "source_rings": [[list(point) for point in ring] for ring in _rings(original_surface)],
                "original_deviation_m": original_deviation,
                "input_triangulated": len(pieces) > 1,
                "normalization_triangulated": False,
                "output_face_ids": [],
            }
            mappings.append(mapping)
            for piece in pieces:
                owners[(solid.solid_id, piece.face_id)] = mapping
            surfaces.extend(pieces)
        vertex_mapping = {}
        noded = node_surfaces(tuple(surfaces), vertex_mapping=vertex_mapping)
        correspondence_by_solid[solid.solid_id] = {
            point: vertex_mapping[representative] for point, representative in cleanup_mapping.items()
        }
        normalization.append(
            {
                "solid_id": solid.solid_id,
                "changed_face_ids": [
                    before.face_id
                    for before, after in zip(surfaces, noded)
                    if _rings(before) != _rings(after)
                ],
                "noded_vertex_occurrences_added": sum(
                    len(ring) for surface in noded for ring in _rings(surface)
                ) - sum(len(ring) for surface in surfaces for ring in _rings(surface)),
                "triangulated_face_count": 0,
            }
        )
        normalized_solids.append(replace(solid, surfaces=noded))

    # Validate before further decomposition: normalization must not bypass the
    # existing validator's 1 mm input limit, topology or self-intersection checks.
    candidate = replace(scene, solids=tuple(normalized_solids))
    validated = validate_scene(candidate)
    if any(
        before.surfaces != after.surfaces
        for before, after in zip(candidate.solids, validated.solids)
    ):
        fail("Vertex normalization did not reach a stable result", "unsupported_geometry")
    planar_solids = []
    for solid, record in zip(validated.solids, normalization):
        record["validation_before_planarization"] = dict(solid.validation)
        surfaces = []
        for surface in solid.surfaces:
            owner = owners[(solid.solid_id, surface.face_id)]
            pieces = (surface,)
            if plane_deviation(surface) > TOLERANCE:
                pieces = _triangulate(
                    surface,
                    tolerance=solid.validation["planarity_tolerance_m"],
                    used_ids=used_by_solid[solid.solid_id],
                    reason="validation_normalization:triangulated_without_vertex_relocation",
                )
                record["triangulated_face_count"] += 1
                owner["normalization_triangulated"] = True
            owner["output_face_ids"].extend(piece.face_id for piece in pieces)
            surfaces.extend(pieces)
        planar_solids.append(replace(solid, surfaces=tuple(surfaces)))
    output = validate_scene(replace(scene, solids=tuple(planar_solids)))
    for solid in output.solids:
        for surface in solid.surfaces:
            face_plane(surface, planarity_tolerance=TOLERANCE)

    correspondence = []
    for original, planar, final, record in zip(
        scene.solids, planar_solids, output.solids, normalization
    ):
        record["post_triangulation_changed_face_ids"] = [
            before.face_id
            for before, after in zip(planar.surfaces, final.surfaces)
            if _rings(before) != _rings(after)
        ]
        record["post_triangulation_noded_vertex_occurrences_added"] = sum(
            len(ring) for surface in final.surfaces for ring in _rings(surface)
        ) - sum(len(ring) for surface in planar.surfaces for ring in _rings(surface))
        vertex_mapping = correspondence_by_solid[original.solid_id]
        final_occurrences = _occurrences(final.surfaces)
        original_occurrences = _occurrences(original.surfaces)
        if set(vertex_mapping.values()) != set(final_occurrences):
            fail("Validation changed the vertex correspondence", "unsupported_geometry")
        movements = []
        for point in sorted(original_occurrences):
            representative = vertex_mapping[point]
            distance = math.dist(point, representative)
            movements.append(distance)
            correspondence.append(
                {
                    "solid_id": original.solid_id,
                    "source": list(point),
                    "output": list(representative),
                    "movement_m": distance,
                    "source_occurrences": original_occurrences[point],
                    "output_occurrences": final_occurrences[representative],
                }
            )
        record["moved_vertex_count"] = sum(distance > 0 for distance in movements)
        record["max_vertex_movement_m"] = max(movements, default=0.0)
        record["validation"] = dict(final.validation)

    before_hash = scene_geometry_hash(scene)
    after_hash = scene_geometry_hash(output)
    ledger = {
        "schema_version": "1.0",
        "mode": "triangulate",
        "max_deviation_m": max_deviation,
        "input_planarity_tolerance_m": tolerance,
        "arithmetic_tolerance_m": TOLERANCE,
        "vertex_equivalence_tolerance_m": VERTEX_EQUIVALENCE_TOLERANCE,
        "removed_consecutive_vertex_count": len(removed_occurrences),
        "removed_consecutive_vertices": removed_occurrences,
        "before_geometry_hash": before_hash,
        "after_geometry_hash": after_hash,
        "changed": before_hash != after_hash,
        "frame": asdict(scene.frame),
        "coordinate_translation": {
            "global_to_local_m": [-value for value in scene.frame.origin],
            "local_to_global_m": list(scene.frame.origin),
            "adaptation_translation_m": [0.0, 0.0, 0.0],
        },
        "max_original_deviation_m": max(item["original_deviation_m"] for item in mappings),
        "input_face_count": len(mappings),
        "output_face_count": sum(len(solid.surfaces) for solid in output.solids),
        "triangulated_face_count": sum(
            len(item["output_face_ids"]) > 1 for item in mappings
        ),
        "input_triangulated_face_count": original_triangulations,
        "normalization_triangulated_face_count": sum(
            record["triangulated_face_count"] for record in normalization
        ),
        "triangles_created": sum(
            len(item["output_face_ids"]) for item in mappings if len(item["output_face_ids"]) > 1
        ),
        "face_mappings": mappings,
        "source_vertex_count": len(correspondence),
        "output_vertex_count": sum(len(_points(solid.surfaces)) for solid in output.solids),
        "moved_vertex_count": sum(item["movement_m"] > 0 for item in correspondence),
        "max_vertex_movement_m": max(
            (item["movement_m"] for item in correspondence), default=0.0
        ),
        "vertex_correspondence": correspondence,
        "validation_normalization": normalization,
        "fidelity_status": "unverified",
    }
    # Keep callers from accidentally publishing NaN/Infinity in the audit trail.
    json.dumps(ledger, allow_nan=False)
    return output, ledger


def write_scene_cityjson(scene: GeometryScene, path: Path, *, lod: str = "2.2") -> Path:
    """Write canonical local metric geometry for an independent CityJSON gate."""
    if lod not in {"1.2", "2.2"}:
        fail("Unsupported export LoD", "lod_unavailable")
    validated = validate_scene(scene)
    for solid in validated.solids:
        for surface in solid.surfaces:
            face_plane(surface, planarity_tolerance=TOLERANCE)
    if scene_geometry_hash(validated) != scene_geometry_hash(scene):
        fail("Serialize an adapted canonical scene, not unreported vertex normalization")
    try:
        crs = CRS.from_user_input(scene.frame.projected_crs)
        if not crs.is_projected or any(
            abs(axis.unit_conversion_factor - 1.0) > 1e-12 for axis in crs.axis_info[:2]
        ):
            raise ValueError("The frame CRS must be projected in metres")
        if len(scene.frame.origin) != 3 or not all(math.isfinite(v) for v in scene.frame.origin):
            raise ValueError("The frame origin must contain three finite coordinates")
        if not scene.frame.vertical_reference:
            raise ValueError("An explicit vertical reference is required")
    except (ValueError, TypeError) as error:
        raise ReconstructionError("cityjson_invalid", str(error)) from error
    vertices, indices, grouped = [], {}, defaultdict(lambda: defaultdict(list))
    for solid in validated.solids:
        sources = {surface.source_object_id for surface in solid.surfaces if surface.source_object_id}
        source = next(iter(sources)) if len(sources) == 1 else solid.reconstruction_id
        if not solid.reconstruction_id:
            fail("CityJSON requires a nonempty reconstruction identifier", "cityjson_invalid")
        grouped[solid.reconstruction_id][source].append(solid)
    objects = {root: {"type": "Building"} for root in grouped}

    def ring_indices(ring):
        result = []
        for point in ring:
            if point not in indices:
                indices[point] = len(vertices)
                vertices.append([float(value) for value in point])
            result.append(indices[point])
        return result

    def geometry(solids):
        boundaries, values, definitions = [], [], []
        for solid in solids:
            shell, shell_values = [], []
            for surface in solid.surfaces:
                if surface.semantic not in {
                    "GroundSurface", "RoofSurface", "WallSurface",
                    "OuterFloorSurface", "OuterCeilingSurface",
                }:
                    fail(f"Unsupported CityJSON semantic {surface.semantic!r}", "cityjson_invalid")
                shell.append([ring_indices(ring) for ring in _rings(surface)])
                shell_values.append(len(definitions))
                definitions.append(
                    {
                        "type": surface.semantic,
                        "geogen_face_id": surface.face_id,
                        "geogen_source_face_id": surface.source_face_id,
                    }
                )
            boundaries.append([shell])
            values.append([shell_values])
        multiple = len(solids) > 1
        return {
            "type": "MultiSolid" if multiple else "Solid",
            "lod": lod,
            "boundaries": boundaries if multiple else boundaries[0],
            "semantics": {
                "surfaces": definitions,
                "values": values if multiple else values[0],
            },
        }

    for root, groups in grouped.items():
        if root in groups and len(groups) > 1:
            fail("Selected parent and child geometry cannot both be serialized", "cityjson_invalid")
        for source, solids in groups.items():
            identifier = root
            if source != root:
                identifier = source
                suffix = 0
                while identifier in objects:
                    suffix += 1
                    identifier = f"{root}:part:{source}:{suffix}"
                objects[identifier] = {"type": "BuildingPart", "parents": [root]}
                objects[root].setdefault("children", []).append(identifier)
            objects[identifier]["attributes"] = {
                "geogen_reconstruction_id": root,
                "geogen_solid_ids": [solid.solid_id for solid in solids],
            }
            objects[identifier]["geometry"] = [geometry(solids)]
    document = {
        "type": "CityJSON",
        "version": "1.1",
        "transform": {"scale": [1, 1, 1], "translate": list(scene.frame.origin)},
        "metadata": {"referenceSystem": scene.frame.projected_crs},
        "CityObjects": objects,
        "vertices": vertices,
    }
    path = Path(path)
    path.write_text(json.dumps(document, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return path
