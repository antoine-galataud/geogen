"""Strict CityJSON/CityJSONSeq import into an explicit local metric frame."""

from __future__ import annotations

import json
import math
from pathlib import Path

from pyproj import CRS

from .types import GeometryFrame, GeometryScene, ReconstructionError, Solid, Surface


def _fail(message):
    raise ReconstructionError("cityjson_invalid", message)


def _triple(value, name):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)
    ):
        _fail(f"{name} must contain three finite numbers")
    try:
        converted = tuple(float(v) for v in value)
    except OverflowError:
        _fail(f"{name} is outside the finite coordinate range")
    if not all(math.isfinite(v) for v in converted):
        _fail(f"{name} must contain three finite numbers")
    return converted


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            _fail(f"Duplicate JSON member {name!r}")
        result[name] = value
    return result


def import_cityjson(path: Path, frame: GeometryFrame, *, lod: str = "2.2") -> GeometryScene:
    """Import only the explicitly requested LoD; never fall back to another LoD.

    Sequence vertices are feature-local. The header transform is applied once,
    before subtracting the supplied frame origin. No implicit reprojection or
    vertical datum conversion takes place.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
        try:
            records = [json.loads(text, object_pairs_hook=_unique_object)]
        except json.JSONDecodeError:
            records = [
                json.loads(line, object_pairs_hook=_unique_object)
                for line in text.splitlines()
                if line.strip()
            ]
    except (OSError, ValueError) as error:
        raise ReconstructionError("cityjson_invalid", f"Cannot read CityJSON: {error}") from error
    if not records or not all(isinstance(record, dict) for record in records):
        _fail("Expected a CityJSON object or a CityJSONSeq header and features")
    header = records[0]
    if header.get("type") != "CityJSON":
        _fail("CityJSONSeq requires a CityJSON header")
    if str(header.get("version")) not in {"1.0", "1.1", "2.0"}:
        _fail("Unsupported or missing CityJSON version")
    if header.get("geometry-templates") or header.get("extensions"):
        _fail("Geometry templates and CityJSON extensions are not supported")
    _triple(frame.origin, "frame origin")
    if not frame.vertical_reference:
        _fail("An explicit vertical reference is required")
    if not isinstance(header.get("metadata", {}), dict):
        _fail("Metadata must be an object")
    try:
        projected = CRS.from_user_input(frame.projected_crs)
        reference = header.get("metadata", {}).get("referenceSystem")
        if not projected.is_projected or any(
            abs(axis.unit_conversion_factor - 1.0) > 1e-12 for axis in projected.axis_info[:2]
        ):
            _fail("The supplied frame must use a projected CRS in metres")
        if reference and not CRS.from_user_input(reference).equals(projected):
            _fail("CityJSON CRS differs from the supplied frame; explicit reprojection is required")
    except (ValueError, TypeError) as error:
        if isinstance(error, ReconstructionError):
            raise
        _fail(f"Invalid coordinate reference system: {error}")
    transform = header.get("transform")
    if transform is None:
        scale, translate = (1.0,) * 3, (0.0,) * 3
    elif isinstance(transform, dict):
        scale = _triple(transform.get("scale"), "transform scale")
        translate = _triple(transform.get("translate"), "transform translate")
        if any(v <= 0 for v in scale):
            _fail("Transform scale must be positive")
    else:
        _fail("Invalid transform")
    sequence = len(records) > 1
    if sequence and (header.get("CityObjects") or header.get("vertices")):
        _fail("Sequence headers must not contain geometry")
    solids = []
    seen = set()
    for feature in records[1:] if sequence else records:
        if sequence and feature.get("type") != "CityJSONFeature":
            _fail("Expected CityJSONFeature after the header")
        if sequence and "transform" in feature:
            _fail("Feature-level transforms are unsupported; use the header transform")
        objects = feature.get("CityObjects")
        vertices = feature.get("vertices")
        if not isinstance(objects, dict) or not objects or not isinstance(vertices, list):
            _fail("Each geometry record needs CityObjects and feature-local vertices")
        points = []
        for vertex in vertices:
            vertex = _triple(vertex, "vertex")
            point = tuple(vertex[i] * scale[i] + translate[i] for i in range(3))
            points.append(_triple(frame.to_local(point), "transformed vertex"))
        parents = {key: set() for key in objects}
        for key, obj in objects.items():
            if (
                not key
                or not isinstance(obj, dict)
                or obj.get("type") not in {"Building", "BuildingPart"}
            ):
                _fail(f"Unsupported CityObject {key!r}; only Building/BuildingPart are accepted")
            for relationship in ("children", "parents"):
                references = obj.get(relationship, [])
                if not isinstance(references, list) or any(
                    not isinstance(r, str) for r in references
                ):
                    _fail(f"{relationship} must be an array of CityObject identifiers")
            for child in obj.get("children", []):
                if child not in objects:
                    _fail(f"Unresolved child {child!r}")
                parents[child].add(key)
            for parent in obj.get("parents", []):
                if parent not in objects:
                    _fail(f"Unresolved parent {parent!r}")
                parents[key].add(parent)
        roots = {}
        for key in objects:
            if parents[key] and objects[key]["type"] != "BuildingPart":
                _fail("Only BuildingParts can have Building parents")
            current, visited = key, set()
            while parents[current]:
                if current in visited or len(parents[current]) != 1:
                    _fail("Cyclic or multiply-parented BuildingPart hierarchy")
                visited.add(current)
                current = next(iter(parents[current]))
            if objects[current]["type"] != "Building":
                _fail("Every BuildingPart must resolve to a root Building")
            roots[key] = current
        if sequence:
            if not isinstance(feature.get("id"), str) or set(roots.values()) != {feature["id"]}:
                _fail("Feature id must identify its sole root Building")
        for root in set(roots.values()):
            if root in seen:
                _fail(f"Duplicate reconstruction id {root!r}")
            seen.add(root)
        selected = {}
        for key, obj in objects.items():
            geometries = obj.get("geometry", [])
            if not isinstance(geometries, list) or any(not isinstance(g, dict) for g in geometries):
                _fail(f"Invalid geometry array for {key!r}")
            matches = [(i, g) for i, g in enumerate(geometries) if str(g.get("lod")) == str(lod)]
            if len(matches) > 1:
                _fail(f"Ambiguous LoD {lod} for {key!r}")
            if matches:
                selected[key] = matches[0]
            elif geometries and not any(key in values for values in parents.values()):
                raise ReconstructionError("lod_unavailable", f"No LoD {lod} for {key!r}")
        for key in selected:
            ancestor = key
            while parents[ancestor]:
                ancestor = next(iter(parents[ancestor]))
                if ancestor in selected:
                    _fail("Selected parent and child geometries would duplicate an exterior")
        for root in set(roots.values()):
            if not any(roots[key] == root for key in selected):
                raise ReconstructionError("lod_unavailable", f"No LoD {lod} for root {root!r}")
        for key, (geometry_index, geometry) in selected.items():
            kind = geometry.get("type")
            if kind not in {"Solid", "MultiSolid"}:
                _fail(f"Unsupported geometry {kind!r}; closed Solid/MultiSolid is required")
            boundaries = geometry.get("boundaries")
            semantics = geometry.get("semantics")
            if not isinstance(semantics, dict):
                _fail(f"Missing surface semantics on {key!r}")
            definitions, values = semantics.get("surfaces"), semantics.get("values")
            if not isinstance(definitions, list) or not isinstance(values, list):
                _fail("Malformed surface semantics")
            if kind == "Solid":
                boundaries, values = [boundaries], [values]
            if not isinstance(boundaries, list) or not boundaries or len(boundaries) != len(values):
                _fail("Solid/semantics nesting mismatch")
            for solid_index, (shells, shell_values) in enumerate(zip(boundaries, values)):
                if (
                    not isinstance(shells, list)
                    or not shells
                    or not isinstance(shell_values, list)
                    or len(shells) != len(shell_values)
                ):
                    _fail("Shell/semantics nesting mismatch")
                solid_id = f"{roots[key]}:{key}:{geometry_index}:{solid_index}"
                imported = []
                for shell_index, (faces, face_values) in enumerate(zip(shells, shell_values)):
                    if (
                        not isinstance(faces, list)
                        or not faces
                        or not isinstance(face_values, list)
                        or len(faces) != len(face_values)
                    ):
                        _fail("Face/semantics nesting mismatch")
                    surfaces = []
                    for face_index, (rings, semantic_index) in enumerate(zip(faces, face_values)):
                        if (
                            isinstance(semantic_index, bool)
                            or not isinstance(semantic_index, int)
                            or not 0 <= semantic_index < len(definitions)
                            or not isinstance(definitions[semantic_index], dict)
                        ):
                            _fail("Every face requires a valid semantic index")
                        semantic = definitions[semantic_index].get("type")
                        if semantic not in {
                            "GroundSurface",
                            "RoofSurface",
                            "WallSurface",
                            "OuterFloorSurface",
                            "OuterCeilingSurface",
                        }:
                            _fail(f"Unsupported surface semantic {semantic!r}")
                        if not isinstance(rings, list) or not rings:
                            _fail("A face needs at least one ring")
                        imported_rings = []
                        for ring in rings:
                            if not isinstance(ring, list) or len(ring) < 3:
                                _fail("A ring needs at least three vertex indices")
                            if any(
                                isinstance(i, bool)
                                or not isinstance(i, int)
                                or not 0 <= i < len(points)
                                for i in ring
                            ):
                                _fail("Vertex index outside the feature-local vertex array")
                            imported_rings.append(tuple(points[i] for i in ring))
                        face_id = f"{solid_id}:{shell_index}:{face_index}"
                        surfaces.append(
                            Surface(
                                face_id=face_id,
                                exterior=imported_rings[0],
                                interiors=tuple(imported_rings[1:]),
                                semantic=semantic,
                                source_object_id=key,
                                source_face_id=face_id,
                                provenance=(f"CityJSON LoD {lod}",),
                            )
                        )
                    imported.append(tuple(surfaces))
                solids.append(
                    Solid(solid_id, roots[key], imported[0], interior_shells=tuple(imported[1:]))
                )
    if not solids:
        _fail("No solids imported")
    return GeometryScene(frame, tuple(solids))
