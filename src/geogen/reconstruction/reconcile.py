"""Typed BDNB/BD TOPO relationships and conservative physical-unit reconciliation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from typing import Any, Iterable

from pyproj import CRS
from shapely import normalize, to_wkt
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from geogen.geometry import parse_geometry
from geogen.models import Building
from geogen.reconstruction.types import ReconstructionError, ReconstructionUnit, SourceIdentity, SourceLink

BDTOPO_ID_FIELDS = (
    "bdtopo_bat_cleabs", "bdtopo_bat_id", "bdtopo_id", "cleabs", "id_bdtopo",
    "l_bdtopo_bat_id", "l_bdtopo_id",
)


def _identifiers(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("["):
            try:
                return _identifiers(json.loads(value))
            except ValueError as error:
                raise ReconstructionError("malformed_identity", "Malformed source identifier array") from error
        if value.startswith("{") and value.endswith("}"):
            value = value[1:-1].split(",")
        else:
            return (value,) if value else ()
    if isinstance(value, (tuple, list)):
        return tuple(sorted({str(item).strip().strip('"') for item in value if item is not None and str(item).strip()}))
    raise ReconstructionError("malformed_identity", "Source identities must be strings or string arrays")


def _row_ids(row: dict[str, Any], names: Iterable[str]) -> set[str]:
    return {value for name in names for value in _identifiers(row.get(name))}


def linked_bdtopo_ids(relation_rows) -> tuple[str, ...]:
    rows = relation_rows.get("rows", ()) if isinstance(relation_rows, dict) else relation_rows
    return tuple(sorted({
        value for row in rows for value in _row_ids(row, BDTOPO_ID_FIELDS)
    }))


def _polygon(raw, crs="EPSG:2154"):
    try:
        if CRS.from_user_input(crs) != CRS.from_epsg(2154):
            raise ValueError("not EPSG:2154")
        geometry = parse_geometry(raw)
    except Exception as error:
        raise ReconstructionError("invalid_outline", "Outline must be a valid EPSG:2154 polygon") from error
    if not isinstance(geometry, (Polygon, MultiPolygon)) or geometry.is_empty or not geometry.is_valid:
        raise ReconstructionError("invalid_outline", "Outline must be a nonempty valid polygon or multipolygon")
    return geometry


def _canonical(geometry) -> str:
    return to_wkt(normalize(geometry), rounding_precision=-1)


def _parts(geometry):
    parts = list(geometry.geoms) if isinstance(geometry, MultiPolygon) else [geometry]
    return sorted(parts, key=_canonical)


def _overlapping(geometries) -> bool:
    if len(geometries) < 2:
        return False
    union = unary_union(geometries)
    return sum(item.area for item in geometries) > union.area * 1.05


def _outline_type(raw) -> str:
    value = raw.get("outline_type", "unknown") if isinstance(raw, dict) else "unknown"
    if value == "groundprint":
        value = "wall_footprint"
    if value not in {"roofprint", "wall_footprint", "unknown"}:
        raise ReconstructionError("unknown_outline_type", "Outline type must be roofprint, wall_footprint or unknown")
    return value


def resolve_units(
    buildings: dict[str, Building], relation_rows, features, *,
    dataset_versions: dict[str, str] | Iterable[tuple[str, str]] | None = None,
) -> tuple[ReconstructionUnit, ...]:
    """Associate whole physical components; never coerce a compatibility ID.

    Pairwise coverage alone cannot describe split/merged groups. Spatial links
    require both-sided coverage, possibly through a nonoverlapping union. Explicit
    provider links retain their many-to-many meaning and group-level attribute scope.
    """
    selected = set(buildings)
    versions = tuple(sorted(dict(dataset_versions or {}).items()))
    rows = relation_rows.get("rows", ()) if isinstance(relation_rows, dict) else relation_rows
    if isinstance(features, dict):
        if features.get("crs", "EPSG:2154") != "EPSG:2154":
            raise ReconstructionError("crs_mismatch", "Physical outlines must use EPSG:2154")
        features = features.get("features", ())
    identities = {
        group_id: {"constructions": set(), "bdtopo": set(), "rnb": set()}
        for group_id in buildings
    }
    group_rows = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            raise ReconstructionError("malformed_identity", "A relationship row is not an object")
        group_id = str(row.get("batiment_groupe_id") or "")
        if group_id not in identities:
            continue
        group_rows[group_id].append(row)
        ids = identities[group_id]
        ids["constructions"].update(_row_ids(row, ("batiment_construction_id",)))
        ids["bdtopo"].update(_row_ids(row, BDTOPO_ID_FIELDS))
        ids["rnb"].update(_row_ids(row, ("rnb_id", "rnb_ids")))

    group_geometries = {}
    for group_id, building in buildings.items():
        actual_group = getattr(building, "batiment_groupe_id", group_id)
        if not actual_group or actual_group != group_id:
            raise ReconstructionError("group_identity_mismatch", "Selected IDs must be explicit BDNB group IDs")
        if building.geometry is not None and not building.fictitious_geometry:
            group_geometries[group_id] = _polygon(building.geometry, building.crs)

    physical = {}
    for feature in features:
        if not isinstance(feature, dict) or not isinstance(feature.get("properties"), dict):
            raise ReconstructionError("invalid_outline", "Malformed BD TOPO feature")
        properties = feature["properties"]
        source_id = str(properties.get("cleabs") or properties.get("bdtopo_id") or "")
        if not source_id:
            raise ReconstructionError("missing_source_id", "BD TOPO feature lacks an explicit cleabs")
        geometry = _polygon(feature.get("geometry"))
        if source_id in physical:
            if _canonical(physical[source_id]["geometry"]) != _canonical(geometry):
                raise ReconstructionError("conflicting_source_geometry", f"Conflicting outlines for {source_id}")
            if physical[source_id]["properties"] != properties:
                raise ReconstructionError("conflicting_source_metadata", f"Conflicting metadata for {source_id}")
            continue
        physical[source_id] = {
            "geometry": geometry, "properties": properties,
            "rnb": _row_ids(properties, ("identifiants_rnb", "rnb_id")),
            "source_kind": "bdtopo", "outline_type": _outline_type(properties),
        }

    # Compute association at the full-feature level before splitting multipart solids.
    accepted: dict[tuple[str, str], tuple[str, tuple[str, ...]]] = {}
    candidates = {}
    ambiguous: set[tuple[str, str]] = set()
    for group_id, ids in identities.items():
        for source_id, item in physical.items():
            if source_id in ids["bdtopo"]:
                accepted[group_id, source_id] = ("explicit_bdtopo", ())
            elif ids["rnb"] & item["rnb"]:
                accepted[group_id, source_id] = ("explicit_rnb", ())
            source = group_geometries.get(group_id)
            if source is not None:
                area = source.intersection(item["geometry"]).area
                if area > 0:
                    candidates[group_id, source_id] = (area / source.area, area / item["geometry"].area)
    for group_id, source in group_geometries.items():
        matched = [source_id for (gid, source_id), (a, b) in candidates.items()
                   if gid == group_id and b >= 0.9]
        shapes = [physical[source_id]["geometry"] for source_id in matched]
        if shapes and _overlapping(shapes):
            ambiguous.update((group_id, source_id) for source_id in matched)
        elif shapes and source.intersection(unary_union(shapes)).area / source.area >= 0.9:
            for source_id in matched:
                accepted.setdefault((group_id, source_id), ("spatial_union", ("bidirectional_union_coverage",)))
    for source_id, item in physical.items():
        matched = [gid for (gid, sid), (a, b) in candidates.items() if sid == source_id and a >= 0.9]
        shapes = [group_geometries[gid] for gid in matched]
        if shapes and _overlapping(shapes):
            ambiguous.update((gid, source_id) for gid in matched)
        elif shapes and item["geometry"].intersection(unary_union(shapes)).area / item["geometry"].area >= 0.9:
            for group_id in matched:
                accepted.setdefault((group_id, source_id), ("spatial_union", ("bidirectional_union_coverage",)))
    for pair in ambiguous:
        if accepted.get(pair, ("", ()))[0] == "spatial_union":
            del accepted[pair]

    units = []
    resolved_selected = set()
    for source_id, item in sorted(physical.items()):
        linked_groups = sorted(gid for gid, sid in accepted if sid == source_id)
        original_parts = _parts(item["geometry"])
        for index, original in enumerate(original_parts):
            source_key = source_id + ":" + str(index)
            reconstruction_id = "recon-" + hashlib.sha256(source_key.encode()).hexdigest()[:20]
            working = original
            outline_type = item["outline_type"]
            decisions = ["outline_source:bdtopo", "bdtopo_outline"]
            if outline_type == "unknown":
                decisions.append("outline_semantics_unverified")
                decisions.append("wall_extent_assumption=roofprint_extruded")
            links = []
            component_groups = []
            for group_id in linked_groups:
                method, reasons = accepted[group_id, source_id]
                source = group_geometries.get(group_id)
                intersection = source.intersection(original).area if source is not None else None
                # Spatially linked multipart components must actually overlap this group.
                if method == "spatial_union" and not intersection:
                    continue
                component_groups.append(group_id)
                links.append(SourceLink(
                    reconstruction_id, "bdnb_group", group_id, {
                        "explicit_bdtopo": "source_relation", "explicit_rnb": "source_relation",
                        "spatial_union": "spatial_verified",
                    }[method], "one_to_one",
                    intersection_area=intersection,
                    source_coverage=intersection / source.area if source is not None else None,
                    physical_coverage=intersection / original.area if intersection is not None else None,
                    iou=intersection / source.union(original).area if source is not None else None,
                    attribute_scope="group", confidence="confirmed" if method != "spatial_union" else "high",
                    reasons=(method,) + reasons,
                    attribution_polygon=source.wkt if source is not None else None,
                ))
            for group_id in sorted(buildings):
                pair = (group_id, source_id)
                if pair in candidates and group_id not in component_groups:
                    source = group_geometries[group_id]
                    intersection = source.intersection(original).area
                    if intersection:
                        links.append(SourceLink(
                            reconstruction_id, "bdnb_group", group_id, "spatial_verified", "unresolved",
                            intersection_area=intersection, source_coverage=intersection / source.area,
                            physical_coverage=intersection / original.area,
                            iou=intersection / source.union(original).area,
                            attribute_scope="group", reasons=(
                                "ambiguous_overlap" if pair in ambiguous else "insufficient_bidirectional_coverage",
                            ), attribution_polygon=source.wkt,
                        ))
            explicit_rnb = set(item["rnb"])
            construction_ids = set()
            for group_id in component_groups:
                for row in group_rows[group_id]:
                    row_bdtopo = set(linked_bdtopo_ids([row]))
                    row_rnb = _row_ids(row, ("rnb_id", "rnb_ids"))
                    directly_linked = source_id in row_bdtopo or bool(row_rnb & item["rnb"])
                    if not directly_linked and row.get("geom_cstr") and not row.get("fictive_geom_cstr"):
                        construction = _polygon(row["geom_cstr"])
                        overlap = construction.intersection(original).area
                        directly_linked = (
                            overlap / construction.area >= .9 and overlap / original.area >= .9
                        )
                    row_constructions = _row_ids(row, ("batiment_construction_id",))
                    if directly_linked:
                        construction_ids.update(row_constructions)
                        explicit_rnb.update(row_rnb)
                    else:
                        for kind, values in (("bdnb_construction", row_constructions), ("rnb", row_rnb)):
                            for value in sorted(values):
                                links.append(SourceLink(
                                    reconstruction_id, kind, value, "source_relation", "unresolved",
                                    attribute_scope="group", confidence="unresolved",
                                    reasons=("physical_attribution_unresolved",),
                                ))
            construction_ids = tuple(sorted(construction_ids))
            identity = SourceIdentity(
                bdnb_group_ids=tuple(component_groups), bdnb_construction_ids=construction_ids,
                bdtopo_ids=(source_id,),
                rnb_ids=tuple(sorted(explicit_rnb)), dataset_versions=versions,
            )
            source_dates = tuple(str(item["properties"][key]) for key in (
                "date_creation", "date_modification", "date_de_confirmation"
            ) if item["properties"].get(key))
            for kind, ids in (
                ("bdtopo", identity.bdtopo_ids), ("bdnb_construction", construction_ids),
                ("rnb", identity.rnb_ids),
            ):
                for value in ids:
                    links.append(SourceLink(
                        reconstruction_id, kind, value, "source_relation", "one_to_one",
                        attribute_scope="construction" if kind == "bdnb_construction" else "unknown",
                        confidence="confirmed", source_dates=source_dates,
                    ))
            resolved_selected.update(selected & set(component_groups))
            units.append(ReconstructionUnit(
                reconstruction_id, identity, original.wkt, _canonical(working),
                hashlib.sha256(_canonical(working).encode()).hexdigest(),
                outline_type=outline_type,
                role="target" if selected & set(component_groups) else "context",
                component_index=index, decisions=tuple(decisions), links=tuple(links),
            ))
    if selected - resolved_selected:
        raise ReconstructionError(
            "unresolved_identity",
            "No unambiguous physical outline for selected groups: " + ", ".join(sorted(selected - resolved_selected)),
        )
    source_units = defaultdict(set)
    unit_sources = defaultdict(set)
    for unit in units:
        for link in unit.links:
            if link.relationship != "unresolved":
                source_units[link.source_type, link.source_id].add(unit.reconstruction_id)
                unit_sources[unit.reconstruction_id, link.source_type].add(link.source_id)
    units = [
        replace(unit, links=tuple(
            replace(link, relationship=(
                "shared_physical_unit"
                if len(unit_sources[unit.reconstruction_id, link.source_type]) > 1
                else "group_member" if len(source_units[link.source_type, link.source_id]) > 1
                else "one_to_one"
            )) if link.relationship != "unresolved" else link
            for link in unit.links
        ))
        for unit in units
    ]
    return tuple(sorted(units, key=lambda item: item.reconstruction_id))
