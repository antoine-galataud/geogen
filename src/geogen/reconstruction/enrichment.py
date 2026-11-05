"""Source-scoped thermal enrichment without replacing reconstructed exteriors."""

from __future__ import annotations

import math
from dataclasses import asdict

from geogen.envelope import glazing_orientations, window_to_wall_ratio
from geogen.models import NOT_PROVIDED
from geogen.thermal_properties import ENVELOPE_FIELDS

from .thermal import make_thermal_scene


def consensus(values):
    """Keep provenance and decline conflicting or partially missing source values."""
    known = {
        key: value
        for key, value in values.items()
        if value is not None and value != NOT_PROVIDED and value != "" and value != ()
    }
    value = next(iter(known.values()), None)
    if not known:
        status, value = "missing", None
    elif any(other != value for other in known.values()):
        status, value = "conflicting", None
    elif len(known) != len(values):
        status, value = "incomplete_source_coverage", None
    else:
        status = "available"
    return {"status": status, "value": value, "source_values": values}


def solid_source_groups(scene):
    """Use exact solid links, never identifier prefixes or compatibility construction IDs."""
    groups = {}
    for solid in scene.solids:
        ids = {
            link.source_id
            for link in solid.links
            if link.source_type == "bdnb_group"
            and link.relationship not in {"unresolved", "unknown", ""}
            and link.confidence not in {"unresolved", "unknown", ""}
        }
        groups[solid.solid_id] = sorted(ids)
    return groups


def glazing_parameters(scene, buildings, report, groups_by_solid):
    """Resolve source hints once, then apply the same decisions to each storey."""
    sources = {
        group: buildings.get(f"FR:{group}")
        for groups in groups_by_solid.values()
        for group in groups
    }
    source_ratios = {
        group: window_to_wall_ratio(building) if building is not None else None
        for group, building in sources.items()
    }
    source_orientations = {
        group: tuple(sorted(glazing_orientations(building))) if building is not None else ()
        for group, building in sources.items()
    }
    by_solid = {
        solid_id: (
            consensus({group: source_ratios[group] for group in groups}),
            consensus({group: source_orientations[group] for group in groups}),
        )
        for solid_id, groups in groups_by_solid.items()
    }
    ratios, orientations, decisions = {}, {}, []
    for space in scene.spaces:
        ratio, orientation = by_solid[space.solid_id]
        if ratio["status"] == "available":
            ratios[space.space_id] = ratio["value"]
        if orientation["status"] == "available":
            orientations[space.space_id] = orientation["value"]
        decisions.append(
            {
                "space_id": space.space_id,
                "source_group_ids": groups_by_solid[space.solid_id],
                "source_ratio": ratios.get(space.space_id),
                "scope": "modeled_wwr_not_representative_dwelling_area",
                "reason": (
                    "source_ratio_or_cli_default"
                    if ratio["status"] in {"available", "missing"}
                    else ratio["status"] + "_source_ratios"
                ),
                "source_decision": ratio,
                "modeled_orientations": list(orientations.get(space.space_id, ())),
                "orientation_decision": orientation,
            }
        )
    report["glazing_assumptions"] = decisions
    return ratios, orientations


def enrich_thermal_scene(scene, buildings, report, *, storey_height):
    """Attempt source-backed levels, with explicit geometric priors and safe fallback."""
    if not math.isfinite(storey_height) or storey_height <= 0:
        raise ValueError("Storey height must be finite and positive")
    groups = solid_source_groups(scene)
    levels, specifications, decisions = {}, {}, {}
    for solid in scene.solids:
        sources = {key: buildings.get(f"FR:{key}") for key in groups[solid.solid_id]}
        attributes = {
            field: consensus(
                {key: getattr(building, field, None) for key, building in sources.items()}
            )
            for field in ENVELOPE_FIELDS
        }
        specifications[solid.solid_id] = {
            "source_group_ids": groups[solid.solid_id],
            "scope": "bdnb_group_values_applied_to_linked_physical_solid",
            "links": [asdict(link) for link in solid.links if link.source_type == "bdnb_group"],
            "values": {
                field: decision["value"]
                for field, decision in attributes.items()
                if decision["status"] == "available"
            },
            "decisions": attributes,
        }
        count = consensus(
            {key: getattr(building, "storeys", None) for key, building in sources.items()}
        )
        ground_z = [
            p[2]
            for face in solid.surfaces
            if face.semantic == "GroundSurface"
            for p in face.exterior
        ]
        roof_z = [
            p[2]
            for face in solid.surfaces
            if face.semantic == "RoofSurface"
            for ring in (face.exterior, *face.interiors)
            for p in ring
        ]
        decision = {
            "source_group_ids": groups[solid.solid_id],
            "source_storeys": count,
            "scope": "per_physical_solid; group count is a modeling assumption, not allocated area",
            "requested_levels_m": [],
        }
        decisions[solid.solid_id] = decision
        if not ground_z or not roof_z:
            decision["reason"] = "missing_ground_or_roof_semantics"
            continue
        base, top = min(ground_z), min(roof_z)
        if top <= base:
            top = max(roof_z)
        height = top - base
        number = count["value"]
        if count["status"] == "missing":
            number = max(1, round(height / storey_height))
            decision.update(
                count_source="reconstructed_height_and_storey_height_prior",
                storey_height_prior_m=storey_height,
            )
        elif count["status"] == "available":
            decision["count_source"] = "bdnb_group"
        else:
            decision["reason"] = "conflicting_or_incomplete_storey_count"
            continue
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or int(number) != number
            or not 1 <= number <= 100
            or height <= 0
        ):
            decision["reason"] = "unsupported_storey_count_or_height"
            continue
        number = int(number)
        cuts = [base + height * index / number for index in range(1, number)]
        levels[solid.solid_id] = cuts
        decision.update(
            requested_storeys=number,
            requested_levels_m=cuts,
            floor_placement="equal_spacing_below_lowest_roof; top_storey_includes_roof_volume",
            base_elevation_m=base,
            nominal_storey_height_m=height / number,
        )
    partitions = {}
    thermal = make_thermal_scene(scene, levels=levels, partition_report=partitions)
    for solid_id, decision in decisions.items():
        decision["partition"] = partitions[solid_id]
        decision["modeled_space_count"] = len(partitions[solid_id]["space_ids"])
    report["storey_enrichment"] = decisions
    properties = {space.space_id: specifications[space.solid_id] for space in thermal.spaces}
    return thermal, properties, groups
