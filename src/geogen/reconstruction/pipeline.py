"""Explicit acquisition, reconstruction and acceptance transitions."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import laspy
import numpy as np
from shapely import contains_xy, from_wkt
from shapely.geometry import mapping
from shapely.ops import unary_union

from geogen.reconstruction.cache import (
    ReconstructionCache,
    artifact,
    checked_artifact,
    content_hash,
    file_hash,
    restore_buildings,
    write_json,
)
from geogen.reconstruction.types import (
    GeometryFrame,
    GeometryScene,
    ReconstructionError,
    ReconstructionUnit,
    SourceIdentity,
    SourceLink,
)

REPORTABLE_EVIDENCE_REASONS = frozenset(
    {
        "insufficient_roof_points",
        "insufficient_ground_points",
        "insufficient_roof_density",
        "insufficient_roof_coverage",
        "insufficient_roof_surface_coverage",
        "large_roof_evidence_gap",
        "roof_residual_threshold_exceeded",
        "ground_residual_threshold_exceeded",
        "coverage_computation_limit",
        "unsupported_overhang_surface_evidence",
        "undefined_outline_semantics",
        "user_assumed_roof_height",
        "bdnb_assumed_roof_height",
    }
)

LOD_FALLBACK_REASONS = frozenset(
    {
        "fallback_lod1",
        "lod_unavailable",
        "geometry_invalid",
        "unsupported_geometry",
        "planarity_tolerance_exceeded",
        "vertex_equivalence_collapsed_edge",
        "vertex_cleanup_degenerate_ring",
        "validator_invalid",
    }
)

# Classified ground is searched from 3 m out to this radius around each outline.
GROUND_SEARCH_RADIUS = 10.0
# Maximum warping of a Roofer roof face absorbed by triangulation (metres).
MAX_FACE_DEVIATION = 0.05


def restore_units(records: list[dict]) -> tuple[ReconstructionUnit, ...]:
    try:
        units = []
        for record in records:
            values = dict(record)
            identity = values.pop("identity")
            identity = SourceIdentity(**{k: tuple(v) for k, v in identity.items()})
            links = tuple(SourceLink(**link) for link in values.pop("links"))
            units.append(ReconstructionUnit(**values, identity=identity, links=links))
        if not units or len({u.reconstruction_id for u in units}) != len(units):
            raise ValueError("Empty or duplicate units")
        for unit in units:
            if unit.role not in {"target", "context"} or unit.outline_type not in {
                "roofprint",
                "wall_footprint",
                "unknown",
            }:
                raise ValueError("Unsupported unit role or outline semantics")
            polygon = from_wkt(unit.working_wkt)
            if not polygon.is_valid or polygon.is_empty or polygon.geom_type != "Polygon":
                raise ValueError("Invalid working outline")
            if unit.crs != "EPSG:2154":
                raise ValueError("Unsupported projected reference")
        return tuple(units)
    except (KeyError, TypeError, ValueError) as error:
        raise ReconstructionError("manifest_invalid", "Invalid reconstruction units") from error


def prepare_points(original: Path, working: Path) -> dict:
    """Preserve original classifications in the replay crop, remap only a working copy.

    IGN class 67 ("bâti divers") is treated as building (6) in the working copy.
    """
    try:
        with laspy.open(original) as reader:
            if reader.header.point_count > 5_000_000:
                raise ReconstructionError(
                    "input_limit", "Crop exceeds the bounded five-million point limit"
                )
            crs = reader.header.parse_crs()
            references = [crs, *(crs.sub_crs_list if crs else [])]
            if not any(c and c.to_epsg() == 2154 for c in references):
                raise ReconstructionError(
                    "crs_mismatch", "The LAS crop must explicitly use EPSG:2154"
                )
            expected = reader.header.point_count
            cloud = reader.read()
            if len(cloud.points) != expected:
                raise ReconstructionError("acquisition_incomplete", "Truncated point-cloud crop")
    except ReconstructionError:
        raise
    except (OSError, ValueError, laspy.errors.LaspyException, EOFError) as error:
        raise ReconstructionError(
            "points_invalid", "Unreadable or truncated classified crop"
        ) from error
    coordinates = np.column_stack((cloud.x, cloud.y, cloud.z))
    if not len(coordinates) or not np.isfinite(coordinates).all():
        raise ReconstructionError("points_invalid", "Empty crop or non-finite point coordinates")
    classes, counts = np.unique(cloud.classification, return_counts=True)
    original_counts = {str(int(c)): int(n) for c, n in zip(classes, counts)}
    classification = np.asarray(cloud.classification).copy()
    classification[classification == 67] = 6
    cloud.classification = classification
    # Roofer depends on point order and PDAL's tile interleaving varies, so the
    # working copy uses a canonical order and quantization origin.
    cloud.points = cloud.points[
        np.lexsort((coordinates[:, 2], coordinates[:, 1], coordinates[:, 0], cloud.gps_time))
    ]
    cloud.change_scaling(offsets=np.floor(coordinates.min(axis=0)))
    cloud.write(working)
    return {
        "original_class_counts": original_counts,
        "class67_policy": "include",
        "removed_point_count": 0,
        "coordinate_scales": list(map(float, cloud.header.scales)),
    }


def ground_references(
    working: Path,
    units: tuple[ReconstructionUnit, ...],
    *,
    diagnostics: dict | None = None,
    allow_missing: bool = False,
) -> dict[str, float]:
    """Expand from 3 m only as needed, using classified ground outside all buildings."""
    cloud = laspy.read(working)
    ground = np.asarray(cloud.classification) == 2
    x, y, z = np.asarray(cloud.x)[ground], np.asarray(cloud.y)[ground], np.asarray(cloud.z)[ground]
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[finite], y[finite], z[finite]
    outlines = {unit.reconstruction_id: from_wkt(unit.working_wkt) for unit in units}
    excluded = unary_union(list(outlines.values()))
    radii = [3.0]
    while radii[-1] < GROUND_SEARCH_RADIUS:
        radii.append(min(radii[-1] * 2, GROUND_SEARCH_RADIUS))
    elevations = {}
    diagnostics = diagnostics if diagnostics is not None else {}
    missing = []
    for unit in units:
        if unit.role != "target":
            continue
        record = {
            "status": "ground_unverified",
            "elevation_m": None,
            "method": None,
            "max_search_radius_m": GROUND_SEARCH_RADIUS,
            "search_radius_m": None,
            "ground_point_count": 0,
            "expanded_search": False,
            "attempts": [],
        }
        diagnostics[unit.reconstruction_id] = record
        for radius in radii:
            ring = outlines[unit.reconstruction_id].buffer(radius).difference(excluded)
            values = z[contains_xy(ring, x, y)]
            record["attempts"].append({"radius_m": radius, "ground_point_count": len(values)})
            record["ground_point_count"] = len(values)
            if len(values) >= 3:
                elevation = float(np.percentile(values, 5))
                elevations[unit.reconstruction_id] = elevation
                record.update(
                    status="observed",
                    elevation_m=elevation,
                    method=f"local_ground_5th_percentile_{radius:g}m_buffer",
                    search_radius_m=radius,
                    expanded_search=radius > 3,
                )
                break
        else:
            missing.append(unit.reconstruction_id)
    if missing and not allow_missing:
        raise ReconstructionError(
            "ground_unverified",
            f"No sufficient local ground within {GROUND_SEARCH_RADIUS:g} m for {', '.join(missing)}",
        )
    return elevations


def read_frame(value: dict) -> GeometryFrame:
    try:
        frame = GeometryFrame(
            projected_crs=value["projected_crs"],
            vertical_reference=value["vertical_reference"],
            origin=tuple(float(x) for x in value["origin"]),
        )
        if frame.projected_crs != "EPSG:2154" or len(frame.origin) != 3:
            raise ValueError("Unsupported frame")
        if frame.vertical_reference not in {"IGN69", "NGF-IGN69", "EPSG:5720"}:
            raise ValueError("Vertical reference is not verified")
        if not all(math.isfinite(x) for x in frame.origin):
            raise ValueError("Non-finite origin")
        return frame
    except (KeyError, TypeError, ValueError) as error:
        raise ReconstructionError(
            "reference_unverified", "A verified Lambert-93 / IGN69 frame is required"
        ) from error


def roofer_diagnostics(
    path: Path, reconstruction_id: str, *, diagnostics: dict | None = None
) -> dict:
    """Freeze the observed output contract; extrusion fallbacks are never LoD2 success."""
    try:
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        features = records[1:]
        if len(features) != 1 or features[0].get("id") != reconstruction_id:
            raise ValueError("Expected exactly one selected feature")
        attributes = features[0]["CityObjects"][reconstruction_id]["attributes"]
        if not isinstance(attributes, dict):
            raise ValueError("Roofer attributes must be an object")
        if diagnostics is not None:
            diagnostics.update(attributes)
        required = ("rf_success", "rf_extrusion_mode", "rf_force_lod11", "rf_pointcloud_unusable")
        missing = [name for name in required if name not in attributes]
        if missing:
            raise ValueError(f"Missing Roofer diagnostics: {', '.join(missing)}")
        for name in ("rf_success", "rf_force_lod11", "rf_pointcloud_unusable"):
            if type(attributes[name]) is not bool:
                raise ValueError(f"{name} must be boolean; got {attributes[name]!r}")
        mode = attributes["rf_extrusion_mode"]
        if not isinstance(mode, str) or mode not in {"standard", "lod11_fallback", "skip", "fail"}:
            raise ValueError(f"Unrecognized rf_extrusion_mode: {mode!r}")
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise ReconstructionError(
            "runtime_diagnostics_missing", f"Cannot establish Roofer reconstruction mode: {error}"
        ) from error
    return attributes


def local_footprints(units, frame: GeometryFrame) -> tuple[dict, dict]:
    """Return disjoint local-frame target outlines, the source of every generated wall.

    Overlaps between source outlines would make solids intersect, so later units
    (by identifier) cede overlapping area to earlier ones; trims are reported.
    Adjacent outlines often disagree by millimetres, so the ceding outline is
    first snapped to its neighbours to avoid slivers below ``MIN_FEATURE``.
    """
    from shapely import snap
    from shapely.affinity import translate

    from geogen.reconstruction.conform import MIN_FEATURE

    footprints, trims = {}, {}
    for unit in sorted(units, key=lambda item: item.reconstruction_id):
        polygon = translate(from_wkt(unit.working_wkt), -frame.origin[0], -frame.origin[1])
        claimed = [other for other in footprints.values() if polygon.intersection(other).area > 1e-6]
        if claimed:
            others = unary_union(claimed)
            snapped = snap(polygon, others, MIN_FEATURE)
            remainder = (snapped if snapped.is_valid else polygon).difference(others)
            parts = sorted(
                getattr(remainder, "geoms", [remainder]), key=lambda p: p.area, reverse=True
            )
            if not parts or parts[0].geom_type != "Polygon" or parts[0].area <= 0:
                raise ReconstructionError(
                    "overlapping_outline", f"{unit.reconstruction_id} is covered by other targets"
                )
            trims[unit.reconstruction_id] = {
                "removed_area_m2": polygon.area - parts[0].area,
                "reason": "overlapping_target_outline_ceded_to_earlier_unit",
            }
            polygon = parts[0]
        footprints[unit.reconstruction_id] = polygon
    return footprints, trims


def _geometry_candidate(
    raw,
    frame,
    unit,
    directory,
    work,
    runtime,
    result,
    candidate,
    *,
    lod,
    footprint,
    ground,
):
    """Rebuild Roofer's roof planes over the source footprint, then validate independently."""
    from geogen.reconstruction.adapt import adapt_scene, write_scene_cityjson
    from geogen.reconstruction.cityjson import import_cityjson
    from geogen.reconstruction.conform import conform_scene
    from geogen.reconstruction.validate import TOLERANCE

    imported = import_cityjson(raw, frame, lod=lod)
    if not imported.solids or any(
        solid.reconstruction_id != unit.reconstruction_id for solid in imported.solids
    ):
        raise ReconstructionError(
            "selection_mismatch", "Roofer output does not match the selected unit"
        )
    conformed, conformance = conform_scene(
        imported,
        footprint,
        ground - frame.origin[2],
        tolerance=max(TOLERANCE, MAX_FACE_DEVIATION),
    )
    candidate["conformance"] = conformance
    scene, audit = adapt_scene(conformed, max_deviation=MAX_FACE_DEVIATION)
    candidate["adaptation"] = audit
    if lod == "1.2":
        scene = replace(
            scene,
            solids=tuple(
                replace(
                    solid,
                    surfaces=tuple(
                        replace(face, provenance=(*face.provenance, "roofer_lod_fallback:1.2"))
                        for face in solid.surfaces
                    ),
                )
                for solid in scene.solids
            ),
        )
    # The rebuilt geometry is never Roofer's raw file, so it is always validated itself.
    adapted = write_scene_cityjson(
        scene,
        directory / ("fallback-lod12.city.json" if lod == "1.2" else "adapted.city.json"),
        lod=lod,
    )
    candidate["adapted_artifact"] = artifact(adapted, work)
    independent = runtime.validate_cityjson(adapted, directory)
    result["conformance"] = conformance
    result["adaptation"] = audit
    result["adapted_artifact"] = candidate["adapted_artifact"]
    result["independent_validation"] = independent
    return scene


def _assumed_height(
    unit: ReconstructionUnit, explicit: float | None, bdnb_heights: dict
) -> dict | None:
    """Roof height for a unit without usable LiDAR roof points: explicit, else BDNB."""
    if explicit is not None:
        return {
            "method": "user_assumed_height_above_observed_ground",
            "source": "user",
            "reason": "user_assumed_roof_height",
            "height_m": explicit,
        }
    groups = unit.identity.bdnb_group_ids
    heights = {bdnb_heights.get(group) for group in groups}
    if len(heights) != 1:
        return None
    (height,) = heights
    if not isinstance(height, (int, float)) or not math.isfinite(height) or not 0 < height <= 300:
        return None
    return {
        "method": "bdnb_height_above_observed_ground",
        "source": "bdnb",
        "reason": "bdnb_assumed_roof_height",
        "height_m": float(height),
        "bdnb_group_ids": list(groups),
    }


def reconstruct_manifest(
    manifest: dict,
    manifest_path: Path,
    work: Path,
    runtime,
    *,
    cache: ReconstructionCache,
    report: dict,
    missing_roof_height: float | None = None,
) -> tuple[GeometryScene, tuple[ReconstructionUnit, ...]]:
    """Reconstruct every target with Roofer; LoD2.2 falls back to LoD1.2 when invalid.

    For units whose LiDAR roof points Roofer reports as unusable, the roof is extruded at
    ``missing_roof_height`` metres above observed ground or, without it, at the unit's
    BDNB height when its groups agree on a single valid value.
    """
    from geogen.reconstruction.evidence import assess_evidence
    from geogen.reconstruction.roofer import reconstruct

    if missing_roof_height is not None and (
        not math.isfinite(missing_roof_height) or not 0 < missing_roof_height <= 300
    ):
        raise ReconstructionError("configuration_invalid", "Invalid assumed roof height")

    units = restore_units(manifest["units"])
    targets = tuple(u for u in units if u.role == "target")
    if not targets:
        raise ReconstructionError("selection_empty", "Manifest contains no targets")
    french = [b for b in restore_buildings(manifest["buildings"]).values() if b.country == "FR"]
    selected = {getattr(b, "batiment_groupe_id", b.code) for b in french}
    bdnb_heights = {getattr(b, "batiment_groupe_id", b.code): b.height for b in french}
    represented = {group for unit in targets for group in unit.identity.bdnb_group_ids}
    if represented != selected or any(
        not unit.identity.bdnb_group_ids or not set(unit.identity.bdnb_group_ids) <= selected
        for unit in targets
    ):
        raise ReconstructionError(
            "selection_mismatch", "Manifest must account for every selected French group"
        )
    frame = read_frame(manifest["frame"])
    footprints, trims = local_footprints(targets, frame)
    original = checked_artifact(manifest_path.parent, manifest["artifacts"]["lidar_original"])
    working = work / "lidar_working.laz"
    point_report = prepare_points(original, working)
    report["point_preparation"] = point_report
    if "lidar_missions" in manifest:
        report["lidar_missions"] = manifest["lidar_missions"]
    report["frame"] = asdict(frame)
    report["target_count"] = len(targets)
    report["context_count"] = len(units) - len(targets)
    context_contacts = {}
    for target in targets:
        outline = from_wkt(target.working_wkt)
        for context in units:
            if (
                context.role == "context"
                and outline.distance(from_wkt(context.working_wkt)) < 0.001
            ):
                context_contacts.setdefault(target.reconstruction_id, []).append(
                    context.reconstruction_id
                )
    report["ground_references"] = {}
    grounds = ground_references(
        working, units, diagnostics=report["ground_references"], allow_missing=True
    )
    solids = []
    failed = []
    for unit in targets:
        result = {
            "reconstruction_id": unit.reconstruction_id,
            "geometry_status": "not_run",
            "association_status": (
                "verified"
                if unit.links
                and all(
                    l.confidence != "unresolved" and l.relationship != "unresolved"
                    for l in unit.links
                )
                else "unresolved"
            ),
            "fidelity_status": "unverified",
            "thermal_status": "not_run",
            "links": [asdict(link) for link in unit.links],
            "metrics": {
                "gross_volume_m3": None,
                "modeled_zone_volume_m3": None,
                "conditioned_volume_m3": None,
            },
            "reason_codes": [],
            "assumptions": [
                "wall_extent_assumption=roofprint_extruded",
                "aggregate_physical_envelope_not_entity_specific_thermal_allocation",
            ],
            "attempted_candidates": [],
            "requires_review": False,
            "adaptation": {"max_face_deviation_m": MAX_FACE_DEVIATION, "changed": False},
        }
        if unit.reconstruction_id in context_contacts:
            result["context_contacts"] = context_contacts[unit.reconstruction_id]
            result["requires_review"] = True
            result["reason_codes"].append("context_boundary_unresolved")
            result["assumptions"].append("unmodeled_context_contact_treated_as_exterior_geometry")
        if unit.reconstruction_id in trims:
            result["outline_trim"] = trims[unit.reconstruction_id]
            result["requires_review"] = True
            result["reason_codes"].append("overlapping_outline_trimmed")
        if unit.reconstruction_id in report.get("lidar_missions", {}).get("straddling_targets", []):
            result["requires_review"] = True
            result["reason_codes"].append("mixed_lidar_missions")
            result["assumptions"].append(
                "points_from_lidar_missions_"
                f"{report['lidar_missions']['acquisition_span_days']}_days_apart"
            )
        ground = report["ground_references"][unit.reconstruction_id]
        if ground["expanded_search"]:
            result["requires_review"] = True
            result["reason_codes"].append("ground_search_expanded")
            result["assumptions"].append(
                f"ground_from_expanded_{ground['search_radius_m']:g}m_neighbourhood"
            )
        report["units"].append(result)
        directory = work / content_hash(unit.reconstruction_id)[:20]
        directory.mkdir()
        try:
            if unit.reconstruction_id not in grounds:
                raise ReconstructionError(
                    "ground_unverified",
                    f"No sufficient local ground within {GROUND_SEARCH_RADIUS:g} m for "
                    f"{unit.reconstruction_id}; inspect ground_references",
                )
            outline = write_json(
                directory / "reconstruction_units.geojson",
                {
                    "type": "FeatureCollection",
                    "crs": {"type": "name", "properties": {"name": "EPSG:2154"}},
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": mapping(from_wkt(unit.working_wkt)),
                            "properties": {
                                "reconstruction_id": unit.reconstruction_id,
                                "ground_elevation": grounds[unit.reconstruction_id],
                            },
                        }
                    ],
                },
            )
            local_points = directory / "lidar_working.laz"
            shutil.copyfile(working, local_points)
            configuration = {
                "source_manifest": content_hash(
                    {
                        **manifest,
                        "artifacts": {
                            name: entry["sha256"] for name, entry in manifest["artifacts"].items()
                        },
                    }
                ),
                "outline": asdict(unit),
                "points": file_hash(working),
                "frame": asdict(frame),
                "class67": "include",
                "image": runtime.image,
                "complexity": None,
                "ground": grounds[unit.reconstruction_id],
                "terrain_strategy": "user",
                "clip_terrain": False,
                "parser_schema": "1.0",
                "validation_schema": "1.0",
            }
            raw = cache.get(configuration)
            if raw is None:
                raw = reconstruct(local_points, outline, directory, runtime)
                cache.put(configuration, raw)
            else:
                copy = directory / "cached.city.jsonl"
                shutil.copyfile(raw, copy)
                raw = copy
            result["attempted_candidates"].append(
                {
                    "lod": "2.2",
                    "artifact": artifact(raw, work),
                    "selection_reason": "pending_geometry_validation",
                }
            )
            result["diagnostics"] = {}
            diagnostics = roofer_diagnostics(
                raw, unit.reconstruction_id, diagnostics=result["diagnostics"]
            )
            result["attempted_candidates"][-1]["diagnostics"] = dict(diagnostics)
            assumed = _assumed_height(unit, missing_roof_height, bdnb_heights)
            height_retry = (
                assumed is not None
                and diagnostics["rf_extrusion_mode"] == "skip"
                and diagnostics["rf_success"] is True
                and diagnostics["rf_pointcloud_unusable"] is True
            )
            if height_retry:
                initial = result["attempted_candidates"][-1]
                initial["selection_reason"] = "rejected"
                initial["failure"] = {
                    "code": "runtime_failed",
                    "message": "Roofer skipped unusable points; assumed-height retry",
                }
                elevation = grounds[unit.reconstruction_id] + assumed["height_m"]
                result["height_fallback"] = {
                    **assumed,
                    "ground_elevation_m": grounds[unit.reconstruction_id],
                    "roof_elevation_m": elevation,
                    "roof_elevation_tolerance_m": 0.011,
                    "original_diagnostics": dict(diagnostics),
                }
                retry_outline = json.loads(outline.read_text())
                retry_outline["features"][0]["properties"]["roof_elevation"] = elevation
                retry_outline = write_json(directory / "assumed-height.geojson", retry_outline)
                retry_configuration = {
                    **configuration,
                    "roof_elevation": elevation,
                    "height_fallback": f"{assumed['source']}_assumed_height_v1",
                }
                raw = cache.get(retry_configuration)
                if raw is None:
                    raw = reconstruct(local_points, retry_outline, directory, runtime)
                    cache.put(retry_configuration, raw)
                else:
                    copied = directory / "cached-assumed-height.city.jsonl"
                    shutil.copyfile(raw, copied)
                    raw = copied
                result["attempted_candidates"].append(
                    {
                            "lod": "1.2",
                        "artifact": artifact(raw, work),
                        "height_fallback": result["height_fallback"],
                        "selection_reason": "pending_geometry_validation",
                    }
                )
                result["diagnostics"] = {}
                diagnostics = roofer_diagnostics(
                    raw, unit.reconstruction_id, diagnostics=result["diagnostics"]
                )
                result["attempted_candidates"][-1]["diagnostics"] = dict(diagnostics)
                if diagnostics["rf_extrusion_mode"] != "lod11_fallback":
                    raise ReconstructionError(
                        "runtime_failed", "Assumed-height retry did not produce a Roofer extrusion"
                    )
            if (
                diagnostics["rf_extrusion_mode"] in {"skip", "fail"}
                or diagnostics["rf_success"] is not True
                or (diagnostics["rf_pointcloud_unusable"] is not False and not height_retry)
            ):
                raise ReconstructionError(
                    "runtime_failed",
                    "Roofer diagnostics do not confirm a usable reconstruction: "
                    f"rf_extrusion_mode={diagnostics['rf_extrusion_mode']!r}, "
                    f"rf_success={diagnostics['rf_success']!r}, "
                    f"rf_pointcloud_unusable={diagnostics['rf_pointcloud_unusable']!r}",
                )
            extrusion = (
                diagnostics["rf_force_lod11"]
                or diagnostics["rf_extrusion_mode"] == "lod11_fallback"
            )
            # Raw Roofer validity is diagnostic only: accepted geometry is rebuilt and
            # validated separately, but validator execution failures stay blocking.
            try:
                result["raw_validation"] = runtime.validate_cityjson(raw, directory)
            except ReconstructionError as error:
                if error.code != "validator_invalid":
                    raise
                result["raw_validation"] = {
                    "validity": False,
                    "reason_code": error.code,
                    "message": str(error),
                }
            selected_lod = "2.2"
            candidate = result["attempted_candidates"][-1]
            try:
                if extrusion and not height_retry:
                    raise ReconstructionError(
                        "fallback_lod1",
                        "Roofer returned an extrusion fallback, not accepted LoD2.2",
                    )
                scene = _geometry_candidate(
                    raw,
                    frame,
                    unit,
                    directory,
                    work,
                    runtime,
                    result,
                    candidate,
                    lod="1.2" if height_retry else "2.2",
                    footprint=footprints[unit.reconstruction_id],
                    ground=grounds[unit.reconstruction_id],
                )
                if height_retry:
                    selected_lod = "1.2"
            except ReconstructionError as error:
                candidate["failure"] = {"code": error.code, "message": str(error)}
                candidate["selection_reason"] = "rejected"
                if height_retry or error.code not in LOD_FALLBACK_REASONS:
                    raise
                result["lod_fallback"] = {
                    "requested_lod": "2.2",
                    "fallback_lod": "1.2",
                    "reason": candidate["failure"],
                }
                candidate = {
                    "lod": "1.2",
                    "artifact": artifact(raw, work),
                    "selection_reason": "pending_geometry_validation",
                }
                result["attempted_candidates"].append(candidate)
                try:
                    scene = _geometry_candidate(
                        raw,
                        frame,
                        unit,
                        directory,
                        work,
                        runtime,
                        result,
                        candidate,
                        lod="1.2",
                        footprint=footprints[unit.reconstruction_id],
                        ground=grounds[unit.reconstruction_id],
                    )
                except ReconstructionError as fallback_error:
                    candidate["failure"] = {
                        "code": fallback_error.code,
                        "message": str(fallback_error),
                    }
                    candidate["selection_reason"] = "rejected"
                    raise
                selected_lod = "1.2"
                result["requires_review"] = True
                result["reason_codes"].append("roofer_lod12_fallback")
                result["assumptions"].append(
                    "roofer_lod12_envelope_not_detailed_roof_reconstruction"
                )
            scene = replace(scene, solids=tuple(replace(s, links=unit.links) for s in scene.solids))
            if height_retry:
                roof_z = result["height_fallback"]["roof_elevation_m"] - frame.origin[2]
                roof_points = [
                    point
                    for solid in scene.solids
                    for face in solid.surfaces
                    if face.semantic == "RoofSurface"
                    for ring in (face.exterior, *face.interiors)
                    for point in ring
                ]
                if not roof_points or any(
                    abs(point[2] - roof_z) > result["height_fallback"]["roof_elevation_tolerance_m"]
                    for point in roof_points
                ):
                    raise ReconstructionError(
                        "height_fallback_mismatch",
                        "Roofer output does not match the assumed elevation",
                    )
                scene = replace(
                    scene,
                    solids=tuple(
                        replace(
                            solid,
                            surfaces=tuple(
                                replace(
                                    face,
                                    provenance=(*face.provenance, assumed["reason"]),
                                )
                                for face in solid.surfaces
                            ),
                        )
                        for solid in scene.solids
                    ),
                )
                result["assumptions"].append(
                    f"{assumed['reason']}_m={assumed['height_m']:g};not_lidar_derived"
                )
            result["selected_lod"] = selected_lod
            result["geometry_status"] = "fallback_lod1" if selected_lod == "1.2" else "accepted"
            result["metrics"]["gross_volume_m3"] = sum(s.volume for s in scene.solids)
            evidence = assess_evidence(scene, (unit,), working)[unit.reconstruction_id]
            if height_retry:
                evidence["fidelity_status"] = "needs_review"
                evidence.setdefault("reasons", []).append(assumed["reason"])
            result["fidelity_status"] = evidence["fidelity_status"]
            result["metrics"].update(evidence.get("metrics", {}))
            result["evidence_method"] = evidence.get("method")
            result["reason_codes"].extend(evidence.get("reasons", []))
            if result["fidelity_status"] != "evidence_passed":
                result["requires_review"] = True
                blocking = set(evidence.get("reasons", [])) - REPORTABLE_EVIDENCE_REASONS
                if blocking or not evidence.get("reasons"):
                    raise ReconstructionError("needs_review", "LiDAR evidence gates require review")
                result["reason_codes"].append("evidence_review_permitted")
            candidate["selection_reason"] = (
                "valid_geometry_with_reported_review"
                if result["requires_review"]
                else "geometry_and_evidence_passed"
            )
            result["selected_candidate"] = len(result["attempted_candidates"]) - 1
            solids.extend(scene.solids)
            write_json(directory / "normalized.json", asdict(scene))
        except ReconstructionError as error:
            result["failure"] = {"code": error.code, "message": str(error)}
            if result["attempted_candidates"]:
                candidate = result["attempted_candidates"][-1]
                candidate["selection_reason"] = "rejected"
                candidate.setdefault("failure", result["failure"])
            if result["geometry_status"] not in {"accepted", "fallback_lod1"}:
                result["geometry_status"] = (
                    "fallback_lod1"
                    if error.code == "fallback_lod1"
                    else (
                        "unsupported_geometry"
                        if error.code.startswith("unsupported")
                        else (
                            "runtime_failed"
                            if error.code.startswith(("runtime", "roofer"))
                            else "geometry_invalid"
                        )
                    )
                )
            result["reason_codes"].append(error.code)
            failed.append(unit.reconstruction_id)
        write_json(directory / "result.json", result)
    report["failed_units"] = failed
    if failed:
        raise ReconstructionError(
            "target_failed", "At least one selected Roofer unit failed; no combined model published"
        )
    return GeometryScene(frame, tuple(solids)), targets
