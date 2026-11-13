"""Bounded online acquisition producing the same manifest used by offline replay."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from pyproj import Transformer
from shapely import from_wkt
from shapely.ops import transform, unary_union

from geogen.bdnb import BdnbClient
from geogen.geometry import parse_geometry
from geogen.reconstruction.cache import (
    artifact,
    building_snapshot,
    content_hash,
    write_json,
)
from geogen.reconstruction.ign import (
    IgnClient,
    build_pdal_pipeline,
    checked_bounds,
)
from geogen.reconstruction.pipeline import (
    GROUND_SEARCH_RADIUS,
    ground_references,
    prepare_points,
)
from geogen.reconstruction.reconcile import resolve_units
from geogen.reconstruction.types import (
    SCHEMA_VERSION,
    GeometryFrame,
    ReconstructionError,
)


def acquire(buildings, request, options, work: Path, runtime) -> Path:
    sources = {}
    for building in buildings.values():
        if building.country != "FR":
            continue
        source_id = getattr(building, "batiment_groupe_id", None)
        if not source_id:
            raise ReconstructionError(
                "association_unresolved", "An explicit BDNB group identity is required"
            )
        geometry = building.geometry
        if geometry is not None and not building.fictitious_geometry:
            geometry = parse_geometry(geometry)
            if building.crs != "EPSG:2154":
                if not building.crs:
                    raise ReconstructionError(
                        "reference_unverified", "Provider geometry CRS is unknown"
                    )
                transformer = Transformer.from_crs(building.crs, "EPSG:2154", always_xy=True)
                geometry = transform(transformer.transform, geometry)
        sources[source_id] = replace(building, geometry=geometry, crs="EPSG:2154")
    with BdnbClient(
        options["api_key"], base_url=options["base_url"], timeout=options["timeout"]
    ) as client:
        relations = client.relationship_rows(tuple(sources), include_rnb=True)
    write_json(work / "relations.json", relations)
    with IgnClient(timeout=options["timeout"]) as ign:
        snapshot = ign.buildings_for_sources(sources, relations)
        write_json(work / "bdtopo.json", snapshot)
        units = resolve_units(
            sources,
            relations,
            snapshot,
            dataset_versions={
                "bdtopo_snapshot": content_hash(snapshot),
                "bdnb_relations_snapshot": content_hash(relations),
            },
        )
        # Context neighbours are only used as outlines; points are needed on the
        # targets and in their ground search rings, which may be far apart.
        targets = {u.reconstruction_id: from_wkt(u.working_wkt) for u in units if u.role == "target"}
        footprint = unary_union(list(targets.values()))
        xmin, ymin, xmax, ymax = footprint.bounds
        acquisition_buffer = GROUND_SEARCH_RADIUS
        extraction = footprint.buffer(acquisition_buffer)
        bounds = checked_bounds(
            (
                xmin - acquisition_buffer,
                ymin - acquisition_buffer,
                xmax + acquisition_buffer,
                ymax + acquisition_buffer,
            )
        )
        # Only the missions under the targets are read; their edge may clip the ring.
        tile_snapshot = ign.tiles(bounds, region=extraction, targets=targets)
        extraction = from_wkt(tile_snapshot["extraction_wkt"])
        write_json(work / "tiles.json", tile_snapshot)
        header_provenance = ign.verify_tile_headers(tile_snapshot)
        write_json(work / "tile-headers.json", header_provenance)
    pipeline = build_pdal_pipeline(
        tile_snapshot, bounds, "/work/lidar.original.laz", polygon=extraction
    )
    pipeline_path = write_json(work / "pdal-pipeline.json", pipeline)
    runtime.run(
        ["pdal", "pipeline", runtime.container_path(pipeline_path, work)], work, network=True
    )
    original = work / "lidar.original.laz"
    if not original.is_file() or original.stat().st_size == 0:
        raise ReconstructionError(
            "acquisition_incomplete", "PDAL did not produce the original classified crop"
        )
    working = work / "acquisition-working.laz"
    point_report = prepare_points(original, working)
    ground_diagnostics = {}
    ground = ground_references(
        working, units, diagnostics=ground_diagnostics, allow_missing=True
    )
    frame = GeometryFrame(
        "EPSG:2154",
        tile_snapshot["tiles"][0]["vertical_reference"],
        ((xmin + xmax) / 2, (ymin + ymax) / 2, min(ground.values()) if ground else 0.0),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "request": request,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "buildings": building_snapshot(buildings),
        "units": [asdict(unit) for unit in units],
        "frame": asdict(frame),
        "extraction_bounds": list(bounds),
        "acquisition_buffer_m": acquisition_buffer,
        "lidar_missions": tile_snapshot["mission_selection"],
        "ground_references": ground_diagnostics,
        "frame_origin_z_method": "observed_ground" if ground else "coordinate_origin_only",
        "point_preparation": point_report,
        "artifacts": {
            "lidar_original": artifact(original, work),
            "relations": artifact(work / "relations.json", work),
            "bdtopo": artifact(work / "bdtopo.json", work),
            "tiles": artifact(work / "tiles.json", work),
            "tile_headers": artifact(work / "tile-headers.json", work),
            "pdal_pipeline": artifact(pipeline_path, work),
        },
    }
    return write_json(work / "manifest.json", manifest)
