"""Roofer/LiDAR CLI orchestration; legacy execution remains in :mod:`geogen.cli`."""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import asdict
from pathlib import Path

import click

from geogen.metadata import portfolio_metadata
from geogen.models import ProviderError
from geogen.osm import ModelError, model_file_name, model_file_suffix, model_name
from geogen.providers import discover_country
from geogen.reconstruction.cache import (
    ReconstructionCache,
    building_snapshot,
    content_hash,
    load_manifest,
    restore_buildings,
    stage_manifest,
    write_json,
)
from geogen.reconstruction.pipeline import reconstruct_manifest
from geogen.reconstruction.publish import OutputTransaction
from geogen.reconstruction.types import (
    GEOGEN_BASE_COMMIT,
    SCHEMA_VERSION,
    ReconstructionError,
)
from geogen.svg import SvgError


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Expected an object")
        return data
    except (OSError, ValueError) as error:
        raise ReconstructionError("configuration_invalid", f"Invalid JSON file: {path}") from error


def _protect_inputs(destinations, inputs):
    if any(
        source.resolve() == destination.resolve()
        for source in inputs
        if source
        for destination in destinations
    ):
        raise ReconstructionError(
            "output_conflict", "An output would replace a reconstruction input"
        )


def run(options: dict, downloader) -> None:
    """Reconstruct French roofs from LiDAR, always retaining diagnostics on failure."""
    addresses = tuple(options["addresses"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "experimental": True,
        "geogen_base_commit": GEOGEN_BASE_COMMIT,
        "requested_backend": "roofer",
        "benchmark_status": "not_evaluated_no_independent_references",
        "energyplus_harness_status": "not_run",
        "units": [],
        "reason_codes": [],
        "outputs_published": False,
    }
    output = options["output"]
    manifest_path = options["reconstruction_input"]
    report_path = (
        output.with_suffix(".reconstruction.json") if output else Path("reconstruction-report.json")
    )
    destinations = [
        p for p in (output, options["svg_output"], options["json_output"], report_path) if p
    ]
    try:
        OutputTransaction(destinations)
        _protect_inputs(destinations, [manifest_path])
    except ReconstructionError as error:
        raise click.ClickException(str(error)) from error
    work = None
    try:
        if any(discover_country(address, options["country"]) != "FR" for address in addresses):
            raise ReconstructionError("unsupported_country", "Roofer supports French units only")
        height = options["roofer_missing_roof_height"]
        if height is not None and not math.isfinite(height):
            raise ReconstructionError("configuration_invalid", "Assumed roof height must be finite")
        report["geometry_policy"] = {"missing_roof_height_m": height}
        cache_root = options["reconstruction_cache"]
        request = {
            "addresses": list(addresses),
            "country": options["country"],
            "max_buildings": options["max_buildings"],
            "provider_endpoint_hash": content_hash(options["base_url"]),
        }
        manifest = None
        if manifest_path:
            unverified_manifest = _read_json(manifest_path)
            artifact_records = unverified_manifest.get("artifacts", {})
            if isinstance(artifact_records, dict):
                _protect_inputs(
                    destinations,
                    [
                        manifest_path.parent / entry["path"]
                        for entry in artifact_records.values()
                        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
                    ],
                )
            manifest = load_manifest(manifest_path, addresses=addresses)
            if manifest["request"] != request:
                raise ReconstructionError(
                    "selection_mismatch",
                    "Replay selection or provider edition differs from this request",
                )
            buildings = restore_buildings(manifest["buildings"])
        else:
            buildings = downloader(
                addresses,
                options["api_key"],
                options["base_url"],
                options["timeout"],
                options["max_buildings"],
                country=options["country"],
                os_api_key=options["os_api_key"],
                os_building_collection=options["os_building_collection"],
                strict=True,
            )
        if not buildings:
            raise ReconstructionError(
                "selection_empty", "No buildings found for the requested selection"
            )
        if any(b.country != "FR" for b in buildings.values()):
            raise ReconstructionError("unsupported_country", "Roofer supports French units only")
        building_name = options["name"] or model_name(b.code for b in buildings.values())
        output = output or Path(
            model_file_name(building_name, model_file_suffix(options["output_format"]))
        )
        report_path = output.with_suffix(".reconstruction.json")
        destinations = [
            p for p in (output, options["svg_output"], options["json_output"], report_path) if p
        ]
        OutputTransaction(destinations)
        _protect_inputs(destinations, [manifest_path])
        cache_root.mkdir(parents=True, exist_ok=True)
        runs = cache_root / "runs"
        runs.mkdir(exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="run-", dir=runs)).resolve()
        if manifest is not None:
            manifest, manifest_path = stage_manifest(manifest, manifest_path, work)
        report["replay_directory"] = str(work)
        report["source_buildings"] = building_snapshot(buildings)
        report["selected_source_ids"] = list(buildings)
        from geogen.reconstruction.enrichment import (
            enrich_thermal_scene,
            glazing_parameters,
        )
        from geogen.reconstruction.runtime import DockerRuntime

        # A replayed manifest is complete: never download data or pull images.
        runtime = DockerRuntime(offline=manifest is not None)
        report["runtime"] = runtime.preflight(work)
        if manifest is None:
            from geogen.reconstruction.acquire import acquire

            manifest_path = acquire(buildings, request, options, work, runtime)
            manifest = load_manifest(manifest_path, addresses=addresses)
        scene, _ = reconstruct_manifest(
            manifest,
            manifest_path,
            work,
            runtime,
            cache=ReconstructionCache(cache_root / "geometry"),
            report=report,
            missing_roof_height=height,
        )
        report["phase"] = "thermal"
        thermal, envelope_properties, groups_by_solid = enrich_thermal_scene(
            scene, buildings, report, storey_height=options["storey_height"]
        )
        report["thermal_assumptions"] = list(thermal.assumptions)
        report["scene_geometry_hash"] = thermal.geometry_hash
        for result in report["units"]:
            result["thermal_status"] = "geometry_only"
            solid_ids = {
                s.solid_id
                for s in scene.solids
                if s.reconstruction_id == result["reconstruction_id"]
            }
            result["metrics"]["modeled_zone_volume_m3"] = sum(
                s.volume for s in thermal.spaces if s.solid_id in solid_ids
            )
        reviewed = sum(r.get("requires_review", False) for r in report["units"])
        lowered = sum(r.get("selected_lod") == "1.2" for r in report["units"])
        assumed = sum("height_fallback" in r for r in report["units"])
        if assumed:
            click.echo(
                f"Roofer: {assumed} unit(s) use an assumed (user or BDNB) roof height, not a "
                f"LiDAR-derived roof. Details: {report_path}",
                err=True,
            )
        if lowered:
            click.echo(
                f"Roofer: {lowered} unit(s) use the LoD1.2 fallback. Details: {report_path}",
                err=True,
            )
        if reviewed:
            click.echo(
                f"Roofer: {reviewed} unit(s) require review. Details: {report_path}", err=True
            )
        context = click.get_current_context()
        ignored = ["roof_pitch", "max_roof_height", "simplify_tolerance"]
        if any(
            context.get_parameter_source(key) == click.core.ParameterSource.COMMANDLINE
            for key in ignored
        ):
            click.echo(
                "Legacy roof pitch, height cap and simplify tolerance do not "
                "modify Roofer geometry.",
                err=True,
            )
        write_json(work / "thermal-scene.json", asdict(thermal))
        from geogen.osm import (
            build_model_from_geometry,
            save_model_from_geometry,
            windows_from_geometry_model,
        )
        from geogen.svg import save_svg_from_geometry

        report["phase"] = "export"
        ratio = options["window_to_wall_ratio"]
        if ratio is not None and ratio > 1:
            ratio /= 100
        window_ratios, window_orientations = glazing_parameters(
            thermal, buildings, report, groups_by_solid
        )
        model = build_model_from_geometry(
            thermal,
            building_name=building_name,
            window_to_wall_ratio=ratio,
            window_ratios=window_ratios,
            window_orientations=window_orientations,
            envelope_properties=envelope_properties,
        )
        enrichment = (
            model.getBuilding()
            .additionalProperties()
            .getFeatureAsString("geogen_thermal_enrichment")
        )
        if enrichment.is_initialized():
            report["envelope_enrichment"] = json.loads(enrichment.get())
        glazing_report = (
            model.getBuilding().additionalProperties().getFeatureAsString("geogen_glazing_report")
        )
        if glazing_report.is_initialized():
            report["glazing_export"] = json.loads(glazing_report.get())
            omitted = sum(
                len(space.get("omitted_windows", [])) for space in report["glazing_export"].values()
            )
            if omitted:
                click.echo(
                    f"Requested windows omitted on {omitted} wall(s); details: {report_path}",
                    err=True,
                )
        report["glazing_assumptions"].append({"default_requested_wwr": ratio})
        model_windows = windows_from_geometry_model(model)
        report["window_count"] = len(model.getSubSurfaces())
        report["outputs_published"] = True
        report["export_status"] = "geometry_only"
        metadata = portfolio_metadata(buildings, source_addresses=addresses)
        metadata["reconstruction"] = report
        with OutputTransaction(destinations) as transaction:
            save_model_from_geometry(
                model, transaction.stage(output), output_format=options["output_format"]
            )
            if options["svg_output"]:
                save_svg_from_geometry(
                    thermal,
                    transaction.stage(options["svg_output"]),
                    width=options["svg_width"],
                    height=options["svg_height"],
                    azimuth=options["svg_azimuth"],
                    elevation=options["svg_elevation"],
                    windows=model_windows,
                )
            if options["json_output"]:
                write_json(transaction.stage(options["json_output"]), metadata)
            write_json(transaction.stage(report_path), report)
            transaction.commit()
        click.echo(
            f"Wrote {output} with {len(model.getSpaces())} space(s); reconstruction report {report_path}"
        )
    except (
        ReconstructionError,
        ProviderError,
        ModelError,
        SvgError,
        OSError,
        ValueError,
        KeyError,
        click.ClickException,
    ) as error:
        report["outputs_published"] = False
        report["reason_codes"].append(getattr(error, "code", "export_or_input_failed"))
        if report.get("phase") in {"thermal", "export"} or isinstance(
            error, (ModelError, SvgError)
        ):
            report["export_status"] = "thermal_export_failed"
            for unit in report["units"]:
                unit["thermal_status"] = "thermal_export_failed"
        if not report["units"]:
            report["unresolved_sources"] = [
                {
                    "source_id": source_id,
                    "status": "not_exported",
                    "reason_codes": report["reason_codes"],
                }
                for source_id in report.get("selected_source_ids", [])
            ]
        if getattr(error, "code", None) != "output_conflict":
            write_json(report_path, report)
        raise click.ClickException(f"{error} (diagnostics: {report_path})") from error
    finally:
        if work is not None:
            write_json(work / "report.json", report)
