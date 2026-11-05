"""Opt-in real Docker/CLI smoke tests; synthetic geometry is not an IGN benchmark.

Run with GEOGEN_RUNTIME_TESTS=1 and the locked image already present locally.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from click.testing import CliRunner
import laspy
import numpy as np
import openstudio
from pyproj import CRS
import pytest
from shapely import contains_xy
from shapely.geometry import Polygon, box, mapping
from shapely.ops import unary_union

from geogen.bdnb import DEFAULT_BASE_URL
from geogen.cli import main
from geogen.models import BuildingGroup
from geogen.reconstruction.cache import (
    artifact,
    building_snapshot,
    content_hash,
    file_hash,
    write_json,
)
from geogen.reconstruction.ign import build_pdal_pipeline
from geogen.reconstruction.pipeline import prepare_points
from geogen.reconstruction.runtime import DockerRuntime
from geogen.reconstruction.types import (
    SCHEMA_VERSION,
    GeometryFrame,
    ReconstructionUnit,
    SourceIdentity,
    SourceLink,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("GEOGEN_RUNTIME_TESTS") != "1",
    reason="Set GEOGEN_RUNTIME_TESTS=1 to use the locally cached pinned Docker image",
)

ADDRESS = "Synthetic reconstruction test, France"
FRAME = GeometryFrame("EPSG:2154", "IGN69", (700000.0, 6600000.0, 100.0))
FIXTURES = Path(__file__).parent / "fixtures" / "reconstruction"


def _manifest(directory: Path, *, oversized: bool = False) -> tuple[Path, Path, dict]:
    directory.mkdir(parents=True)
    if oversized:
        shapes = {
            "oversized": Polygon([
                (700000, 6600000), (700300, 6600000),
                (700300, 6600300), (700000, 6600300),
            ]),
        }
    else:
        shapes = {
            "box": Polygon([
                (700000, 6600000), (700010, 6600000),
                (700010, 6600010), (700000, 6600010),
            ]),
            "courtyard": Polygon(
                [(700020, 6600000), (700040, 6600000),
                 (700040, 6600020), (700020, 6600020)],
                holes=[[(700026, 6600006), (700026, 6600014),
                        (700034, 6600014), (700034, 6600006)]],
            ),
        }
    footprint = unary_union(list(shapes.values()))
    xmin, ymin, xmax, ymax = footprint.bounds
    x, y = np.meshgrid(
        np.arange(xmin - 3.75, xmax + 4, 0.5),
        np.arange(ymin - 3.75, ymax + 4, 0.5),
    )
    x, y = x.ravel(), y.ravel()
    roof = contains_xy(footprint, x, y)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array(FRAME.origin)
    header.add_crs(CRS.from_epsg(2154))
    cloud = laspy.LasData(header)
    cloud.x, cloud.y = x, y
    cloud.z = np.where(roof, 110.0, 100.0)
    cloud.classification = np.where(roof, 6, 2).astype(np.uint8)
    original = directory / "classified.original.laz"
    cloud.write(original)

    buildings, units = {}, []
    for name, polygon in shapes.items():
        group_id = f"synthetic-group-{name}"
        buildings[f"FR:{group_id}"] = BuildingGroup(
            batiment_groupe_id=group_id, crs="EPSG:2154", geometry=polygon,
            height=10.0, ground_elevation=100.0, storeys=1,
            footprint_area=polygon.area, fictitious_geometry=False, address=ADDRESS,
        )
        link = SourceLink(
            reconstruction_id=name, source_type="bdnb_group", source_id=group_id,
            method="synthetic_fixture_identity", relationship="group_membership",
            attribute_scope="group_only", confidence="verified",
            source_dates=("synthetic-no-acquisition-date",),
        )
        units.append(ReconstructionUnit(
            reconstruction_id=name,
            identity=SourceIdentity(
                bdnb_group_ids=(group_id,), bdtopo_ids=(f"synthetic-physical-{name}",),
                dataset_versions=(("synthetic", "1"),),
            ),
            original_wkt=polygon.wkt, working_wkt=polygon.wkt,
            geometry_revision=content_hash(polygon.wkt), outline_type="roofprint",
            decisions=("synthetic_geometry_not_a_real_building",), links=(link,),
        ))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "request": {
            "addresses": [ADDRESS], "country": "FR", "max_buildings": 10,
            "provider_endpoint_hash": content_hash(DEFAULT_BASE_URL),
        },
        "buildings": building_snapshot(buildings),
        "units": [asdict(unit) for unit in units],
        "frame": asdict(FRAME),
        "artifacts": {"lidar_original": artifact(original, directory)},
    }
    path = write_json(directory / "manifest.json", manifest)
    return path, original, shapes





def _invoke(manifest: Path, root: Path, *, name: str, output_format="osm", extra_args=()):
    output = root / f"{name}.{output_format}"
    svg = root / f"{name}.svg"
    metadata = root / f"{name}.json"
    report_path = root / f"{name}.reconstruction.json"
    result = CliRunner().invoke(main, [
        ADDRESS, "--country", "FR", "--base-url", DEFAULT_BASE_URL,
        "--reconstruction-input", str(manifest),
        "--roof-source", "roofer", "--reconstruction-cache", str(root / "cache"),
        "--output", str(output),
        "--svg-output", str(svg), "--json-output", str(metadata),
        "--output-format", output_format,
        "--window-to-wall-ratio", ".2", *extra_args,
    ])
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    return result, report, output, svg, metadata


def _assert_exports(report, output, svg, metadata):
    assert report["outputs_published"] is True
    assert report["export_status"] == "geometry_only"
    assert report["benchmark_status"] == "not_evaluated_no_independent_references"
    assert report["energyplus_harness_status"] == "not_run"
    assert report["frame"] == json.loads(json.dumps(asdict(FRAME)))
    assert {unit["reconstruction_id"] for unit in report["units"]} == {"box", "courtyard"}
    for unit in report["units"]:
        name = unit["reconstruction_id"]
        assert unit["geometry_status"] == "accepted"
        assert unit["fidelity_status"] == "evidence_passed"
        assert unit["thermal_status"] == "geometry_only"
        assert unit["diagnostics"]["rf_extrusion_mode"] == "standard"
        assert unit["independent_validation"]["validity"] is True
        assert Path(unit["independent_validation"]["report_path"]).is_file()
        assert unit["metrics"]["gross_volume_m3"] == pytest.approx(
            1000.0 if name == "box" else 3360.0
        )
        assert unit["metrics"]["conditioned_volume_m3"] is None
        assert unit["links"][0]["source_id"] == f"synthetic-group-{name}"
        assert unit["evidence_method"]["coverage_cell_size_m"] == 0.5
        ground = report["ground_references"][name]
        assert ground["elevation_m"] == 100.0
        assert ground["method"] == "local_ground_5th_percentile_3m_buffer"
        assert not ground["expanded_search"]
    work = Path(report["replay_directory"])
    scene = json.loads((work / "thermal-scene.json").read_text())
    shared_hash = report["scene_geometry_hash"]
    assert scene["geometry_hash"] == shared_hash
    assert json.loads(metadata.read_text())["reconstruction"]["scene_geometry_hash"] == shared_hash
    root = ET.parse(svg).getroot()
    assert root.attrib["data-geometry-hash"] == shared_hash
    svg_metadata = json.loads(root.find("{http://www.w3.org/2000/svg}metadata").text)
    assert svg_metadata["frame"] == report["frame"]
    assert svg_metadata["fidelity_status"] == "unverified"
    assert svg_metadata["window_count"] == report["window_count"] > 0
    assert svg_metadata["window_source"] == "actual_model_geometry"
    assert root.findall(".//*[@class='window']")
    loaded = openstudio.osversion.VersionTranslator().loadModel(openstudio.path(str(output)))
    assert loaded.is_initialized()
    model = loaded.get()
    assert len(model.getSpaces()) == 2
    assert model.getSite().elevation() == 100.0
    properties = model.getBuilding().additionalProperties()
    assert properties.getFeatureAsString("geogen_geometry_hash").get() == shared_hash
    assert properties.getFeatureAsString("geogen_fidelity_status").get() == "experimental_unverified"
    assert sum(space.volume() for space in model.getSpaces()) == pytest.approx(4360.0)
    assert len(model.getSubSurfaces()) == report["window_count"]
    window_area = sum(window.grossArea() for window in model.getSubSurfaces())
    exterior_wall_area = sum(
        wall.grossArea() for wall in model.getSurfaces()
        if wall.surfaceType() == "Wall" and wall.outsideBoundaryCondition() == "Outdoors"
    )
    assert window_area == pytest.approx(exterior_wall_area * 0.2)
    source_objects = {
        surface.additionalProperties().getFeatureAsString("geogen_source_object_id").get()
        for surface in model.getSurfaces()
    }
    assert source_objects == {"box-0", "courtyard-0"}
    return work


def test_published_usage_example(tmp_path, offline_runtime):
    manifest = Path(__file__).resolve().parents[1] / "examples" / "roofer" / "manifest.json"
    original = manifest.parent / "classified.original.laz"
    checksum = file_hash(original)
    result, report, output, svg, metadata = _invoke(manifest, tmp_path, name="example")
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    _assert_exports(report, output, svg, metadata)
    assert file_hash(original) == checksum


def test_native_roofer_enrichment_exports_storeys_materials_and_windows(tmp_path, offline_runtime):
    manifest, original, _ = _manifest(tmp_path / "input")
    checksum = file_hash(original)
    data = json.loads(manifest.read_text())
    for building, count in zip(data["buildings"], (2, 3)):
        building["data"].update(
            storeys=count,
            glazed_ratio=0.25,
            wall_u_value=0.4,
            roof_u_value=0.25,
            floor_u_value=0.5,
            window_u_value=1.6,
            window_solar_factor=0.6,
        )
    write_json(manifest, data)
    result, report, output, svg, metadata = _invoke(manifest, tmp_path, name="enriched")
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    assert report["outputs_published"]
    assert all(unit["independent_validation"]["validity"] for unit in report["units"])
    assert sorted(d["modeled_space_count"] for d in report["storey_enrichment"].values()) == [2, 3]
    assert all(
        d["partition"]["status"] == "partitioned" for d in report["storey_enrichment"].values()
    )
    model = openstudio.osversion.VersionTranslator().loadModel(openstudio.path(str(output))).get()
    assert len(model.getBuildingStorys()) == len(model.getSpaces()) == 5
    assert len(model.getThermalZones()) == 5
    assert all(space.buildingStory().is_initialized() for space in model.getSpaces())
    assert sum(space.volume() for space in model.getSpaces()) == pytest.approx(4360)
    for surface in model.getSurfaces():
        assert surface.construction().is_initialized()
        if surface.outsideBoundaryCondition() == "Surface":
            adjacent = surface.adjacentSurface().get()
            assert adjacent.adjacentSurface().get().handle() == surface.handle()
    assert report["window_count"] == len(model.getSubSurfaces()) > 0
    for window in model.getSubSurfaces():
        layer = window.construction().get().to_Construction().get().layers()[0]
        assert layer.to_SimpleGlazing().get().uFactor() == pytest.approx(1.6)
    exterior_area = sum(
        surface.grossArea()
        for surface in model.getSurfaces()
        if surface.surfaceType() == "Wall" and surface.outsideBoundaryCondition() == "Outdoors"
    )
    assert sum(w.grossArea() for w in model.getSubSurfaces()) == pytest.approx(exterior_area * 0.25)
    shared_hash = report["scene_geometry_hash"]
    assert ET.parse(svg).getroot().attrib["data-geometry-hash"] == shared_hash
    assert json.loads(metadata.read_text())["reconstruction"]["scene_geometry_hash"] == shared_hash
    calls_before_replay = len(offline_runtime)
    result, replay, idf, _, _ = _invoke(
        manifest, tmp_path, name="enriched-idf", output_format="idf"
    )
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    assert replay["scene_geometry_hash"] == shared_hash
    assert not any(
        call["args"][0] == "roofer" and call["args"][1] not in {"--help", "--version"}
        for call in offline_runtime[calls_before_replay:]
    )
    assert "BuildingSurface:Detailed," in idf.read_text()
    assert "WindowMaterial:SimpleGlazingSystem," in idf.read_text()
    assert file_hash(original) == checksum


def test_offline_cli_real_box_courtyard_and_cached_replay(tmp_path, offline_runtime):
    manifest, original, shapes = _manifest(tmp_path / "input")
    original_hash = file_hash(original)
    result, report, output, svg, metadata = _invoke(manifest, tmp_path, name="first")
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}\n{json.dumps(report, indent=2)}"
    first_work = _assert_exports(report, output, svg, metadata)
    assert file_hash(original) == original_hash
    working = laspy.read(first_work / "lidar_working.laz")
    assert set(working.classification) == {2, 6}
    first_validations = {
        unit["independent_validation"]["report_path"] for unit in report["units"]
    }
    calls_before_replay = len(offline_runtime)
    second, replay, output, svg, metadata = _invoke(manifest, tmp_path, name="replay")
    assert second.exit_code == 0, f"{second.output}\n{second.exception!r}\n{json.dumps(replay, indent=2)}"
    replay_work = _assert_exports(replay, output, svg, metadata)
    assert replay_work != first_work
    assert replay["scene_geometry_hash"] == report["scene_geometry_hash"]
    replay_calls = offline_runtime[calls_before_replay:]
    assert not any(
        call["args"][0] == "roofer" and call["args"][1] not in {"--help", "--version"}
        for call in replay_calls
    )
    actual_validations = [
        call for call in replay_calls
        if call["args"][0] == "val3dity" and "--report" in call["args"]
    ]
    # Each unit validates the cached raw Roofer output and its rebuilt geometry anew.
    validated = sorted(Path(call["args"][1]).name for call in actual_validations)
    assert validated == ["adapted.city.json"] * len(shapes) + ["cached.city.jsonl"] * len(shapes)
    assert not first_validations.intersection(
        unit["independent_validation"]["report_path"] for unit in replay["units"]
    )
    assert file_hash(original) == original_hash


def test_native_two_copc_readers_preserve_original_and_class67(tmp_path, offline_runtime):
    _, source, _ = _manifest(tmp_path / "input")
    base = laspy.read(source)
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = base.header.scales
    header.offsets = base.header.offsets
    header.add_crs(CRS.from_epsg(2154))
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = base.x, base.y, base.z
    classification = np.asarray(base.classification).copy()
    roof_indices = np.flatnonzero(classification == 6)
    classification[roof_indices[::3]] = 67
    cloud.classification = classification
    cloud.add_extra_dim(laspy.ExtraBytesParams(name="SyntheticId", type=np.uint32))
    cloud.SyntheticId = np.arange(len(cloud.points), dtype=np.uint32)
    cloud.z = np.asarray(base.z) + (np.arange(len(cloud.points)) % 7) * .001
    split_x = 700020.0
    cloud.point_source_id = np.where(np.asarray(cloud.x) < split_x, 11, 22).astype(np.uint16)
    bounds = (
        float(np.min(cloud.x)) - .25, float(np.min(cloud.y)) - .25,
        float(np.max(cloud.x)) + .25, float(np.max(cloud.y)) + .25,
    )
    runtime = DockerRuntime(offline=True)
    copcs, tiles = [], []
    for index, side in enumerate(("left", "right")):
        selected = np.asarray(cloud.x) < split_x
        if side == "right":
            selected = ~selected
        part = laspy.LasData(cloud.header.copy())
        part.points = laspy.ScaleAwarePointRecord(
            cloud.points[selected].array.copy(), cloud.header.point_format,
            cloud.header.scales.copy(), cloud.header.offsets.copy(),
        )
        if side == "right":
            part.change_scaling(
                scales=np.array([.0001, .0001, .0001]),
                offsets=np.array([700020.0, 6600000.0, 90.0]),
            )
        assert {2, 6, 67}.issubset(set(part.classification))
        input_path = tmp_path / f"{side}.laz"
        part.write(input_path)
        copc = tmp_path / f"{side}.copc.laz"
        conversion = write_json(tmp_path / f"{side}-to-copc.json", {"pipeline": [
            {"type": "readers.las", "filename": runtime.container_path(input_path, tmp_path)},
            {"type": "writers.copc", "filename": runtime.container_path(copc, tmp_path),
             "forward": "all", "extra_dims": "all",
             **{f"scale_{axis}": float(part.header.scales[i]) for i, axis in enumerate("xyz")},
             **{f"offset_{axis}": float(part.header.offsets[i]) for i, axis in enumerate("xyz")}},
        ]})
        runtime.run(["pdal", "pipeline", runtime.container_path(conversion, tmp_path)], tmp_path)
        with laspy.open(copc) as reader:
            assert any(vlr.user_id == "copc" for vlr in reader.header.vlrs)
            assert reader.header.point_count == int(selected.sum())
            assert np.array_equal(reader.header.scales, part.header.scales)
        copcs.append(copc)
        tile_bounds = (
            bounds[0] if index == 0 else split_x, bounds[1],
            split_x if index == 0 else bounds[2], bounds[3],
        )
        tiles.append({
            "id": f"synthetic-{side}",
            "url": f"https://data.geopf.fr/telechargement/download/LiDARHD/{side}.copc.laz",
            "horizontal_crs": "EPSG:2154", "vertical_reference": "IGN69",
            "acquisition_epoch": "synthetic-same-mission",
            "metadata": {"geometry": mapping(box(*tile_bounds))},
        })
    graph = build_pdal_pipeline(tiles, bounds, "/work/crop.original.laz")
    for writer in (stage for stage in graph["pipeline"] if stage["type"] == "writers.las"):
        assert all(writer[f"scale_{axis}"] == .001 for axis in "xyz")
        assert all(writer[f"offset_{axis}"] == "auto" for axis in "xyz")
    readers = [stage for stage in graph["pipeline"] if stage["type"] == "readers.copc"]
    assert len(readers) == 2
    # Exercise the production graph without fetching real public IGN tiles.
    for reader, copc in zip(readers, copcs):
        reader["filename"] = runtime.container_path(copc, tmp_path)
    pipeline = write_json(tmp_path / "crop.json", graph)
    runtime.run(["pdal", "pipeline", runtime.container_path(pipeline, tmp_path)], tmp_path)
    original_path = tmp_path / "crop.original.laz"
    original_hash = file_hash(original_path)
    original = laspy.read(original_path)
    assert np.array_equal(original.header.scales, [.001, .001, .001])
    assert len(original.points) == len(cloud.points)
    assert set(original.point_source_id) == {11, 22}
    assert set(original.SyntheticId) == set(cloud.SyntheticId)
    original_order = np.argsort(original.SyntheticId)
    assert np.array_equal(original.classification[original_order], classification)
    # PDAL's automatic offsets are not round, so compare well below the 1 mm quantum.
    for axis in "xyz":
        assert np.allclose(
            np.asarray(getattr(original, axis))[original_order],
            np.asarray(getattr(cloud, axis)),
            rtol=0,
            atol=1e-6,
        )
    prepared_path = tmp_path / "prepared.laz"
    preparation = prepare_points(original_path, prepared_path)
    prepared = laspy.read(prepared_path)
    expected = classification.copy()
    expected[expected == 67] = 6
    assert np.array_equal(prepared.classification[np.argsort(prepared.SyntheticId)], expected)
    assert preparation["original_class_counts"]["67"] == int(np.count_nonzero(classification == 67))
    assert file_hash(original_path) == original_hash


def test_offline_cli_idf_with_glazing_preserves_geometry(tmp_path, offline_runtime):
    manifest, _, _ = _manifest(tmp_path / "input")
    result, report, output, svg, metadata = _invoke(
        manifest, tmp_path, name="spaces", output_format="idf"
    )
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}\n{json.dumps(report, indent=2)}"
    assert report["outputs_published"] is True
    assert report["window_count"] > 0
    shared_hash = report["scene_geometry_hash"]
    assert output.read_text().startswith(f"!- geogen_geometry_hash: {shared_hash}\n")
    translated = openstudio.IdfFile.load(openstudio.path(str(output)))
    assert translated.is_initialized()
    idf = translated.get()
    assert len(idf.getObjectsByType(openstudio.IddObjectType("Zone"))) == 2
    assert len(idf.getObjectsByType(
        openstudio.IddObjectType("FenestrationSurface:Detailed")
    )) == report["window_count"]
    scene = json.loads((Path(report["replay_directory"]) / "thermal-scene.json").read_text())
    assert len(scene["spaces"]) == 2
    assert sum(space["volume"] for space in scene["spaces"]) == pytest.approx(4360.0)
    assert scene["geometry_hash"] == shared_hash
    assert json.loads(metadata.read_text())["reconstruction"]["scene_geometry_hash"] == shared_hash
    root = ET.parse(svg).getroot()
    assert root.attrib["data-geometry-hash"] == shared_hash
    svg_metadata = json.loads(root.find("{http://www.w3.org/2000/svg}metadata").text)
    assert svg_metadata["window_count"] == report["window_count"]
    assert svg_metadata["window_source"] == "actual_model_geometry"
