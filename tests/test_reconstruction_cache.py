"""Replay integrity and multi-output publication do not depend on Docker."""

import json

import pytest

from geogen.models import BuildingGroup
from geogen.reconstruction.cache import (
    ReconstructionCache,
    artifact,
    building_snapshot,
    checked_artifact,
    content_hash,
    load_manifest,
    restore_buildings,
    stage_manifest,
    write_json,
)
from geogen.reconstruction.publish import OutputTransaction
from geogen.reconstruction.types import ReconstructionError


def test_source_snapshot_preserves_compatibility_field_without_retyping():
    source = BuildingGroup("group", batiment_construction_id="rnb-compat", crs="EPSG:2154")
    restored = restore_buildings(building_snapshot({"FR:group": source}))
    assert restored["FR:group"] == source


def test_artifact_checks_checksum_and_confines_paths(tmp_path):
    path = tmp_path / "input.laz"
    path.write_bytes(b"original crop")
    record = artifact(path, tmp_path)
    assert checked_artifact(tmp_path, record) == path
    path.write_bytes(b"truncated")
    with pytest.raises(ReconstructionError, match="corrupt"):
        checked_artifact(tmp_path, record)
    with pytest.raises(ReconstructionError):
        checked_artifact(tmp_path, {"path": "../outside", "sha256": "0" * 64})


def test_cache_key_changes_with_parameters_and_detects_corruption(tmp_path):
    cache = ReconstructionCache(tmp_path / "cache")
    source = tmp_path / "source.jsonl"
    source.write_text('{"vertices":[]}')
    configuration = {"image": "digest", "class67": "include", "frame": [1, 2, 3]}
    cached = cache.put(configuration, source)
    assert cache.get(configuration) == cached
    assert cache.get({**configuration, "class67": "exclude"}) is None
    cached.write_text("broken")
    assert cache.get(configuration) is None


def test_cache_repeat_put_is_idempotent(tmp_path):
    cache = ReconstructionCache(tmp_path / "cache")
    source = tmp_path / "source.jsonl"
    source.write_text("{}")
    assert cache.put({"key": 1}, source) == cache.put({"key": 1}, source)


def test_manifest_requires_matching_selection_and_complete_inputs(tmp_path):
    source = tmp_path / "original.laz"
    source.write_bytes(b"points")
    manifest = {
        "schema_version": "1.0",
        "request": {"addresses": ["1 rue Paris"]},
        "buildings": building_snapshot({"FR:group": BuildingGroup("group")}),
        "artifacts": {"lidar_original": artifact(source, tmp_path)},
    }
    path = write_json(tmp_path / "manifest.json", manifest)
    assert load_manifest(path, addresses=("1 rue Paris",)) == manifest
    with pytest.raises(ReconstructionError):
        load_manifest(path, addresses=("different request",))
    source.unlink()
    with pytest.raises(ReconstructionError):
        load_manifest(path)


def test_failed_export_keeps_all_existing_outputs(tmp_path):
    model, svg = tmp_path / "model.osm", tmp_path / "model.svg"
    model.write_text("old model")
    svg.write_text("old svg")
    with pytest.raises(RuntimeError):
        with OutputTransaction([model, svg]) as transaction:
            transaction.stage(model).write_text("new model")
            raise RuntimeError("SVG validation failed")
    assert model.read_text() == "old model"
    assert svg.read_text() == "old svg"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.osm", "model.svg"]


def test_complete_transaction_and_duplicate_destinations(tmp_path):
    paths = [tmp_path / "model.osm", tmp_path / "model.svg"]
    with OutputTransaction(paths) as transaction:
        for path in paths:
            transaction.stage(path).write_text(path.suffix)
        transaction.commit()
    assert [p.read_text() for p in paths] == [".osm", ".svg"]
    with pytest.raises(ReconstructionError):
        OutputTransaction([paths[0], paths[0]])


def test_nonfinite_json_is_not_published(tmp_path):
    path = tmp_path / "report.json"
    write_json(path, {"volume": None})
    with pytest.raises(ValueError):
        write_json(path, {"volume": float("nan")})
    assert json.loads(path.read_text()) == {"volume": None}
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})


def test_staged_replay_is_self_contained(tmp_path):
    source = tmp_path / "source"
    replay = tmp_path / "replay"
    source.mkdir()
    replay.mkdir()
    crop = source / "crop.laz"
    crop.write_bytes(b"preserved source classifications")
    manifest = {
        "schema_version": "1.0",
        "request": {"addresses": ["1 rue Paris"]},
        "buildings": building_snapshot({"FR:group": BuildingGroup("group")}),
        "artifacts": {"lidar_original": artifact(crop, source)},
    }
    path = write_json(source / "manifest.json", manifest)
    staged, staged_path = stage_manifest(manifest, path, replay)
    crop.unlink()
    assert load_manifest(staged_path)["artifacts"] == staged["artifacts"]
    assert (
        checked_artifact(replay, staged["artifacts"]["lidar_original"]).read_bytes()
        == b"preserved source classifications"
    )
