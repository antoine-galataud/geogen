"""Checksummed local replay bundles and content-addressed reconstruction artifacts."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from geogen.models import Building, BuildingGroup, OsBuilding
from geogen.reconstruction.types import SCHEMA_VERSION, ReconstructionError

DEFAULT_CACHE = Path.home() / ".cache" / "geogen" / "reconstruction"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> Path:
    """Publish one complete JSON file without truncating a previous report."""
    payload = canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        name = Path(temporary.name)
        try:
            temporary.write(payload + b"\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        except BaseException:
            name.unlink(missing_ok=True)
            raise
    try:
        os.replace(name, path)
    finally:
        name.unlink(missing_ok=True)
    return path


def building_snapshot(buildings: dict[str, Building]) -> list[dict]:
    snapshots = []
    for key, building in buildings.items():
        data = asdict(building)
        if isinstance(building.geometry, BaseGeometry):
            data["geometry"] = mapping(building.geometry)
        elif isinstance(building.geometry, bytes):
            data["geometry"] = building.geometry.hex()
        snapshots.append({"key": key, "type": type(building).__name__, "data": data})
    return json.loads(canonical_json(snapshots))


def restore_buildings(snapshots: list[dict]) -> dict[str, Building]:
    classes = {"Building": Building, "BuildingGroup": BuildingGroup, "OsBuilding": OsBuilding}
    buildings = {}
    try:
        if not isinstance(snapshots, list) or not snapshots:
            raise ValueError("Empty source snapshot")
        for snapshot in snapshots:
            cls = classes[snapshot["type"]]
            data = dict(snapshot["data"])
            for name in ("glazed_areas", "glazing_orientations", "walls"):
                if name in data:
                    data[name] = tuple(
                        tuple(value) if isinstance(value, list) else value for value in data[name]
                    )
            allowed = {f.name for f in fields(cls) if f.init}
            building = cls(**{k: v for k, v in data.items() if k in allowed})
            key = snapshot["key"]
            if key != f"{building.country}:{building.code}" or key in buildings:
                raise ValueError("Duplicate or inconsistent building identity")
            buildings[key] = building
    except (KeyError, TypeError, ValueError) as error:
        raise ReconstructionError("manifest_invalid", "Invalid provider snapshot") from error
    return buildings


def artifact(path: Path, root: Path) -> dict[str, str]:
    return {"path": str(path.resolve().relative_to(root.resolve())), "sha256": file_hash(path)}


def checked_artifact(root: Path, entry: dict) -> Path:
    """Artifacts must stay inside the bundle, including after symlink resolution."""
    try:
        relative = Path(entry["path"])
        expected = entry["sha256"]
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Artifact path escapes replay bundle")
        path = (root / relative).resolve()
        path.relative_to(root.resolve())
        if len(expected) != 64 or not path.is_file() or file_hash(path) != expected:
            raise ValueError("Missing or corrupt artifact")
        return path
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise ReconstructionError(
            "cache_input_missing", "Replay artifact is missing, outside the bundle, or corrupt"
        ) from error


def load_manifest(path: Path, *, addresses: tuple[str, ...] | None = None) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported manifest schema")
        if addresses is not None and list(addresses) != manifest["request"]["addresses"]:
            raise ValueError("Manifest addresses differ from the requested selection")
        restore_buildings(manifest["buildings"])
        if not isinstance(manifest["artifacts"], dict):
            raise ValueError("Artifacts must be a keyed object")
        for entry in manifest["artifacts"].values():
            checked_artifact(path.parent, entry)
    except ReconstructionError:
        raise
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise ReconstructionError(
            "manifest_invalid", "Invalid or incomplete replay manifest"
        ) from error
    return manifest


def stage_manifest(manifest: dict, source: Path, work: Path) -> tuple[dict, Path]:
    """Make a replay self-contained, including when the input bundle is elsewhere."""
    directory = work / "acquisition"
    directory.mkdir()
    entries = {}
    for key, entry in manifest["artifacts"].items():
        original = checked_artifact(source.parent, entry)
        destination = directory / (content_hash(key)[:16] + original.suffix)
        shutil.copyfile(original, destination)
        entries[key] = artifact(destination, work)
    result = {**manifest, "artifacts": entries}
    return result, write_json(work / "manifest.json", result)


class ReconstructionCache:
    """Cache only complete files; every read rechecks their content hashes."""

    def __init__(self, root: Path = DEFAULT_CACHE):
        self.root = root

    def get(self, configuration: dict) -> Path | None:
        directory = self.root / content_hash(configuration)
        manifest = directory / "artifact.json"
        if not manifest.exists():
            return None
        try:
            entry = json.loads(manifest.read_text(encoding="utf-8"))
            return checked_artifact(directory, entry)
        except (OSError, ValueError, ReconstructionError):
            return None

    def put(self, configuration: dict, source: Path) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        directory = self.root / content_hash(configuration)
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            staging = Path(temporary)
            cached = staging / "model.city.jsonl"
            shutil.copyfile(source, cached)
            write_json(staging / "artifact.json", artifact(cached, staging))
            try:
                os.rename(staging, directory)
            except OSError as error:
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
                existing = self.get(configuration)
                if existing is None:
                    raise ReconstructionError(
                        "cache_corrupt", "Existing cache entry failed checksum validation"
                    )
        return checked_artifact(directory, json.loads((directory / "artifact.json").read_text()))
