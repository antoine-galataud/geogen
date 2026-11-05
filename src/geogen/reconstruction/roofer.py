"""Small adapter for the observed, locked Roofer CLI (not a quality gate)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from uuid import uuid4

from .runtime import DEFAULT_IMAGE, DockerRuntime
from .types import ReconstructionError

ID_ATTRIBUTE = "reconstruction_id"
GROUND_ATTRIBUTE = "ground_elevation"
ROOF_ATTRIBUTE = "roof_elevation"


def reconstruct(
    input_laz: Path,
    outline_geojson: Path,
    work_dir: Path,
    runtime: DockerRuntime,
) -> Path:
    """Return raw output, retaining Roofer diagnostics and extrusion modes.

    Elevation attributes are absolute elevations in the input vertical datum,
    not building heights. Missing attributes never become zero elevations.
    """
    work_dir = Path(work_dir).resolve()
    input_laz, outline_geojson = Path(input_laz), Path(outline_geojson)
    for path in (input_laz, outline_geojson):
        runtime.container_path(path, work_dir)
        if not path.is_file() or not path.stat().st_size:
            raise ReconstructionError("roofer_input_missing", f"Missing reconstruction input: {path}")
    try:
        outlines = json.loads(outline_geojson.read_text())
        features = outlines["features"]
        if outlines["type"] != "FeatureCollection" or not features:
            raise ValueError("A nonempty FeatureCollection is required")
        identifiers = [feature["properties"][ID_ATTRIBUTE] for feature in features]
        if any(not isinstance(i, str) or not i for i in identifiers) or len(set(identifiers)) != len(identifiers):
            raise ValueError("Every outline must have a unique string reconstruction_id")
        for feature in features:
            if feature["geometry"]["type"] not in {"Polygon", "MultiPolygon"}:
                raise ValueError("Outlines must be polygons")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ReconstructionError("roofer_input_invalid", f"Invalid outline source: {exc}") from exc
    if any(feature["properties"].get(GROUND_ATTRIBUTE) is None for feature in features):
        raise ReconstructionError(
            "roofer_ground_missing",
            "Every reconstruction unit requires an independently established ground_elevation",
        )
    options = [
        "roofer", "-j", str(runtime.jobs), "--id-attribute", ID_ATTRIBUTE,
        "--srs", "EPSG:2154", "--lod12", "--lod22", "--no-clip-terrain",
        "--h-terrain-strategy", "user",
    ]
    for attribute, flag in (
        (GROUND_ATTRIBUTE, "--h-terrain-attribute"), (ROOF_ATTRIBUTE, "--h-roof-attribute"),
    ):
        values = [f["properties"].get(attribute) for f in features]
        supplied = [v is not None for v in values]
        if any(supplied):
            if not all(supplied):
                raise ReconstructionError(
                    "roofer_fallback_mixed",
                    f"Run units with and without {attribute} separately to avoid implicit zero fallback",
                )
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                raise ReconstructionError("roofer_fallback_invalid", f"{attribute} must contain finite absolute elevations")
            options.extend([flag, attribute])
    candidate = work_dir / f"roofer-{uuid4().hex}"
    candidate.mkdir()
    output_dir = candidate / "output"
    args = [
        *options, runtime.container_path(input_laz, work_dir),
        runtime.container_path(outline_geojson, work_dir),
        runtime.container_path(output_dir, work_dir),
    ]
    invocation = {
        "image": runtime.image, "args": args,
        "complexity": 0.888 if runtime.image == DEFAULT_IMAGE else None,
        "complexity_source": "runtime_default",
        "ground_strategy": "user", "units": identifiers,
    }
    provenance = candidate / "invocation.json"
    provenance.write_text(json.dumps(invocation, indent=2) + "\n")
    try:
        result = runtime.run(args, work_dir)
    except BaseException as exc:
        invocation["failure"] = getattr(exc, "code", type(exc).__name__)
        provenance.write_text(json.dumps(invocation, indent=2) + "\n")
        raise
    invocation.update(stdout_path=str(result.stdout_path), stderr_path=str(result.stderr_path))
    provenance.write_text(json.dumps(invocation, indent=2) + "\n")
    files = sorted(output_dir.glob("*.city.jsonl"))
    if len(files) != 1 or not files[0].stat().st_size:
        raise ReconstructionError("roofer_output_missing", f"Expected one nonempty raw CityJSONSeq file in {output_dir}")
    try:
        with files[0].open() as stream:
            header = json.loads(next(stream))
            if header.get("type") != "CityJSON" or not isinstance(header.get("transform"), dict):
                raise ValueError("Missing CityJSON sequence header and transform")
            count = 0
            for line in stream:
                if not line.strip():
                    continue
                feature = json.loads(line)
                if feature.get("type") != "CityJSONFeature" or not feature.get("CityObjects"):
                    raise ValueError("Invalid CityJSON feature")
                count += 1
            if count == 0:
                raise ValueError("No reconstructed features")
    except (OSError, ValueError, AttributeError, StopIteration) as exc:
        raise ReconstructionError("roofer_output_invalid", f"Invalid raw sequence: {files[0]}: {exc}") from exc
    return files[0]
