"""Digest-pinned, resource-bounded external reconstruction tools."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
from typing import Any
from uuid import uuid4

from .types import ReconstructionError

DEFAULT_IMAGE = (
    "3dgi/3dbag-pipeline-tools@sha256:"
    "055d02c5ea9ed23945e2f96725a1637e0d9fc66c68b78b92148ce953ba8c22c5"
)
TOOL_PATH = (
    "/opt/3dbag-pipeline/tools/bin:/opt/3dbag-pipeline/venv/bin:"
    "/usr/local/bin:/usr/bin:/bin"
)
TOOLS = frozenset({"roofer", "ogr2ogr", "ogrinfo", "pdal", "val3dity", "cjio"})


@dataclass(frozen=True)
class RunResult:
    command: tuple[str, ...]
    returncode: int
    stdout_path: Path
    stderr_path: Path

    @property
    def stdout(self) -> str:
        return self.stdout_path.read_text(encoding="utf-8", errors="replace")

    @property
    def stderr(self) -> str:
        return self.stderr_path.read_text(encoding="utf-8", errors="replace")


class DockerRuntime:
    """Run only explicit tool argv, exposing a single working directory."""

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        timeout: float = 300,
        jobs: int = 4,
        offline: bool = False,
    ) -> None:
        if not re.fullmatch(r"[\w./:-]+@sha256:[0-9a-f]{64}", image):
            raise ReconstructionError("runtime_unpinned", "Runtime image must use an immutable sha256 digest")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ReconstructionError("runtime_configuration", "Runtime timeout must be positive and finite")
        if isinstance(jobs, bool) or not isinstance(jobs, int) or not 1 <= jobs <= 64:
            raise ReconstructionError("runtime_configuration", "Runtime jobs must be an integer from 1 to 64")
        self.image = image
        self.timeout = timeout
        self.jobs = jobs
        self.offline = offline
        self._image_ready = False
        self._preflight: dict[str, Any] | None = None
        self._preflight_work_dir: Path | None = None

    def _docker(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["docker", *args], capture_output=True, text=True,
                timeout=self.timeout, check=False,
            )
        except FileNotFoundError as exc:
            raise ReconstructionError("runtime_missing", "Docker executable is unavailable") from exc
        except subprocess.TimeoutExpired as exc:
            raise ReconstructionError("runtime_timeout", "Docker operation exceeded its time limit") from exc

    def _ensure_image(self) -> None:
        if self._image_ready:
            return
        if shutil.which("docker") is None:
            raise ReconstructionError("runtime_missing", "Docker executable is unavailable")
        daemon = self._docker(["version", "--format", "{{.Server.Version}}"])
        if daemon.returncode or not daemon.stdout.strip():
            raise ReconstructionError("runtime_unavailable", "Docker daemon is unavailable")
        inspected = self._docker(["image", "inspect", self.image])
        if inspected.returncode:
            if self.offline:
                raise ReconstructionError("runtime_image_missing", "Pinned image is not cached; offline mode forbids pulling")
            pulled = self._docker(["pull", self.image])
            if pulled.returncode:
                raise ReconstructionError("runtime_pull_failed", "Could not pull the pinned runtime image")
            inspected = self._docker(["image", "inspect", self.image])
        try:
            images = json.loads(inspected.stdout)
            digests = images[0]["RepoDigests"]
            expected_digest = self.image.rsplit("@", 1)[1]
            matches = any(d.rsplit("@", 1)[-1] == expected_digest for d in digests)
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            matches = False
        if inspected.returncode or not matches:
            raise ReconstructionError("runtime_image_mismatch", "Cached image does not match the required digest")
        self._image_ready = True

    @staticmethod
    def container_path(path: Path, work_dir: Path) -> str:
        try:
            relative = Path(path).resolve().relative_to(Path(work_dir).resolve())
        except ValueError as exc:
            raise ReconstructionError("runtime_path_escape", "Runtime inputs and outputs must be inside work_dir") from exc
        return (Path("/work") / relative).as_posix()

    def _remove_container(self, name: str) -> None:
        try:
            subprocess.run(
                ["docker", "rm", "--force", name], capture_output=True,
                timeout=15, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def run(
        self, args: list[str], work_dir: Path, *, network: bool = False
    ) -> RunResult:
        if not args or any(not isinstance(a, str) or "\0" in a for a in args) or args[0] not in TOOLS:
            raise ReconstructionError("runtime_command", "An explicit supported executable and string argv are required")
        if self.offline and network:
            raise ReconstructionError("runtime_offline", "Offline mode forbids container network access")
        self._ensure_image()
        work_dir = Path(work_dir).resolve()
        if not work_dir.is_dir() or "," in str(work_dir):
            raise ReconstructionError("runtime_workdir", "A mountable working directory is required")
        name = f"geogen-{uuid4().hex}"
        logs = work_dir / "runtime-logs" / name
        logs.mkdir(parents=True)
        scratch = logs / "scratch"
        scratch.mkdir()
        stdout_path, stderr_path = logs / "stdout.log", logs / "stderr.log"
        command = [
            "docker", "run", "--rm", "--pull", "never", "--name", name,
            "--network", "bridge" if network else "none",
            "--memory", "4g", "--memory-swap", "4g",
            "--cpus", str(min(self.jobs, os.cpu_count() or 1, 2)),
            "--pids-limit", "256", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--read-only",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--mount", f"type=bind,src={work_dir},dst=/work",
            "--workdir", self.container_path(logs, work_dir), "--entrypoint", "/usr/bin/env",
            self.image, f"PATH={TOOL_PATH}", f"HOME={self.container_path(logs, work_dir)}",
            f"TMPDIR={self.container_path(scratch, work_dir)}", *args,
        ]
        process: subprocess.Popen | None = None
        previous_term = None
        can_handle_term = threading.current_thread() is threading.main_thread()

        def cancel(signum: int, frame: Any) -> None:
            raise ReconstructionError("runtime_cancelled", "Runtime operation was cancelled")

        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                if can_handle_term:
                    previous_term = signal.signal(signal.SIGTERM, cancel)
                process = subprocess.Popen(
                    command, stdout=stdout, stderr=stderr, start_new_session=True,
                )
                try:
                    returncode = process.wait(timeout=self.timeout)
                except subprocess.TimeoutExpired as exc:
                    raise ReconstructionError(
                        "runtime_timeout", f"Tool exceeded {self.timeout}s; diagnostics: {stderr_path}"
                    ) from exc
            if returncode:
                raise ReconstructionError(
                    "runtime_failed", f"{args[0]} exited {returncode}; diagnostics: {stderr_path}"
                )
            return RunResult(tuple(command), returncode, stdout_path, stderr_path)
        except OSError as exc:
            raise ReconstructionError("runtime_failed", f"Could not execute Docker: {exc}") from exc
        finally:
            # Killing the CLI alone does not stop the daemon-owned container.
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            self._remove_container(name)
            if can_handle_term and previous_term is not None:
                signal.signal(signal.SIGTERM, previous_term)
            shutil.rmtree(scratch, ignore_errors=True)

    def preflight(self, work_dir: Path | None = None) -> dict[str, Any]:
        work_dir = Path(work_dir).resolve() if work_dir is not None else Path.cwd()
        if self._preflight is not None and self._preflight_work_dir == work_dir:
            return self._preflight
        self._ensure_image()
        work_dir.mkdir(parents=True, exist_ok=True)
        versions = {}
        for tool, key in (
            ("roofer", "roofer"), ("ogr2ogr", "gdal"), ("pdal", "pdal"),
            ("val3dity", "val3dity"), ("cjio", "cjio"),
        ):
            result = self.run([tool, "--version"], work_dir)
            versions[key] = (result.stdout + result.stderr).strip()
        help_result = self.run(["roofer", "--help"], work_dir)
        help_text = help_result.stdout + help_result.stderr
        required = (
            "--jobs", "--polygon-source-layer", "--id-attribute", "--srs",
            "--[no-]lod12", "--[no-]lod22", "--h-terrain-strategy",
            "--h-terrain-attribute", "--h-roof-attribute", "--[no-]clip-terrain",
        )
        if any(option not in help_text for option in required):
            raise ReconstructionError("runtime_capability", "Pinned Roofer lacks required reconstruction capabilities")
        validator = self.run(["val3dity", "--help"], work_dir)
        if "CityJSONL" not in validator.stdout or "--report" not in validator.stdout:
            raise ReconstructionError("runtime_capability", "Validator must support CityJSONL and JSON reports")
        lock = json.loads(Path(__file__).with_name("runtime-lock.json").read_text())
        if self.image == DEFAULT_IMAGE:
            for tool, expected in lock["versions"].items():
                if expected not in versions[tool]:
                    raise ReconstructionError("runtime_version", f"Unexpected {tool} version in pinned image")
        self._preflight = {
            "image": self.image, "versions": versions,
            "capabilities": {
                "cityjsonseq": True, "lod12": True, "lod22": True,
                "per_unit_ground_fallback": True, "independent_val3dity": True,
            },
            "offline": self.offline, "jobs": self.jobs, "timeout": self.timeout,
            "logs": str(work_dir / "runtime-logs"),
        }
        self._preflight_work_dir = work_dir
        return self._preflight

    def validate_cityjson(self, path: Path, work_dir: Path) -> dict[str, Any]:
        path, work_dir = Path(path), Path(work_dir).resolve()
        source = self.container_path(path, work_dir)
        if not path.is_file() or not path.stat().st_size:
            raise ReconstructionError("validator_input_missing", "Raw CityJSON validation requires a nonempty input file")
        reports = work_dir / "validation"
        reports.mkdir(exist_ok=True)
        report_path = reports / f"val3dity-{uuid4().hex}.json"
        result = self.run(
            ["val3dity", source, "--report", self.container_path(report_path, work_dir)], work_dir
        )
        try:
            report = json.loads(report_path.read_text())
        except (OSError, ValueError) as exc:
            raise ReconstructionError(
                "validator_report_missing", f"Validator did not produce a readable report: {report_path}"
            ) from exc
        valid = (
            isinstance(report, dict) and report.get("type") == "val3dity_report"
            and report.get("validity") is True
            and report.get("all_errors") == [] and report.get("dataset_errors") == []
            and isinstance(report.get("features"), list) and bool(report["features"])
        )
        has_solid = False
        if valid:
            for feature in report["features"]:
                if not isinstance(feature, dict) or feature.get("validity") is not True:
                    valid = False
                    break
                primitives = feature.get("primitives")
                if not isinstance(primitives, list) or not primitives or feature.get("errors", []):
                    valid = False
                    break
                for primitive in primitives:
                    if not isinstance(primitive, dict) or primitive.get("validity") is not True or primitive.get("errors", []):
                        valid = False
                        break
                    has_solid |= primitive.get("type") in {"Solid", "CompositeSolid", "MultiSolid"}
        if not valid or not has_solid:
            raise ReconstructionError("validator_invalid", f"Independent raw solid validation failed: {report_path}")
        return {
            **report, "report_path": str(report_path),
            "stdout_path": str(result.stdout_path), "stderr_path": str(result.stderr_path),
            "runtime_image": self.image,
        }
