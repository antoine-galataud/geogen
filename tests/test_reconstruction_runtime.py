"""No-network runtime contracts and real-output parser boundary fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import signal
import subprocess

import pytest

from geogen.reconstruction import runtime as runtime_module
from geogen.reconstruction.roofer import reconstruct
from geogen.reconstruction.runtime import DEFAULT_IMAGE, DockerRuntime, RunResult
from geogen.reconstruction.types import ReconstructionError

FIXTURES = Path(__file__).parent / "fixtures" / "reconstruction"


@pytest.fixture
def runtime():
    result = DockerRuntime(offline=True)
    result._image_ready = True
    return result


def fake_process(monkeypatch, *, returncode=0, interruption=None):
    calls = []

    class Process:
        pid = 987654321

        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))
            self.done = False
            self.interrupted = False
            kwargs["stdout"].write(b"diagnostic output\n")
            kwargs["stderr"].write(b"diagnostic error\n")

        def wait(self, timeout=None):
            if interruption and not self.interrupted:
                self.interrupted = True
                if interruption == "sigterm":
                    signal.raise_signal(signal.SIGTERM)
                raise interruption
            self.done = True
            return returncode

        def poll(self):
            return returncode if self.done else None

    monkeypatch.setattr(runtime_module.subprocess, "Popen", Process)
    removed = []
    monkeypatch.setattr(DockerRuntime, "_remove_container", lambda self, name: removed.append(name))
    killed = []
    monkeypatch.setattr(runtime_module.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    return calls, removed, killed


@pytest.mark.parametrize("image", ["roofer:latest", "roofer:v1", "roofer@sha256:abc", "roofer@sha256:" + "G" * 64])
def test_runtime_requires_immutable_digest(image):
    with pytest.raises(ReconstructionError, match="immutable"):
        DockerRuntime(image)


@pytest.mark.parametrize("settings", [{"timeout": 0}, {"timeout": float("nan")}, {"jobs": 0}, {"jobs": True}, {"jobs": 65}])
def test_runtime_rejects_unbounded_configuration(settings):
    with pytest.raises(ReconstructionError):
        DockerRuntime(**settings)


def test_runtime_mount_and_argv_are_restricted(runtime, tmp_path, monkeypatch):
    calls, removed, killed = fake_process(monkeypatch)
    result = runtime.run(["pdal", "pipeline", "/work/path with spaces;echo nope.json"], tmp_path)
    command, kwargs = calls[0]
    assert command[:2] == ["docker", "run"]
    assert command[command.index("--pull") + 1] == "never"
    assert command[command.index("--network") + 1] == "none"
    assert command.count("--mount") == 1
    assert command[command.index("--mount") + 1] == f"type=bind,src={tmp_path},dst=/work"
    assert command[command.index("--entrypoint") + 1] == "/usr/bin/env"
    assert command[-1] == "/work/path with spaces;echo nope.json"
    assert not kwargs.get("shell", False)
    assert kwargs["start_new_session"] is True
    assert "--memory" in command and "--cpus" in command and "--pids-limit" in command
    assert "--read-only" in command and "--cap-drop" in command
    assert not any("docker.sock" in value for value in command)
    assert result.stdout == "diagnostic output\n"
    assert result.stderr == "diagnostic error\n"
    assert result.returncode == 0
    assert len(removed) == 1 and not killed


@pytest.mark.parametrize("args", [[], ["sh", "-c", "echo bad"], ["/bin/roofer"], ["pdal", "a\0b"], [["pdal"]]])
def test_runtime_disallows_shells_and_unknown_executables(runtime, tmp_path, args):
    with pytest.raises(ReconstructionError) as error:
        runtime.run(args, tmp_path)
    assert error.value.code == "runtime_command"


def test_runtime_offline_never_enables_network(runtime, tmp_path):
    with pytest.raises(ReconstructionError) as error:
        runtime.run(["pdal", "--version"], tmp_path, network=True)
    assert error.value.code == "runtime_offline"


def test_runtime_failure_preserves_diagnostics_and_removes_container(runtime, tmp_path, monkeypatch):
    calls, removed, killed = fake_process(monkeypatch, returncode=7)
    with pytest.raises(ReconstructionError) as error:
        runtime.run(["roofer", "--version"], tmp_path)
    assert error.value.code == "runtime_failed"
    assert "exited 7" in str(error.value)
    assert list(tmp_path.glob("runtime-logs/*/stderr.log"))[0].read_text() == "diagnostic error\n"
    assert removed and not killed


@pytest.mark.parametrize("interruption", [
    subprocess.TimeoutExpired("docker", 1), KeyboardInterrupt(), "sigterm",
])
def test_runtime_timeout_and_cancellation_remove_container_and_process_group(
    runtime, tmp_path, monkeypatch, interruption
):
    calls, removed, killed = fake_process(monkeypatch, interruption=interruption)
    handler = signal.getsignal(signal.SIGTERM)
    with pytest.raises((ReconstructionError, KeyboardInterrupt)):
        runtime.run(["roofer", "--version"], tmp_path)
    assert len(removed) == 1
    assert killed == [(987654321, signal.SIGKILL)]
    assert signal.getsignal(signal.SIGTERM) == handler
    assert not list(tmp_path.glob("runtime-logs/*/scratch"))


def test_container_path_rejects_symlink_escape(tmp_path):
    child = tmp_path / "work"
    child.mkdir()
    (child / "link").symlink_to(tmp_path)
    assert DockerRuntime.container_path(child / "data.laz", child) == "/work/data.laz"
    with pytest.raises(ReconstructionError):
        DockerRuntime.container_path(child / "link" / "secret", child)


def image_preflight(monkeypatch, *, cached=True, digest=DEFAULT_IMAGE, pull_ok=True):
    calls = []
    monkeypatch.setattr(runtime_module.shutil, "which", lambda name: "/usr/bin/docker")

    def docker(self, args):
        calls.append(args)
        if args[0] == "version":
            return subprocess.CompletedProcess(args, 0, "28.0.4", "")
        if args[0] == "pull":
            return subprocess.CompletedProcess(args, 0 if pull_ok else 1, "", "")
        pulled = any(c[0] == "pull" for c in calls)
        data = json.dumps([{"RepoDigests": [digest]}])
        return subprocess.CompletedProcess(args, 0 if cached or pulled else 1, data, "")

    monkeypatch.setattr(DockerRuntime, "_docker", docker)
    return calls


def test_offline_missing_image_fails_without_pull(monkeypatch):
    calls = image_preflight(monkeypatch, cached=False)
    with pytest.raises(ReconstructionError) as error:
        DockerRuntime(offline=True)._ensure_image()
    assert error.value.code == "runtime_image_missing"
    assert not any(call[0] == "pull" for call in calls)


def test_online_pulls_only_immutable_digest(monkeypatch):
    calls = image_preflight(monkeypatch, cached=False)
    DockerRuntime()._ensure_image()
    assert ["pull", DEFAULT_IMAGE] in calls


def test_cached_image_digest_is_verified(monkeypatch):
    image_preflight(monkeypatch, digest="other@sha256:" + "0" * 64)
    with pytest.raises(ReconstructionError) as error:
        DockerRuntime(offline=True)._ensure_image()
    assert error.value.code == "runtime_image_mismatch"


def test_missing_docker_has_actionable_failure(monkeypatch):
    monkeypatch.setattr(runtime_module.shutil, "which", lambda name: None)
    with pytest.raises(ReconstructionError) as error:
        DockerRuntime(offline=True).preflight()
    assert error.value.code == "runtime_missing"


def test_online_pull_failure_is_not_ignored(monkeypatch):
    image_preflight(monkeypatch, cached=False, pull_ok=False)
    with pytest.raises(ReconstructionError) as error:
        DockerRuntime()._ensure_image()
    assert error.value.code == "runtime_pull_failed"


@pytest.mark.parametrize("missing", [None, "capability", "version"])
def test_preflight_records_or_rejects_actual_capabilities(runtime, tmp_path, monkeypatch, missing):
    monkeypatch.chdir(tmp_path)
    lock = json.loads(Path(runtime_module.__file__).with_name("runtime-lock.json").read_text())
    versions = {**lock["versions"], "ogr2ogr": lock["versions"]["gdal"]}
    required = (
        "--jobs --polygon-source-layer --id-attribute --srs --[no-]lod12 --[no-]lod22 "
        "--h-terrain-strategy --h-terrain-attribute --h-roof-attribute "
        "--[no-]clip-terrain"
    )

    def run(args, work_dir, **kwargs):
        stdout = work_dir / f"{args[0]}-{args[1]}.stdout.log"
        stderr = work_dir / f"{args[0]}-{args[1]}.stderr.log"
        if args[1] == "--version":
            text = "wrong version" if missing == "version" else versions[args[0]]
        elif args[0] == "roofer":
            text = "" if missing == "capability" else required
        else:
            text = "CityJSONL --report"
        stdout.write_text(text)
        stderr.write_text("")
        return RunResult(tuple(args), 0, stdout, stderr)

    monkeypatch.setattr(runtime, "run", run)
    if missing:
        with pytest.raises(ReconstructionError) as error:
            runtime.preflight()
        assert error.value.code == f"runtime_{missing}"
    else:
        report = runtime.preflight()
        assert report["capabilities"]["independent_val3dity"] is True
        assert report["versions"]["roofer"] == versions["roofer"]
        assert report["image"] == DEFAULT_IMAGE
        assert runtime.preflight() is report


def test_preflight_keeps_logs_in_explicit_workspace_and_refreshes_for_replay(
    runtime, tmp_path, monkeypatch
):
    lock = json.loads(Path(runtime_module.__file__).with_name("runtime-lock.json").read_text())
    workspaces = []

    def run(args, work_dir, **kwargs):
        workspaces.append(work_dir)
        stdout, stderr = work_dir / "stdout.log", work_dir / "stderr.log"
        if args[1] == "--version":
            text = lock["versions"]["gdal" if args[0] == "ogr2ogr" else args[0]]
        elif args[0] == "roofer":
            text = (
                "--jobs --polygon-source-layer --id-attribute --srs --[no-]lod12 --[no-]lod22 "
                "--h-terrain-strategy --h-terrain-attribute --h-roof-attribute "
                "--[no-]clip-terrain"
            )
        else:
            text = "CityJSONL --report"
        stdout.write_text(text)
        stderr.write_text("")
        return RunResult(tuple(args), 0, stdout, stderr)

    monkeypatch.setattr(runtime, "run", run)
    first, replay = tmp_path / "first", tmp_path / "replay"
    assert runtime.preflight(first)["logs"] == str(first / "runtime-logs")
    assert runtime.preflight(replay)["logs"] == str(replay / "runtime-logs")
    assert set(workspaces) == {first, replay}
    assert not (tmp_path / ".geogen-runtime").exists()


def valid_report():
    return {
        "type": "val3dity_report", "validity": True, "all_errors": [], "dataset_errors": [],
        "features": [{
            "id": "box", "validity": True, "errors": [],
            "primitives": [{"type": "Solid", "validity": True, "errors": []}],
        }],
    }


def validator_mock(runtime, monkeypatch, report):
    def run(args, work_dir, **kwargs):
        path = work_dir / Path(args[-1]).relative_to("/work")
        if report is not None:
            path.write_text(json.dumps(report))
        stdout, stderr = work_dir / "stdout.log", work_dir / "stderr.log"
        stdout.write_text("VALID :)")
        stderr.write_text("")
        return RunResult(tuple(args), 0, stdout, stderr)

    monkeypatch.setattr(runtime, "run", run)


def test_validator_uses_report_not_success_output(runtime, tmp_path, monkeypatch):
    source = tmp_path / "raw.city.jsonl"
    source.write_text("raw")
    report = valid_report()
    report["validity"] = False
    validator_mock(runtime, monkeypatch, report)
    with pytest.raises(ReconstructionError) as error:
        runtime.validate_cityjson(source, tmp_path)
    assert error.value.code == "validator_invalid"


@pytest.mark.parametrize("report", [
    None, {}, [], {**valid_report(), "features": []},
    {**valid_report(), "all_errors": [101]},
    {**valid_report(), "dataset_errors": ["missing object"]},
    {**valid_report(), "features": [{"validity": True, "primitives": []}]},
    {**valid_report(), "features": [{"validity": True, "primitives": [{"type": "Solid", "validity": False}]}]},
    {**valid_report(), "features": [{"validity": True, "primitives": [{"type": "MultiSurface", "validity": True}]}]},
])
def test_validator_fails_closed_on_missing_or_invalid_report(runtime, tmp_path, monkeypatch, report):
    source = tmp_path / "raw.city.jsonl"
    source.write_text("raw")
    validator_mock(runtime, monkeypatch, report)
    with pytest.raises(ReconstructionError):
        runtime.validate_cityjson(source, tmp_path)


def test_validator_returns_independent_report_with_artifact_paths(runtime, tmp_path, monkeypatch):
    source = tmp_path / "raw.city.jsonl"
    source.write_text("raw")
    validator_mock(runtime, monkeypatch, valid_report())
    report = runtime.validate_cityjson(source, tmp_path)
    assert report["validity"] is True
    assert report["runtime_image"] == DEFAULT_IMAGE
    assert Path(report["report_path"]).is_file()


def test_validator_rejects_missing_input_without_running(runtime, tmp_path):
    with pytest.raises(ReconstructionError) as error:
        runtime.validate_cityjson(tmp_path / "missing.city.jsonl", tmp_path)
    assert error.value.code == "validator_input_missing"


@pytest.fixture
def inputs(tmp_path):
    cloud = tmp_path / "input.laz"
    cloud.write_bytes(b"input")
    outline = tmp_path / "outlines.geojson"
    outline.write_text(json.dumps({
        "type": "FeatureCollection", "features": [{
            "type": "Feature", "properties": {"reconstruction_id": "box", "ground_elevation": 100.0},
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 0]]]},
        }],
    }))
    return cloud, outline


def roofer_mock(runtime, monkeypatch, *, data=None, failure=False):
    calls = []

    def run(args, work_dir, **kwargs):
        calls.append(args)
        output = work_dir / Path(args[-1]).relative_to("/work")
        output.mkdir()
        if data is not None:
            (output / "tile.city.jsonl").write_text(data)
        if failure:
            raise ReconstructionError("runtime_failed", "backend failed")
        stdout, stderr = work_dir / "stdout.log", work_dir / "stderr.log"
        stdout.write_text("complete")
        stderr.write_text("")
        return RunResult(tuple(args), 0, stdout, stderr)

    monkeypatch.setattr(runtime, "run", run)
    return calls


def test_roofer_default_uses_observed_flags_and_no_fabricated_elevations(
    runtime, tmp_path, monkeypatch, inputs
):
    data = (FIXTURES / "roofer-box-courtyard.city.jsonl").read_text()
    calls = roofer_mock(runtime, monkeypatch, data=data)
    output = reconstruct(*inputs, tmp_path, runtime)
    args = calls[0]
    assert args[:3] == ["roofer", "-j", "4"]
    assert "--lod12" in args and "--lod22" in args and "--no-clip-terrain" in args
    assert args[args.index("--h-terrain-strategy") + 1] == "user"
    assert "--complexity-factor" not in args
    assert args[args.index("--h-terrain-attribute") + 1] == "ground_elevation"
    assert "--h-roof-attribute" not in args
    assert output.read_text() == data
    assert "rf_extrusion_mode" in output.read_text()
    invocation = json.loads((output.parent.parent / "invocation.json").read_text())
    assert invocation["complexity"] == 0.888
    assert invocation["complexity_source"] == "runtime_default"
    assert invocation["ground_strategy"] == "user"
    assert "accepted" not in invocation


def test_custom_digest_does_not_claim_unobserved_complexity_default(runtime, tmp_path, monkeypatch, inputs):
    runtime.image = "custom/roofer@sha256:" + "a" * 64
    calls = roofer_mock(
        runtime, monkeypatch, data=(FIXTURES / "roofer-box-courtyard.city.jsonl").read_text()
    )
    output = reconstruct(*inputs, tmp_path, runtime)
    invocation = json.loads((output.parent.parent / "invocation.json").read_text())
    assert invocation["complexity"] is None
    assert invocation["complexity_source"] == "runtime_default"
    assert "--complexity-factor" not in calls[0]


def test_roofer_rejects_mixed_missing_ground_instead_of_passing_null_fallback(runtime, tmp_path, inputs):
    cloud, outline = inputs
    data = json.loads(outline.read_text())
    second = json.loads(json.dumps(data["features"][0]))
    second["properties"] = {"reconstruction_id": "other"}
    data["features"].append(second)
    outline.write_text(json.dumps(data))
    with pytest.raises(ReconstructionError) as error:
        reconstruct(cloud, outline, tmp_path, runtime)
    assert error.value.code == "roofer_ground_missing"


def test_roofer_missing_ground_blocks_before_subprocess(runtime, tmp_path, monkeypatch, inputs):
    cloud, outline = inputs
    data = json.loads(outline.read_text())
    del data["features"][0]["properties"]["ground_elevation"]
    outline.write_text(json.dumps(data))
    calls = roofer_mock(runtime, monkeypatch)
    with pytest.raises(ReconstructionError) as error:
        reconstruct(cloud, outline, tmp_path, runtime)
    assert error.value.code == "roofer_ground_missing" and not calls


@pytest.mark.parametrize("data", [None, "", '{"type":"CityJSON","transform":{}}\n', '{"type":"CityJSON","transform":{}}\n{'])
def test_roofer_rejects_zero_exit_with_missing_or_partial_output(runtime, tmp_path, monkeypatch, inputs, data):
    roofer_mock(runtime, monkeypatch, data=data)
    with pytest.raises(ReconstructionError):
        reconstruct(*inputs, tmp_path, runtime)


def test_roofer_failed_run_never_reuses_old_or_partial_output(runtime, tmp_path, monkeypatch, inputs):
    fixture = (FIXTURES / "roofer-box-courtyard.city.jsonl").read_text()
    roofer_mock(runtime, monkeypatch, data=fixture)
    previous = reconstruct(*inputs, tmp_path, runtime)
    roofer_mock(runtime, monkeypatch, data=fixture, failure=True)
    with pytest.raises(ReconstructionError):
        reconstruct(*inputs, tmp_path, runtime)
    assert previous.read_text() == fixture
    records = [json.loads(p.read_text()) for p in tmp_path.glob("roofer-*/invocation.json")]
    assert len(records) == 2 and sum("failure" in r for r in records) == 1


def test_frozen_fixture_is_real_immutable_capture():
    metadata = json.loads((FIXTURES / "roofer-runtime-metadata.json").read_text())
    content = (FIXTURES / metadata["fixture"]).read_bytes()
    assert hashlib.sha256(content).hexdigest() == metadata["sha256"]
    records = [json.loads(line) for line in content.splitlines()]
    assert records[0]["type"] == "CityJSON"
    assert {feature["id"] for feature in records[1:]} == {"box", "courtyard"}
    courtyard = next(feature for feature in records[1:] if feature["id"] == "courtyard")
    solid = courtyard["CityObjects"]["courtyard-0"]["geometry"][1]
    assert solid["lod"] == "2.2" and len(solid["boundaries"][0][0]) == 2
    assert metadata["independent_validation"]["validity"] is True
