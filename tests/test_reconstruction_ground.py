"""Bounded ground selection and replay after unsuccessful acquisition-time selection."""

import json
import shutil

import laspy
import numpy as np
import pytest
from click.testing import CliRunner
from pyproj import CRS
from shapely.geometry import box

from geogen.cli import main
from geogen.models import BuildingGroup
from geogen.reconstruction.acquire import acquire
from geogen.reconstruction.cache import content_hash, load_manifest
from geogen.reconstruction.pipeline import ground_references
from geogen.reconstruction.types import (
    ReconstructionError,
    ReconstructionUnit,
    SourceIdentity,
)

X, Y = 700000, 6600000
BASE = "https://api.bdnb.io/v1/bdnb"
ADDRESS = "Synthetic ground test, France"


def unit(name="target", *, outline=None, role="target"):
    outline = outline if outline is not None else box(X, Y, X + 10, Y + 10)
    return ReconstructionUnit(
        name,
        SourceIdentity(bdnb_group_ids=("group",)),
        outline.wkt,
        outline.wkt,
        content_hash(outline.wkt),
        role=role,
    )


def cloud(path, points, classifications=None):
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.add_crs(CRS.from_epsg(2154))
    header.scales = np.array([0.001] * 3)
    header.offsets = np.array([X, Y, 100.0])
    data = laspy.LasData(header)
    data.x, data.y, data.z = np.array(points).T
    data.classification = classifications if classifications is not None else [2] * len(points)
    data.write(path)
    return path


def test_near_ground_is_preferred_to_lower_distant_ground(tmp_path):
    path = cloud(
        tmp_path / "points.laz",
        [(X - 1, Y + i, 100 + i) for i in range(1, 4)] + [(X - 8, Y + i, 20) for i in range(1, 4)],
    )
    report = {}
    values = ground_references(path, (unit(),), diagnostics=report)
    assert values == {"target": pytest.approx(101.1)}
    assert report["target"]["search_radius_m"] == 3
    assert report["target"]["attempts"] == [{"radius_m": 3, "ground_point_count": 3}]
    assert not report["target"]["expanded_search"]


def test_search_expands_only_until_sufficient_points(tmp_path):
    path = cloud(tmp_path / "points.laz", [(X - 4, Y + i, 100 + i) for i in range(1, 4)])
    report = {}
    values = ground_references(path, (unit(),), diagnostics=report)
    assert values["target"] == pytest.approx(101.1)
    assert report["target"]["expanded_search"]
    assert report["target"]["method"] == "local_ground_5th_percentile_6m_buffer"
    assert report["target"]["attempts"] == [
        {"radius_m": 3, "ground_point_count": 0},
        {"radius_m": 6, "ground_point_count": 3},
    ]


def test_neighbour_and_target_ground_classifications_are_excluded(tmp_path):
    path = cloud(
        tmp_path / "points.laz",
        [(X + 5, Y + i, 1) for i in range(1, 4)]
        + [(X - 4, Y + i, 2) for i in range(1, 4)]
        + [(X - 8, Y + i, 100) for i in range(1, 4)],
    )
    neighbour = unit("context", outline=box(X - 5, Y, X, Y + 10), role="context")
    report = {}
    assert ground_references(path, (unit(), neighbour), diagnostics=report) == {"target": 100}
    assert report["target"]["search_radius_m"] == 10
    assert report["target"]["ground_point_count"] == 3
    assert "context" not in report


def test_ground_in_courtyard_is_eligible(tmp_path):
    outline = box(X, Y, X + 20, Y + 20).difference(box(X + 5, Y + 5, X + 15, Y + 15))
    path = cloud(tmp_path / "points.laz", [(X + 6, Y + i, 100) for i in range(6, 9)])
    assert ground_references(path, (unit(outline=outline),)) == {"target": 100}


def test_roof_returns_do_not_substitute_for_missing_ground(tmp_path):
    path = cloud(tmp_path / "points.laz", [(X - 1, Y + i, 100) for i in range(3)], [6] * 3)
    report = {}
    assert ground_references(path, (unit(),), diagnostics=report, allow_missing=True) == {}
    assert report["target"]["elevation_m"] is None
    assert report["target"]["ground_point_count"] == 0
    with pytest.raises(ReconstructionError):
        ground_references(path, (unit(),))


def test_reports_all_missing_targets_and_retains_valid_ones(tmp_path):
    path = cloud(tmp_path / "points.laz", [(X - 1, Y + i, 100) for i in range(3)])
    missing = unit("missing", outline=box(X + 100, Y, X + 110, Y + 10))
    report = {}
    assert ground_references(path, (unit(), missing), diagnostics=report, allow_missing=True) == {
        "target": 100
    }
    assert report["missing"]["status"] == "ground_unverified"
    assert report["target"]["status"] == "observed"


@pytest.mark.parametrize("offset", [4, 15])
def test_acquisition_retains_manifest_and_expands_crop(monkeypatch, tmp_path, offset):
    from geogen.reconstruction.ign import IgnClient
    from geogen.reconstruction.runtime import DockerRuntime

    selected = unit()
    building = BuildingGroup("group", geometry=selected.working_wkt, crs="EPSG:2154")
    source = cloud(tmp_path / "source.laz", [(X - offset, Y + i, 100) for i in range(3)])
    monkeypatch.setattr(
        "geogen.reconstruction.acquire.BdnbClient.relationship_rows", lambda *a, **k: {}
    )
    monkeypatch.setattr(IgnClient, "buildings_for_sources", lambda *a, **k: {})
    monkeypatch.setattr("geogen.reconstruction.acquire.resolve_units", lambda *a, **k: (selected,))
    monkeypatch.setattr(IgnClient, "verify_tile_headers", lambda *a: {})
    extents = []

    def tiles(self, bounds):
        extents.append(bounds)
        return {"tiles": [{"vertical_reference": "IGN69"}]}

    monkeypatch.setattr(IgnClient, "tiles", tiles)
    monkeypatch.setattr("geogen.reconstruction.acquire.build_pdal_pipeline", lambda *a, **k: {})

    class Runtime:
        def container_path(self, path, work):
            return str(path)

        def run(self, args, work, **kwargs):
            shutil.copyfile(source, work / "lidar.original.laz")

    request = {
        "addresses": [ADDRESS],
        "country": "FR",
        "max_buildings": 10,
        "provider_endpoint_hash": content_hash(BASE),
    }
    options = {
        "api_key": None,
        "base_url": BASE,
        "timeout": 10,
    }
    work = tmp_path / "acquired"
    path = acquire({"FR:group": building}, request, options, work, Runtime())
    manifest = load_manifest(path)
    buffer = 10
    unverified = offset > buffer
    assert extents == [(X - buffer, Y - buffer, X + 10 + buffer, Y + 10 + buffer)]
    assert manifest["acquisition_buffer_m"] == buffer
    assert manifest["ground_references"]["target"]["status"] == (
        "ground_unverified" if unverified else "observed"
    )
    assert manifest["frame"]["origin"][2] == (0 if unverified else 100)
    assert manifest["frame_origin_z_method"] == (
        "coordinate_origin_only" if unverified else "observed_ground"
    )

    if not unverified:
        return
    # Replaying a crop without usable ground yields diagnostics, not a lost acquisition.
    monkeypatch.setattr(DockerRuntime, "preflight", lambda *a: {})
    monkeypatch.setattr(
        "geogen.reconstruction.roofer.reconstruct", lambda *a, **k: pytest.fail("roofer")
    )
    result = CliRunner().invoke(
        main,
        [
            ADDRESS,
            "--country",
            "FR",
            "--base-url",
            BASE,
            "--roof-source",
            "roofer",
            "--reconstruction-input",
            str(path),
            "--reconstruction-cache",
            str(tmp_path / "cache"),
            "--output",
            str(tmp_path / "model.osm"),
        ],
    )
    assert result.exit_code != 0
    report = json.loads((tmp_path / "model.reconstruction.json").read_text())
    assert report["failed_units"] == ["target"]
    assert report["ground_references"]["target"]["status"] == "ground_unverified"
    assert report["units"][0]["failure"]["code"] == "ground_unverified"
    assert not report["outputs_published"]
