"""Point preparation and frame handling preserve measured coordinates."""

import laspy
import numpy as np
import pytest
from pyproj import CRS

from geogen.reconstruction.pipeline import ground_references, prepare_points, read_frame
from geogen.reconstruction.types import (
    ReconstructionError,
    ReconstructionUnit,
    SourceIdentity,
)


def _cloud(path, crs=2154):
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.add_crs(CRS.from_epsg(crs))
    header.scales = np.array([0.001] * 3)
    header.offsets = np.array([700000, 6600000, 100], dtype=float)
    cloud = laspy.LasData(header)
    cloud.x = [700000, 700001, 700002, 700003]
    cloud.y = [6600000, 6600000, 6600000, 6600000]
    cloud.z = [100, 100, 110, 110]
    cloud.classification = [2, 2, 6, 67]
    cloud.write(path)


def test_class67_working_copy_keeps_original_and_coordinates(tmp_path):
    original, working = tmp_path / "original.laz", tmp_path / "working.laz"
    _cloud(original)
    before = original.read_bytes()
    report = prepare_points(original, working)
    assert original.read_bytes() == before
    assert list(laspy.read(working).classification) == [2, 2, 6, 6]
    assert report["original_class_counts"] == {"2": 2, "6": 1, "67": 1}
    assert np.array_equal(laspy.read(original).z, laspy.read(working).z)


def test_projected_crs_does_not_verify_vertical_reference():
    with pytest.raises(ReconstructionError):
        read_frame({"projected_crs": "EPSG:2154", "vertical_reference": "", "origin": [0, 0, 0]})
    frame = read_frame(
        {
            "projected_crs": "EPSG:2154",
            "vertical_reference": "IGN69",
            "origin": [700000, 6600000, 100],
        }
    )
    assert frame.to_local((700010, 6600020, 110)) == (10, 20, 10)


def test_no_crs_override_of_incompatible_points(tmp_path):
    original = tmp_path / "original.laz"
    _cloud(original, 27700)
    with pytest.raises(ReconstructionError) as error:
        prepare_points(original, tmp_path / "working.laz")
    assert error.value.code == "crs_mismatch"


def test_distant_ground_minimum_is_not_used(tmp_path):
    original = tmp_path / "original.laz"
    _cloud(original)
    unit = ReconstructionUnit(
        "unit",
        SourceIdentity(),
        "POLYGON ((10 10,20 10,20 20,10 20,10 10))",
        "POLYGON ((10 10,20 10,20 20,10 20,10 10))",
        "revision",
    )
    with pytest.raises(ReconstructionError) as error:
        ground_references(original, (unit,))
    assert error.value.code == "ground_unverified"


def test_truncated_crop_has_explicit_failure(tmp_path):
    original = tmp_path / "original.laz"
    original.write_bytes(b"truncated")
    with pytest.raises(ReconstructionError) as error:
        prepare_points(original, tmp_path / "working.laz")
    assert error.value.code == "points_invalid"


def test_extra_target_without_selected_source_is_rejected(tmp_path):
    from dataclasses import asdict

    from geogen.models import BuildingGroup
    from geogen.reconstruction.cache import ReconstructionCache, building_snapshot
    from geogen.reconstruction.pipeline import reconstruct_manifest

    polygon = (
        "POLYGON ((700000 6600000,700010 6600000,700010 6600010,700000 6600010,700000 6600000))"
    )
    units = [
        ReconstructionUnit(
            "selected", SourceIdentity(bdnb_group_ids=("group",)), polygon, polygon, "hash"
        ),
        ReconstructionUnit("unrelated", SourceIdentity(), polygon, polygon, "hash"),
    ]
    manifest = {
        "units": [asdict(unit) for unit in units],
        "buildings": building_snapshot({"FR:group": BuildingGroup("group")}),
    }
    with pytest.raises(ReconstructionError) as error:
        reconstruct_manifest(
            manifest,
            tmp_path / "manifest.json",
            tmp_path,
            None,
            cache=ReconstructionCache(tmp_path / "cache"),
            report={"units": []},
        )
    assert error.value.code == "selection_mismatch"
