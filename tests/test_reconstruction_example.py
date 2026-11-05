"""The published example must remain portable and internally consistent."""

from pathlib import Path

import laspy
from shapely import from_wkt

from geogen.reconstruction.cache import checked_artifact, content_hash, load_manifest
from geogen.reconstruction.pipeline import restore_units

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "roofer" / "manifest.json"


def test_published_example_inputs():
    manifest = load_manifest(EXAMPLE, addresses=("Synthetic reconstruction test, France",))
    assert "not real IGN or BDNB observations" in manifest["description"]
    units = restore_units(manifest["units"])
    assert {unit.reconstruction_id for unit in units} == {"box", "courtyard"}
    assert sum(from_wkt(unit.working_wkt).area for unit in units) == 436
    courtyard = next(unit for unit in units if unit.reconstruction_id == "courtyard")
    assert len(from_wkt(courtyard.working_wkt).interiors) == 1
    for unit in units:
        assert unit.geometry_revision == content_hash(unit.working_wkt)
        assert unit.links[0].method == "manual"
        assert unit.links[0].relationship == "one_to_one"
        assert unit.identity.bdtopo_ids == ()
    cloud = laspy.read(checked_artifact(EXAMPLE.parent, manifest["artifacts"]["lidar_original"]))
    assert len(cloud.points) == 5376
    assert cloud.header.parse_crs().to_epsg() == 2154
    assert set(cloud.classification) == {2, 6}
    assert set(cloud.z[cloud.classification == 2]) == {100}
    assert set(cloud.z[cloud.classification == 6]) == {110}
