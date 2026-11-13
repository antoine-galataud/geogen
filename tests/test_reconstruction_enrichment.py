"""Automatic source enrichment must not substitute a roof or fabricate measurements."""

import json
import shutil
from dataclasses import replace

import openstudio
import pytest
from click.testing import CliRunner

from geogen.cli import main
from geogen.models import BuildingGroup
from geogen.reconstruction import enrichment as enrichment_module
from geogen.reconstruction.enrichment import (
    enrich_thermal_scene,
    glazing_parameters,
    solid_source_groups,
)
from geogen.reconstruction.types import (
    GeometryScene,
    SourceLink,
    ThermalScene,
    ThermalSpace,
)
from tests.test_reconstruction_geometry import FRAME, prism, profile_solid
from tests.test_reconstruction_policy import EXAMPLE, captured_reconstruction


def linked(solid, *groups):
    return replace(
        solid,
        links=tuple(
            SourceLink(
                solid.reconstruction_id,
                "bdnb_group",
                group,
                "relationship",
                "one_to_one",
                attribute_scope="group",
                confidence="confirmed",
            )
            for group in groups
        ),
    )


@pytest.fixture
def partition_stub(monkeypatch):
    captured = {}

    def partition(scene, *, levels, partition_report):
        captured.update(levels)
        spaces = []
        for solid in scene.solids:
            space_ids = []
            for level in range(len(levels.get(solid.solid_id, [])) + 1):
                space_id = f"{solid.solid_id}:{level}"
                space_ids.append(space_id)
                spaces.append(ThermalSpace(space_id, solid.solid_id, (), 1))
            partition_report[solid.solid_id] = {
                "status": "test_partition",
                "space_ids": space_ids,
            }
        return ThermalScene(scene.frame, tuple(spaces), "hash")

    monkeypatch.setattr("geogen.reconstruction.enrichment.make_thermal_scene", partition)
    return captured


def test_source_count_places_floors_below_roof_without_using_source_height(partition_stub):
    solid = linked(profile_solid(((0, 6), (5, 9), (10, 6))), "g")
    scene = GeometryScene(FRAME, (solid,))
    buildings = {"FR:g": BuildingGroup("g", storeys=3, height=99, wall_insulation="ITE")}
    report = {}
    thermal, properties, groups = enrich_thermal_scene(scene, buildings, report, storey_height=3)
    assert partition_stub == {solid.solid_id: [2, 4]}
    assert len(thermal.spaces) == 3
    assert groups == {solid.solid_id: ["g"]}
    assert all(p["values"]["wall_insulation"] == "ITE" for p in properties.values())
    assert all(p["source_group_ids"] == ["g"] for p in properties.values())
    assert report["storey_enrichment"][solid.solid_id]["count_source"] == "bdnb_group"
    assert scene.solids == (solid,)


def test_missing_count_uses_reported_geometry_prior(partition_stub):
    solid = linked(prism(height=9), "g")
    report = {}
    enrich_thermal_scene(
        GeometryScene(FRAME, (solid,)),
        {"FR:g": BuildingGroup("g")},
        report,
        storey_height=3,
    )
    assert partition_stub[solid.solid_id] == [3, 6]
    decision = report["storey_enrichment"][solid.solid_id]
    assert decision["count_source"] == "reconstructed_height_and_storey_height_prior"
    assert decision["storey_height_prior_m"] == 3


@pytest.mark.parametrize("second_count", [None, 4])
def test_ambiguous_count_does_not_apply_one_group_to_whole_solid(second_count, partition_stub):
    solid = linked(prism(), "a", "b")
    buildings = {
        "FR:a": BuildingGroup("a", storeys=2, wall_insulation="ITE"),
        "FR:b": BuildingGroup("b", storeys=second_count, wall_insulation="ITI"),
    }
    report = {}
    thermal, properties, _ = enrich_thermal_scene(
        GeometryScene(FRAME, (solid,)), buildings, report, storey_height=3
    )
    assert partition_stub == {}
    assert len(thermal.spaces) == 1
    assert "wall_insulation" not in next(iter(properties.values()))["values"]
    assert report["storey_enrichment"][solid.solid_id]["reason"] == (
        "conflicting_or_incomplete_storey_count"
    )


@pytest.mark.parametrize("count", [True, -1, 0, 1.5, 101, float("inf")])
def test_invalid_source_count_is_not_used(count, partition_stub):
    solid = linked(prism(), "g")
    report = {}
    enrich_thermal_scene(
        GeometryScene(FRAME, (solid,)),
        {"FR:g": BuildingGroup("g", storeys=count)},
        report,
        storey_height=3,
    )
    assert partition_stub == {}
    assert report["storey_enrichment"][solid.solid_id]["reason"] == (
        "unsupported_storey_count_or_height"
    )


def test_links_not_prefixes_or_untyped_construction_aliases_supply_properties():
    solid = linked(prism("unit:part"), "g")
    unresolved = replace(solid.links[0], source_id="other", confidence="unresolved")
    solid = replace(solid, links=(*solid.links, unresolved))
    buildings = {
        "FR:g": BuildingGroup("g", batiment_construction_id="not-a-typed-construction"),
        "FR:other": BuildingGroup("other", glazed_ratio=0.4, glazing_orientations=("Sud",)),
    }
    groups = solid_source_groups(GeometryScene(FRAME, (solid,)))
    assert groups == {"unit:part": ["g"]}
    assert solid_source_groups(GeometryScene(FRAME, (prism("unit:part"),))) == {"unit:part": []}
    thermal = ThermalScene(FRAME, (ThermalSpace("space", solid.solid_id, (), 1),), "hash")
    report = {}
    assert glazing_parameters(thermal, buildings, report, groups) == ({}, {})
    assert report["glazing_assumptions"][0]["source_group_ids"] == ["g"]


def test_glazing_is_repeated_per_storey_not_allocated_from_dwelling_area(
    partition_stub, monkeypatch
):
    from unittest.mock import Mock

    ratio_lookup = Mock(wraps=enrichment_module.window_to_wall_ratio)
    orientation_lookup = Mock(wraps=enrichment_module.glazing_orientations)
    monkeypatch.setattr(enrichment_module, "window_to_wall_ratio", ratio_lookup)
    monkeypatch.setattr(enrichment_module, "glazing_orientations", orientation_lookup)
    solid = linked(prism(), "g")
    buildings = {
        "FR:g": BuildingGroup(
            "g",
            storeys=2,
            glazed_areas=(("north", 10),),
            wall_area=40,
            glazing_orientations=("Nord",),
        )
    }
    report = {"units": []}
    thermal, _, groups = enrich_thermal_scene(
        GeometryScene(FRAME, (solid,)), buildings, report, storey_height=3
    )
    ratios, orientations = glazing_parameters(thermal, buildings, report, groups)
    assert ratios == {space.space_id: 0.25 for space in thermal.spaces}
    assert orientations == {space.space_id: ("north",) for space in thermal.spaces}
    assert all(d["modeled_orientations"] == ["north"] for d in report["glazing_assumptions"])
    ratio_lookup.assert_called_once_with(buildings["FR:g"])
    orientation_lookup.assert_called_once_with(buildings["FR:g"])


@pytest.mark.parametrize(
    "ratio, orientations, status",
    [(0.3, ("Sud",), "conflicting"), (None, (), "incomplete_source_coverage")],
)
def test_conflicting_or_missing_glazing_is_not_silently_spread(ratio, orientations, status):
    thermal = ThermalScene(FRAME, (ThermalSpace("space", "solid", (), 1),), "hash")
    buildings = {
        "FR:a": BuildingGroup("a", glazed_ratio=0.2, glazing_orientations=("Nord",)),
        "FR:b": BuildingGroup("b", glazed_ratio=ratio, glazing_orientations=orientations),
    }
    report = {"units": []}
    assert glazing_parameters(thermal, buildings, report, {"solid": ["a", "b"]}) == ({}, {})
    decision = report["glazing_assumptions"][0]
    assert decision["source_decision"]["status"] == status
    assert decision["orientation_decision"]["status"] == status


def test_cli_always_attempts_storeys_envelope_and_windows(monkeypatch, tmp_path):
    from geogen.reconstruction.runtime import DockerRuntime

    manifest = json.loads(EXAMPLE.read_text())
    for building in manifest["buildings"]:
        building["data"].update(
            storeys=2,
            glazed_ratio=0.2,
            wall_insulation="ITE",
            wall_u_value=0.4,
            roof_u_value=0.25,
            floor_u_value=0.5,
            window_u_value=1.6,
            window_solar_factor=0.6,
        )
    source = EXAMPLE.parent / manifest["artifacts"]["lidar_original"]["path"]
    shutil.copyfile(source, tmp_path / source.name)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(DockerRuntime, "preflight", lambda *a: {"image": "test"})
    monkeypatch.setattr(DockerRuntime, "validate_cityjson", lambda *a: {"validity": True})
    monkeypatch.setattr("geogen.reconstruction.roofer.reconstruct", captured_reconstruction)
    monkeypatch.setattr(
        "geogen.reconstruction.evidence.assess_evidence",
        lambda scene, units, path: {
            unit.reconstruction_id: {
                "fidelity_status": "evidence_passed",
                "metrics": {},
                "reasons": [],
            }
            for unit in units
        },
    )
    output = tmp_path / "model.osm"
    result = CliRunner().invoke(
        main,
        [
            "Synthetic reconstruction test, France",
            "--country",
            "FR",
            "--roof-source",
            "roofer",
            "--reconstruction-input",
            str(path),
            "--reconstruction-cache",
            str(tmp_path / "cache"),
            "--output",
            str(output),
            "--svg-output",
            str(tmp_path / "model.svg"),
            "--json-output",
            str(tmp_path / "metadata.json"),
        ],
    )
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    report = json.loads(output.with_suffix(".reconstruction.json").read_text())
    assert report["outputs_published"]
    assert len(report["storey_enrichment"]) == 2
    assert all(d["modeled_space_count"] == 2 for d in report["storey_enrichment"].values())
    assert report["envelope_enrichment"]
    assert all(
        entry["applied_counts"]["wall_constructions"] > 0
        and entry["applied_counts"]["window_constructions"] > 0
        for entry in report["envelope_enrichment"].values()
    )
    assert report["window_count"] > 0
    model = openstudio.osversion.VersionTranslator().loadModel(openstudio.path(str(output))).get()
    assert len(model.getSpaces()) == len(model.getBuildingStorys()) == 4
    assert sum(space.volume() for space in model.getSpaces()) == pytest.approx(4360)
    assert all(space.buildingStory().is_initialized() for space in model.getSpaces())
    assert all(window.construction().is_initialized() for window in model.getSubSurfaces())
    translated = openstudio.energyplus.ForwardTranslator()
    translated.translateModel(model)
    assert not translated.errors()
    assert any(surface.outsideBoundaryCondition() == "Surface" for surface in model.getSurfaces())
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["reconstruction"]["scene_geometry_hash"] == report["scene_geometry_hash"]
