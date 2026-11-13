"""Sourced envelope enrichment must not change reconstructed geometry."""

from __future__ import annotations

import json

import openstudio
import pytest

from geogen.envelope import Envelope, envelope_from_group
from geogen.geometry import Footprint
from geogen.models import BuildingGroup
from geogen.osm import (
    ModelError,
    build_model,
    build_model_from_geometry,
    save_model_from_geometry,
)
from geogen.reconstruction.thermal import make_thermal_scene
from geogen.reconstruction.types import GeometryScene
from geogen.thermal_properties import apply_envelope_properties
from tests.test_reconstruction_geometry import FRAME, prism

CLASSIFICATIONS = {
    "wall_insulation": "isolation par l'extérieur",
    "upper_floor_insulation": "isolé",
    "lower_floor_insulation": "non isolé",
    "glazing_type": "double vitrage",
    "roof_material": "ardoise",
}
NUMBERS = {
    "wall_u_value": 0.3,
    "roof_u_value": 0.2,
    "floor_u_value": 0.4,
    "window_u_value": 1.4,
    "window_solar_factor": 0.55,
}


def scene(*solids, **kwargs):
    return make_thermal_scene(GeometryScene(FRAME, solids or (prism(),)), **kwargs)


def record(values=None, **extra):
    return {
        "source_group_ids": ["bdnb-group-1"],
        "scope": "bdnb_group_aggregate",
        "values": CLASSIFICATIONS if values is None else values,
        **extra,
    }


def feature(obj, name):
    return obj.additionalProperties().getFeatureAsString(name).get()


def report(model, space="building"):
    return json.loads(feature(model.getBuilding(), "geogen_thermal_enrichment"))[space]


def test_classifications_are_retained_without_inventing_constructions():
    model = build_model_from_geometry(
        scene(), window_to_wall_ratio=0.25, envelope_properties={"building": record()}
    )
    assert not model.getConstructions()
    for surface in model.getSurfaces():
        expected = {
            "Wall": {"wall_insulation": CLASSIFICATIONS["wall_insulation"]},
            "RoofCeiling": {
                key: CLASSIFICATIONS[key] for key in ("upper_floor_insulation", "roof_material")
            },
            "Floor": {"lower_floor_insulation": CLASSIFICATIONS["lower_floor_insulation"]},
        }[surface.surfaceType()]
        data = json.loads(feature(surface, "geogen_envelope_properties"))
        assert data["values"] == expected
        assert data["source_group_ids"] == ["bdnb-group-1"]
        assert data["scope"] == "bdnb_group_aggregate"
        assert not surface.construction().is_initialized()
    for window in model.getSubSurfaces():
        assert feature(window, "geogen_glazing_type") == "double vitrage"
        assert feature(window, "geogen_thermal_scope") == "bdnb_group_aggregate"
        assert not window.construction().is_initialized()
    enrichment = report(model)
    assert enrichment["applied_counts"]["classified_surfaces"] == 6
    assert enrichment["applied_counts"]["classified_windows"] == 4
    assert enrichment["applied_counts"]["wall_constructions"] == 0
    assert {"field": "wall_u_value", "reason": "missing"} in enrichment["omissions"]
    assert enrichment["status"] == "partial"


def test_enrichment_is_scoped_by_exact_thermal_space_id():
    model = build_model_from_geometry(
        scene(prism("a"), prism("b", offset=(20, 0, 0))),
        window_to_wall_ratio=0.25,
        envelope_properties={"a": record({**CLASSIFICATIONS, **NUMBERS})},
    )
    for space in model.getSpaces():
        for surface in space.surfaces():
            assert surface.construction().is_initialized() == (space.nameString() == "a")
            assert surface.additionalProperties().hasFeature("geogen_wall_insulation") == (
                space.nameString() == "a" and surface.surfaceType() == "Wall"
            )
    assert report(model, "b")["status"] == "unavailable"
    assert not any(report(model, "b")["applied_counts"].values())


@pytest.mark.parametrize("mapping", [{"unknown": record()}, {"Building": record()}, []])
def test_invalid_space_mappings_are_rejected(mapping):
    with pytest.raises(ModelError, match="unknown thermal spaces"):
        build_model_from_geometry(scene(), envelope_properties=mapping)


@pytest.mark.parametrize(
    "entry",
    [None, {"values": []}, record(source_group_ids="group"), record(scope=0), record(conflicts=0)],
)
def test_invalid_source_record_shape_is_rejected(entry):
    with pytest.raises(ModelError):
        build_model_from_geometry(scene(), envelope_properties={"building": entry})


def test_numeric_materials_use_equivalent_resistance_and_sourced_optics():
    model = build_model_from_geometry(
        scene(), window_to_wall_ratio=0.25, envelope_properties={"building": record(NUMBERS)}
    )
    resistances = {"Wall": 1 / 0.3 - 0.17, "RoofCeiling": 1 / 0.2 - 0.14, "Floor": 1 / 0.4 - 0.21}
    for surface in model.getSurfaces():
        construction = surface.construction().get().to_Construction().get()
        material = construction.layers()[0].to_MasslessOpaqueMaterial().get()
        assert material.thermalResistance() == pytest.approx(resistances[surface.surfaceType()])
        assert json.loads(feature(construction, "geogen_thermal_assumption"))[
            "equivalent_layer_resistance_m2k_w"
        ] == pytest.approx(material.thermalResistance())
    for window in model.getSubSurfaces():
        construction = window.construction().get().to_Construction().get()
        material = construction.layers()[0].to_SimpleGlazing().get()
        assert material.uFactor() == pytest.approx(1.4)
        assert material.solarHeatGainCoefficient() == pytest.approx(0.55)
        assert not material.visibleTransmittance().is_initialized()
    enrichment = report(model)
    assert (
        enrichment["bdnb_numeric_field_mapping"]["floor_u_value"]
        == "u_plancher_bas_brut_deperditif"
    )
    assert enrichment["bdnb_numeric_field_mapping"]["window_u_value"] == "uw"
    roof_assumption = next(a for a in enrichment["assumptions"] if a["field"] == "roof_u_value")
    assert roof_assumption["bdnb_source_field"] == "u_plancher_haut_deperditif"
    assert roof_assumption["source_scope"] == "bdnb_group_aggregate"
    assert "no attic exclusion" in roof_assumption["mapping_assumption"]
    assert enrichment["applied_counts"] == {
        "classified_surfaces": 0,
        "classified_windows": 0,
        "wall_constructions": 4,
        "roof_constructions": 1,
        "floor_constructions": 1,
        "window_constructions": 4,
    }
    assert "ground_limitation" in next(
        a for a in enrichment["assumptions"] if a["field"] == "floor_u_value"
    )
    assert (
        next(a for a in enrichment["assumptions"] if a["field"] == "window_construction")[
            "visible_transmittance"
        ]
        is None
    )


@pytest.mark.parametrize("output_format", ["osm", "idf"])
def test_sourced_constructions_and_geometry_survive_native_reload(tmp_path, output_format):
    thermal = scene()
    baseline = build_model_from_geometry(thermal, window_to_wall_ratio=0.25)
    model = build_model_from_geometry(
        thermal,
        window_to_wall_ratio=0.25,
        envelope_properties={"building": record({**CLASSIFICATIONS, **NUMBERS})},
    )
    for name in ("geogen_geometry_hash", "geogen_geometry_manifest"):
        assert feature(model.getBuilding(), name) == feature(baseline.getBuilding(), name)
    path = save_model_from_geometry(model, tmp_path / f"model.{output_format}", output_format)
    if output_format == "osm":
        loaded = (
            openstudio.osversion.VersionTranslator().loadModel(openstudio.path(str(path))).get()
        )
        assert len(loaded.getMasslessOpaqueMaterials()) == 3
        assert len(loaded.getSimpleGlazings()) == 1
        assert all(surface.construction().is_initialized() for surface in loaded.getSurfaces())
        assert report(loaded) == report(model)
        assert feature(loaded.getBuilding(), "geogen_geometry_hash") == thermal.geometry_hash
    else:
        loaded = openstudio.IdfFile.load(openstudio.path(str(path))).get()
        materials = loaded.getObjectsByType(openstudio.IddObjectType("Material:NoMass"))
        assert len(materials) == 3
        assert sorted(obj.getDouble(2).get() for obj in materials) == pytest.approx(
            sorted((1 / 0.3 - 0.17, 1 / 0.2 - 0.14, 1 / 0.4 - 0.21))
        )
        glazing = loaded.getObjectsByType(
            openstudio.IddObjectType("WindowMaterial:SimpleGlazingSystem")
        )
        assert len(glazing) == 1
        assert glazing[0].getDouble(1).get() == pytest.approx(1.4)
        assert glazing[0].getDouble(2).get() == pytest.approx(0.55)
        assert not glazing[0].getDouble(3).is_initialized()


@pytest.mark.parametrize(
    "invalid", [False, 0, -0.5, float("nan"), float("inf"), "0.3", [0.2, 0.5], 10**400]
)
def test_invalid_numeric_values_leave_constructions_unset(invalid):
    model = build_model_from_geometry(
        scene(), envelope_properties={"building": record({"wall_u_value": invalid})}
    )
    assert not model.getConstructions()
    assert {"field": "wall_u_value", "reason": "invalid"} in report(model)["omissions"]


def test_unknown_classifications_and_conflicting_numeric_values_stay_unset():
    model = build_model_from_geometry(
        scene(),
        window_to_wall_ratio=0.25,
        envelope_properties={
            "building": record(
                {"wall_insulation": "inconnu", "glazing_type": "N/A", **NUMBERS},
                conflicts=list(NUMBERS),
            )
        },
    )
    assert not model.getConstructions()
    assert not report(model)["values"]
    assert all(
        {"field": field, "reason": "conflicting"} in report(model)["omissions"] for field in NUMBERS
    )
    assert not any(
        surface.additionalProperties().hasFeature("geogen_wall_insulation")
        for surface in model.getSurfaces()
    )


@pytest.mark.parametrize("status", ["conflicting", "incomplete_source_coverage", "missing"])
def test_source_consensus_omissions_are_preserved(status):
    decisions = {"wall_u_value": {"status": status, "source_values": {"a": 0.3, "b": None}}}
    links = [{"source_type": "bdnb_group", "source_id": "a", "attribute_scope": "group_aggregate"}]
    model = build_model_from_geometry(
        scene(),
        envelope_properties={
            "building": record(
                {"wall_u_value": 0.3, "wall_insulation": "not provided"},
                decisions=decisions,
                links=links,
            )
        },
    )
    assert not model.getConstructions()
    assert not report(model)["values"]
    assert {"field": "wall_u_value", "reason": status} in report(model)["omissions"]
    assert report(model)["decisions"] == decisions
    assert report(model)["links"] == links


def test_invalid_source_evidence_remains_reported_as_valid_json():
    model = build_model_from_geometry(
        scene(),
        envelope_properties={
            "building": record(
                {"wall_u_value": float("nan")},
                decisions={
                    "wall_u_value": {"status": "invalid", "source_values": {"a": float("nan")}}
                },
            )
        },
    )
    assert not model.getConstructions()
    assert report(model)["decisions"]["wall_u_value"]["source_values"]["a"] == {
        "invalid_numeric": "nan"
    }
    json.dumps(report(model), allow_nan=False)


@pytest.mark.parametrize(
    "values",
    [
        {"window_u_value": 1.4},
        {"window_solar_factor": 0.55},
        {"window_u_value": 1.4, "window_solar_factor": 1.1},
        {"window_u_value": 1.4, "window_solar_factor": -0.1},
    ],
)
def test_simple_glazing_requires_valid_u_and_solar_factor(values):
    model = build_model_from_geometry(
        scene(), window_to_wall_ratio=0.25, envelope_properties={"building": record(values)}
    )
    assert not model.getSimpleGlazings()
    assert all(not window.construction().is_initialized() for window in model.getSubSurfaces())
    assert {
        "field": "window_construction",
        "reason": "requires_both_u_value_and_solar_factor",
    } in report(model)["omissions"]


@pytest.mark.parametrize(
    "u_value, reason",
    [
        (10.0, "nonpositive_resistance_after_films"),
        (5e-324, "nonfinite_equivalent_resistance"),
    ],
)
def test_invalid_equivalent_resistance_is_not_assigned(u_value, reason):
    model = build_model_from_geometry(
        scene(), envelope_properties={"building": record({"wall_u_value": u_value})}
    )
    assert not model.getMasslessOpaqueMaterials()
    assert {
        "field": "wall_u_value",
        "reason": reason,
    } in report(
        model
    )["omissions"]


def test_enrichment_does_not_replace_or_classify_interfaces():
    thermal = scene(prism("a"), prism("b", offset=(10, 0, 0)), levels={"a": [3.0], "b": [3.0]})
    model = build_model_from_geometry(
        thermal,
        window_to_wall_ratio=0.25,
        envelope_properties={
            space.space_id: record({**CLASSIFICATIONS, **NUMBERS}) for space in thermal.spaces
        },
    )
    interfaces = [s for s in model.getSurfaces() if s.outsideBoundaryCondition() == "Surface"]
    assert interfaces
    for surface in interfaces:
        assert not surface.additionalProperties().hasFeature("geogen_envelope_properties")
        construction = surface.construction().get().to_Construction().get()
        assert "GEOMETRY ONLY" in construction.nameString()
        assert construction.layers()[0].to_MasslessOpaqueMaterial().get().thermalResistance() == 0.1
        assert not surface.subSurfaces()
    assert feature(model.getBuilding(), "geogen_geometry_hash") == thermal.geometry_hash


@pytest.mark.parametrize("second_z", [0.0, 10.0])
def test_building_stories_are_per_solid_and_use_local_elevations(tmp_path, second_z):
    thermal = scene(
        prism("a"),
        prism("b", offset=(20, 0, second_z)),
        levels={"a": [3.0], "b": [second_z + 3]},
    )
    model = build_model_from_geometry(thermal)
    assert len(model.getBuildingStorys()) == 4
    for space in model.getSpaces():
        source = next(s for s in thermal.spaces if s.space_id == space.nameString())
        story = space.buildingStory().get()
        assert story.nameString() == f"{source.solid_id} Storey {source.storey_number}"
        assert story.nominalZCoordinate().get() == source.nominal_z
        assert space.zOrigin() == 0.0
        assert story.nominalZCoordinate().get() < FRAME.origin[2]
    path = save_model_from_geometry(model, tmp_path / "stories.osm")
    loaded = openstudio.osversion.VersionTranslator().loadModel(openstudio.path(str(path))).get()
    assert len(loaded.getBuildingStorys()) == 4
    assert all(space.buildingStory().is_initialized() for space in loaded.getSpaces())
    assert feature(loaded.getBuilding(), "geogen_geometry_hash") == thermal.geometry_hash


def test_unsupported_subsurfaces_are_reported_without_fabrication():
    model = build_model_from_geometry(scene(), window_to_wall_ratio=0.25)
    for window in model.getSubSurfaces():
        assert window.setSubSurfaceType("GlassDoor")
    apply_envelope_properties(model, {"building": record(NUMBERS)})
    assert not model.getSimpleGlazings()
    assert sum(o["reason"] == "unsupported_subsurface" for o in report(model)["omissions"]) == 4
    assert all(not window.construction().is_initialized() for window in model.getSubSurfaces())


def test_sdk_material_rejection_is_reported_and_cleans_up(monkeypatch):
    monkeypatch.setattr(
        openstudio.model.MasslessOpaqueMaterial, "setThermalResistance", lambda *args: False
    )
    model = build_model_from_geometry(
        scene(), envelope_properties={"building": record({"wall_u_value": 0.3})}
    )
    assert not model.getConstructions()
    assert not model.getMasslessOpaqueMaterials()
    assert {
        "field": "wall_u_value",
        "reason": "sdk_rejected_material",
    } in report(
        model
    )["omissions"]


def test_shared_helper_enriches_legacy_spaces_without_changing_geometry():
    footprint = Footprint(
        name="legacy",
        ring=((652000, 6862000), (652000, 6862010), (652020, 6862010), (652020, 6862000)),
        height=6,
        storeys=2,
        envelope=Envelope(window_to_wall_ratio=0.25),
    )
    model = build_model([footprint])
    before = {
        surface.nameString(): [(p.x(), p.y(), p.z()) for p in surface.vertices()]
        for surface in model.getSurfaces()
    }
    reports = apply_envelope_properties(
        model,
        {space.nameString(): record({**CLASSIFICATIONS, **NUMBERS}) for space in model.getSpaces()},
    )
    assert len(reports) == 2
    assert sum(r["applied_counts"]["window_constructions"] for r in reports.values()) == 8
    assert all(
        not surface.additionalProperties().hasFeature("geogen_envelope_properties")
        for surface in model.getSurfaces()
        if surface.outsideBoundaryCondition() == "Surface"
    )
    assert before == {
        surface.nameString(): [(p.x(), p.y(), p.z()) for p in surface.vertices()]
        for surface in model.getSurfaces()
    }


def test_legacy_build_automatically_enriches_storeys_and_attic():
    group = BuildingGroup("bdnb-group-1", glazed_ratio=0.25, **CLASSIFICATIONS, **NUMBERS)
    footprint = Footprint(
        name="legacy",
        ring=((652000, 6862000), (652000, 6862010), (652020, 6862010), (652020, 6862000)),
        height=6,
        storeys=2,
        envelope=envelope_from_group(group),
    )
    model = build_model([footprint])
    reports = json.loads(feature(model.getBuilding(), "geogen_thermal_enrichment"))
    assert set(reports) == {space.nameString() for space in model.getSpaces()}
    assert len(reports) == 3
    assert reports["legacy Attic Space"]["applied_counts"]["roof_constructions"] == 4
    assert all(r["source_group_ids"] == [group.code] for r in reports.values())
    assert sum(r["applied_counts"]["window_constructions"] for r in reports.values()) == 8
    assert all(
        surface.construction().is_initialized()
        for surface in model.getSurfaces()
        if surface.outsideBoundaryCondition() == "Outdoors"
    )


def test_real_legacy_unknown_defaults_do_not_become_classifications():
    footprint = Footprint(
        name="undocumented",
        ring=((652000, 6862000), (652000, 6862010), (652020, 6862010), (652020, 6862000)),
        height=3,
        storeys=1,
        envelope=envelope_from_group(BuildingGroup("unknown")),
    )
    model = build_model([footprint])
    assert not model.getConstructions()
    assert not report(model, "undocumented Storey 1 Space")["values"]
    assert not any(
        surface.additionalProperties().hasFeature("geogen_envelope_properties")
        for surface in model.getSurfaces()
    )
