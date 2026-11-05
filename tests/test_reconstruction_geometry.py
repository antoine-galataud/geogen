"""Synthetic gates for the experimental solid route, not a fidelity benchmark."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, replace
from pathlib import Path

import openstudio
import pytest

from geogen.envelope import cardinal_point
from geogen.osm import (
    ModelError,
    build_model_from_geometry,
    save_model_from_geometry,
    windows_from_geometry_model,
)
from geogen.reconstruction.cityjson import import_cityjson
from geogen.reconstruction.thermal import make_thermal_scene, validate_thermal_scene
from geogen.reconstruction.types import (
    GeometryFrame,
    GeometryScene,
    ReconstructionError,
    Solid,
    Surface,
)
from geogen.reconstruction.validate import face_plane, plane_deviation, validate_scene
from geogen.svg import SvgError, save_svg_from_geometry

FRAME = GeometryFrame("EPSG:2154", "IGN69", (652000.0, 6862000.0, 35.0))


def prism(name="building", *, width=10.0, depth=10.0, height=6.0, holes=(), offset=(0.0, 0.0, 0.0)):
    outer = ((0.0, 0.0), (width, 0.0), (width, depth), (0.0, depth))

    def ring3(ring, z):
        return tuple((x + offset[0], y + offset[1], z + offset[2]) for x, y in ring)

    faces = [
        Surface(
            f"{name}:floor",
            ring3(outer[::-1], 0),
            tuple(ring3(r[::-1], 0) for r in holes),
            "GroundSurface",
        ),
        Surface(
            f"{name}:roof",
            ring3(outer, height),
            tuple(ring3(r, height) for r in holes),
            "RoofSurface",
        ),
    ]
    for ri, ring in enumerate((outer, *holes)):
        for i, (a, b) in enumerate(zip(ring, (*ring[1:], ring[0]))):
            points = (*ring3((a, b), 0), *ring3((b, a), height))
            faces.append(Surface(f"{name}:wall:{ri}:{i}", points, semantic="WallSurface"))
    return Solid(
        name,
        name,
        tuple(replace(s, source_face_id=s.face_id, source_object_id=name) for s in faces),
    )


def profile_solid(profile, name="profile"):
    depth = 10.0
    width = profile[-1][0]
    front = ((0.0, 0.0, 0.0), (width, 0.0, 0.0), *tuple((x, 0.0, z) for x, z in profile[::-1]))
    faces = [
        Surface(f"{name}:front", front, semantic="WallSurface"),
        Surface(
            f"{name}:back", tuple((x, depth, z) for x, _, z in front[::-1]), semantic="WallSurface"
        ),
        Surface(
            f"{name}:ground",
            ((0.0, depth, 0.0), (width, depth, 0.0), (width, 0.0, 0.0), (0.0, 0.0, 0.0)),
            semantic="GroundSurface",
        ),
        Surface(
            f"{name}:left",
            (
                (0.0, depth, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, profile[0][1]),
                (0.0, depth, profile[0][1]),
            ),
            semantic="WallSurface",
        ),
        Surface(
            f"{name}:right",
            (
                (width, 0.0, 0.0),
                (width, depth, 0.0),
                (width, depth, profile[-1][1]),
                (width, 0.0, profile[-1][1]),
            ),
            semantic="WallSurface",
        ),
    ]
    for i, ((x0, z0), (x1, z1)) in enumerate(zip(profile, profile[1:])):
        faces.append(
            Surface(
                f"{name}:top:{i}",
                ((x0, 0.0, z0), (x1, 0.0, z1), (x1, depth, z1), (x0, depth, z0)),
                semantic="WallSurface" if x0 == x1 else "RoofSurface",
            )
        )
    return Solid(name, name, tuple(replace(s, source_face_id=s.face_id) for s in faces))


def cityjson(solids, *, lod="2.2", transform=None):
    vertices, indices, objects = [], {}, {}
    definitions = [
        {"type": semantic} for semantic in ("GroundSurface", "RoofSurface", "WallSurface")
    ]

    def ring_indices(ring):
        result = []
        for point in ring:
            if point not in indices:
                indices[point] = len(vertices)
                vertices.append(list(point))
            result.append(indices[point])
        return result

    for solid in solids:
        boundaries = [
            [
                [ring_indices(r) for r in (surface.exterior, *surface.interiors)]
                for surface in solid.surfaces
            ]
        ]
        values = [
            [
                next(i for i, d in enumerate(definitions) if d["type"] == s.semantic)
                for s in solid.surfaces
            ]
        ]
        objects[solid.reconstruction_id] = {
            "type": "Building",
            "geometry": [
                {
                    "type": "Solid",
                    "lod": lod,
                    "boundaries": boundaries,
                    "semantics": {"surfaces": definitions, "values": values},
                }
            ],
        }
    doc = {"type": "CityJSON", "version": "2.0", "CityObjects": objects, "vertices": vertices}
    if transform:
        doc["transform"] = transform
    return doc


def write_json(path, doc):
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


@pytest.fixture
def courtyard():
    return prism(holes=(((3.0, 3.0), (3.0, 7.0), (7.0, 7.0), (7.0, 3.0)),))


@pytest.mark.parametrize(
    "profile,volume",
    [
        (((0.0, 4.0), (5.0, 4.0), (5.0, 7.0), (10.0, 7.0)), 550.0),
        (((0.0, 7.0), (5.0, 4.0), (10.0, 7.0)), 550.0),
        (((0.0, 4.0), (5.0, 7.0), (10.0, 4.0)), 550.0),
    ],
)
def test_steps_valleys_and_ridges_keep_their_exact_volume(profile, volume):
    solid = profile_solid(profile)
    scene = validate_scene(GeometryScene(FRAME, (solid,)))
    assert scene.solids[0].volume == pytest.approx(volume)
    thermal = make_thermal_scene(scene)
    assert len(thermal.spaces) == 1
    assert thermal.spaces[0].volume == pytest.approx(volume)
    source_roofs = [s for s in solid.surfaces if s.semantic == "RoofSurface"]
    thermal_roofs = [
        s
        for space in thermal.spaces
        for s in space.surfaces
        if s.source_face_id in {r.face_id for r in source_roofs}
    ]
    assert sum(face_plane(s)[4].area for s in thermal_roofs) == pytest.approx(
        sum(face_plane(s)[4].area for s in source_roofs)
    )


def test_transform_applied_once_and_feature_vertices_are_local(tmp_path):
    header = {
        "type": "CityJSON",
        "version": "2.0",
        "CityObjects": {},
        "vertices": [],
        "transform": {"scale": [0.1, 0.1, 0.1], "translate": list(FRAME.origin)},
    }
    features = []
    for name, shift in (("building", 0), ("second", 20)):
        local = cityjson([prism(name, offset=(shift, 0.0, 0.0))])
        features.append(
            {
                "type": "CityJSONFeature",
                "id": name,
                "CityObjects": local["CityObjects"],
                "vertices": [[v * 10 for v in p] for p in local["vertices"]],
            }
        )
    path = tmp_path / "sequence.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in (header, *features)), encoding="utf-8")
    scene = validate_scene(import_cityjson(path, FRAME))
    assert [s.reconstruction_id for s in scene.solids] == ["building", "second"]
    assert [s.volume for s in scene.solids] == pytest.approx([600.0, 600.0])
    assert min(p[0] for s in scene.solids for f in s.surfaces for p in f.exterior) == 0
    assert max(p[0] for s in scene.solids for f in s.surfaces for p in f.exterior) == 30


def test_building_part_and_multisolid_semantic_nesting(tmp_path):
    doc = cityjson([prism()])
    root = doc["CityObjects"].pop("building")
    geometry = root["geometry"][0]
    geometry["type"] = "MultiSolid"
    geometry["boundaries"] = [geometry["boundaries"]]
    geometry["semantics"]["values"] = [geometry["semantics"]["values"]]
    doc["CityObjects"] = {
        "building": {"type": "Building", "children": ["part"]},
        "part": {"type": "BuildingPart", "parents": ["building"], "geometry": [geometry]},
    }
    scene = import_cityjson(
        write_json(tmp_path / "parts.json", doc), replace(FRAME, origin=(0.0, 0.0, 0.0))
    )
    assert scene.solids[0].reconstruction_id == "building"
    assert {s.source_object_id for s in scene.solids[0].surfaces} == {"part"}
    assert validate_scene(scene).solids[0].volume == pytest.approx(600.0)


def test_courtyard_holes_survive_import_and_export(tmp_path, courtyard):
    doc = cityjson([courtyard], transform={"scale": [1, 1, 1], "translate": list(FRAME.origin)})
    scene = validate_scene(import_cityjson(write_json(tmp_path / "court.json", doc), FRAME))
    assert scene.solids[0].volume == pytest.approx(504.0)
    assert sum(len(s.interiors) for s in scene.solids[0].surfaces) == 2
    thermal = make_thermal_scene(scene)
    assert [s.volume for s in thermal.spaces] == pytest.approx([504.0])
    model = build_model_from_geometry(thermal)
    assert sum(s.volume() for s in model.getSpaces()) == pytest.approx(504.0)
    for surface_type in ("Floor", "RoofCeiling"):
        assert sum(
            s.grossArea() for s in model.getSurfaces() if s.surfaceType() == surface_type
        ) == pytest.approx(84.0)
    svg = save_svg_from_geometry(thermal, tmp_path / "court.svg", azimuth=15, elevation=40)
    root = ET.parse(svg).getroot()
    assert root.attrib["data-geometry-hash"] == thermal.geometry_hash
    assert (
        model.getBuilding().additionalProperties().getFeatureAsString("geogen_geometry_hash").get()
        == thermal.geometry_hash
    )
    assert "unverified" in svg.read_text()


def test_partial_party_walls_are_split_and_reciprocal():
    a = prism("a")
    b = prism("b", depth=5.0, height=3.0, offset=(10.0, 2.0, 0.0))
    thermal = make_thermal_scene(GeometryScene(FRAME, (a, b)))
    shared = [s for space in thermal.spaces for s in space.surfaces if s.boundary == "surface"]
    assert len(shared) == 2
    assert all(face_plane(s)[4].area == pytest.approx(15.0) for s in shared)
    assert shared[0].interface_id == shared[1].interface_id
    assert {s.adjacent_space_id for s in shared} == {"a", "b"}
    assert sum(space.volume for space in thermal.spaces) == pytest.approx(750.0)
    model = build_model_from_geometry(thermal)
    assert sum(s.outsideBoundaryCondition() == "Surface" for s in model.getSurfaces()) == 2


def test_collinear_shared_edges_are_noded():
    solid = prism()
    roof = solid.surfaces[1]
    points = (roof.exterior[0], (5.0, 0.0, 6.0), *roof.exterior[1:])
    solid = replace(
        solid, surfaces=(solid.surfaces[0], replace(roof, exterior=points), *solid.surfaces[2:])
    )
    scene = validate_scene(GeometryScene(FRAME, (solid,)))
    assert scene.solids[0].volume == pytest.approx(600.0)
    assert sum((5.0, 0.0, 6.0) in s.exterior for s in scene.solids[0].surfaces) == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-face",
        "nonplanar",
        "reverse",
        "bad-hole",
        "bowtie",
        "duplicate-face",
    ],
)
def test_malformed_solids_fail_closed(mutation, courtyard):
    faces = list(courtyard.surfaces)
    if mutation == "missing-face":
        faces.pop()
    elif mutation == "nonplanar":
        points = list(faces[1].exterior)
        points[0] = (0.0, 0.0, 7.0)
        faces[1] = replace(faces[1], exterior=tuple(points))
    elif mutation == "reverse":
        faces = [
            replace(s, exterior=s.exterior[::-1], interiors=tuple(r[::-1] for r in s.interiors))
            for s in faces
        ]
    elif mutation == "bad-hole":
        faces[1] = replace(
            faces[1], interiors=(tuple((x + 20, y, z) for x, y, z in faces[1].interiors[0]),)
        )
    elif mutation == "bowtie":
        ring = faces[1].exterior
        faces[1] = replace(faces[1], exterior=(ring[0], ring[2], ring[1], ring[3]))
    else:
        faces.append(replace(faces[1], face_id="duplicate"))
    with pytest.raises(ReconstructionError):
        validate_scene(GeometryScene(FRAME, (replace(courtyard, surfaces=tuple(faces)),)))


@pytest.mark.parametrize(
    "mutation",
    [
        "lod",
        "surface-type",
        "vertex",
        "semantics",
        "hierarchy",
        "parent-child",
        "transform",
        "crs",
    ],
)
def test_malformed_cityjson_is_not_silently_downgraded(tmp_path, mutation):
    doc = cityjson([prism()])
    geometry = doc["CityObjects"]["building"]["geometry"][0]
    if mutation == "lod":
        geometry["lod"] = "1.2"
    elif mutation == "surface-type":
        geometry["type"] = "MultiSurface"
    elif mutation == "vertex":
        geometry["boundaries"][0][0][0][0] = -1
    elif mutation == "semantics":
        geometry["semantics"]["values"] = geometry["semantics"]["values"][0]
    elif mutation == "hierarchy":
        doc["CityObjects"]["building"]["parents"] = ["building"]
    elif mutation == "parent-child":
        doc["CityObjects"]["building"]["children"] = ["part"]
        doc["CityObjects"]["part"] = {"type": "BuildingPart", "geometry": [geometry]}
    elif mutation == "transform":
        doc["transform"] = {"scale": [0, 1, 1], "translate": [0, 0, 0]}
    else:
        doc["metadata"] = {"referenceSystem": "EPSG:4326"}
    with pytest.raises(ReconstructionError):
        import_cityjson(write_json(tmp_path / "invalid.json", doc), FRAME)


def test_overlapping_and_nested_solids_rejected():
    for other in (
        prism("b", offset=(5.0, 0.0, 0.0)),
        prism("b", width=2.0, depth=2.0, height=2.0, offset=(2.0, 2.0, 2.0)),
    ):
        with pytest.raises(ReconstructionError):
            validate_scene(GeometryScene(FRAME, (prism("a"), other)))


def test_unresolved_void_shells_explicitly_rejected():
    solid = prism()
    solid = replace(solid, interior_shells=(prism("void").surfaces,))
    with pytest.raises(ReconstructionError, match="interior shells"):
        validate_scene(GeometryScene(FRAME, (solid,)))


def test_geometry_hash_rejects_mutation():
    scene = make_thermal_scene(GeometryScene(FRAME, (prism(),)))
    surface = scene.spaces[0].surfaces[0]
    modified = replace(surface, boundary="adiabatic")
    scene = replace(
        scene,
        spaces=(replace(scene.spaces[0], surfaces=(modified, *scene.spaces[0].surfaces[1:])),),
    )
    with pytest.raises(ReconstructionError, match="hash"):
        validate_thermal_scene(scene)


@pytest.mark.parametrize("output_format", ["osm", "idf"])
def test_strict_export_reloads_native_files(tmp_path, output_format, courtyard):
    scene = make_thermal_scene(GeometryScene(FRAME, (courtyard,)))
    model = build_model_from_geometry(scene)
    path = save_model_from_geometry(model, tmp_path / f"building.{output_format}", output_format)
    assert path.exists()
    assert scene.geometry_hash in path.read_text(encoding="utf-8")


def test_strict_export_rejects_translation_errors(tmp_path, monkeypatch):
    scene = make_thermal_scene(GeometryScene(FRAME, (prism(),)))
    model = build_model_from_geometry(scene)
    real = openstudio.energyplus.ForwardTranslator

    class BadTranslator:
        def translateModel(self, model):
            return real().translateModel(model)

        def errors(self):
            return [type("Error", (), {"logMessage": lambda self: "synthetic translation error"})()]

    monkeypatch.setattr(openstudio.energyplus, "ForwardTranslator", BadTranslator)
    with pytest.raises(ModelError, match="synthetic translation error"):
        save_model_from_geometry(model, tmp_path / "bad.osm")
    assert not (tmp_path / "bad.osm").exists()


def test_solid_export_does_not_call_roof_inference(tmp_path, monkeypatch):
    import geogen.osm
    import geogen.svg

    def forbidden(*args, **kwargs):
        raise AssertionError("Proxy roof generation is forbidden")

    monkeypatch.setattr(geogen.osm, "_add_roof", forbidden)
    monkeypatch.setattr(geogen.svg, "_roof_drawables", forbidden)
    source = profile_solid(((0.0, 7.0), (5.0, 4.0), (10.0, 7.0)))
    scene = make_thermal_scene(GeometryScene(FRAME, (source,)))
    model = build_model_from_geometry(scene)
    assert model.getSpaces()[0].volume() == pytest.approx(550.0)
    assert save_svg_from_geometry(scene, tmp_path / "valley.svg").exists()


def test_requested_lod_selected_without_a_fallback(tmp_path):
    doc = cityjson([prism()], lod="1.2")
    geometry = json.loads(json.dumps(doc["CityObjects"]["building"]["geometry"][0]))
    geometry["lod"] = "2.2"
    doc["CityObjects"]["building"]["geometry"].append(geometry)
    scene = import_cityjson(write_json(tmp_path / "lod.json", doc), FRAME)
    assert len(scene.solids) == 1
    assert scene.solids[0].solid_id.endswith(":1:0")


def test_multisolid_import_uses_separate_shell_lists(tmp_path):
    doc = cityjson([prism("a"), prism("b", offset=(20.0, 0.0, 0.0))])
    a, b = [doc["CityObjects"][name]["geometry"][0] for name in ("a", "b")]
    a["type"] = "MultiSolid"
    a["boundaries"] = [a["boundaries"], b["boundaries"]]
    a["semantics"]["values"] = [a["semantics"]["values"], b["semantics"]["values"]]
    doc["CityObjects"].pop("b")
    scene = validate_scene(import_cityjson(write_json(tmp_path / "multi.json", doc), FRAME))
    assert len(scene.solids) == 2
    assert {s.reconstruction_id for s in scene.solids} == {"a"}
    assert sum(s.volume for s in scene.solids) == pytest.approx(1200.0)


def test_closed_but_self_intersecting_triangle_shell_rejected():
    solid = prism()
    triangles = []
    for face in solid.surfaces:
        points = tuple((5.0, 5.0, -2.0) if p == (10.0, 10.0, 6.0) else p for p in face.exterior)
        for i in (1, 2):
            triangles.append(
                replace(
                    face,
                    face_id=f"{face.face_id}:{i}",
                    exterior=(points[0], points[i], points[i + 1]),
                )
            )
    with pytest.raises(ReconstructionError, match="intersection"):
        validate_scene(GeometryScene(FRAME, (replace(solid, surfaces=tuple(triangles)),)))


def test_strict_save_rejects_shifted_geometry(tmp_path):
    scene = make_thermal_scene(GeometryScene(FRAME, (prism(),)))
    model = build_model_from_geometry(scene)
    surface = model.getSurfaces()[0]
    normal = surface.outwardNormal()
    surface.setVertices(
        openstudio.Point3dVector(
            [
                openstudio.Point3d(p.x() + normal.x(), p.y() + normal.y(), p.z() + normal.z())
                for p in surface.vertices()
            ]
        )
    )
    with pytest.raises(ModelError):
        save_model_from_geometry(model, tmp_path / "shifted.osm")


def test_pinned_roofer_fixture_imports_and_keeps_courtyard():
    path = Path(__file__).parent / "fixtures/reconstruction/roofer-box-courtyard.city.jsonl"
    frame = GeometryFrame("EPSG:2154", "IGN69", (700000.0, 6600000.0, 100.0))
    scene = validate_scene(import_cityjson(path, frame))
    assert {s.reconstruction_id: s.volume for s in scene.solids} == pytest.approx(
        {
            "box": 1000.0,
            "courtyard": 3360.0,
        }
    )
    assert sum(len(s.interiors) for solid in scene.solids for s in solid.surfaces) == 2


def test_duplicate_json_members_are_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"type":"CityJSON","type":"CityJSONFeature"}', encoding="utf-8")
    with pytest.raises(ReconstructionError, match="Duplicate JSON"):
        import_cityjson(path, FRAME)


def test_model_records_explicit_geometry_only_interface_material():
    # Two adjacent units share a reciprocal party wall.
    scene = make_thermal_scene(
        GeometryScene(FRAME, (prism("a"), prism("b", offset=(10.0, 0.0, 0.0))))
    )
    model = build_model_from_geometry(scene)
    assumption = (
        model.getBuilding()
        .additionalProperties()
        .getFeatureAsString("geogen_construction_assumption")
        .get()
    )
    assert "not simulation-ready" in assumption
    assert "R=0.1" in assumption
    interfaces = [s for s in model.getSurfaces() if s.outsideBoundaryCondition() == "Surface"]
    assert len(interfaces) == 2
    assert all(s.construction().is_initialized() for s in interfaces)


def test_strict_save_rejects_space_translation(tmp_path):
    scene = make_thermal_scene(GeometryScene(FRAME, (prism(),)))
    model = build_model_from_geometry(scene)
    model.getSpaces()[0].setXOrigin(1.0)
    with pytest.raises(ModelError, match="geometry"):
        save_model_from_geometry(model, tmp_path / "shifted-space.osm")


def test_explicit_window_ratio_is_retained_in_strict_idf(tmp_path):
    scene = make_thermal_scene(GeometryScene(FRAME, (prism(),)))
    model = build_model_from_geometry(scene, window_to_wall_ratio=0.3)
    assert len(model.getSubSurfaces()) == 4
    assert sum(s.grossArea() for s in model.getSubSurfaces()) == pytest.approx(72.0)
    assert save_model_from_geometry(model, tmp_path / "glazed.idf", output_format="idf").exists()


def test_per_space_glazing_overrides_global_without_duplicating_geometry(tmp_path):
    scene = make_thermal_scene(
        GeometryScene(
            FRAME,
            (
                prism("a"),
                prism("b", offset=(20.0, 0.0, 0.0)),
            ),
        )
    )
    model = build_model_from_geometry(scene, window_to_wall_ratio=0.2, window_ratios={"a": 0.4})
    areas = {
        space.nameString(): sum(
            window.grossArea() for surface in space.surfaces() for window in surface.subSurfaces()
        )
        for space in model.getSpaces()
    }
    assert areas == pytest.approx({"a": 96.0, "b": 48.0})
    assert sum(space.volume() for space in model.getSpaces()) == pytest.approx(1200.0)
    assert save_model_from_geometry(model, tmp_path / "per-space.idf", output_format="idf").exists()


def test_per_space_zero_ratio_disables_global_glazing():
    scene = make_thermal_scene(GeometryScene(FRAME, (prism(),)))
    model = build_model_from_geometry(
        scene, window_to_wall_ratio=0.3, window_ratios={"building": 0.0}
    )
    assert len(model.getSubSurfaces()) == 0


@pytest.mark.parametrize(
    "ratios", [{"unknown": 0.2}, {"building": float("nan")}, {"building": 1.0}]
)
def test_invalid_per_space_window_ratios_fail_closed(ratios):
    scene = make_thermal_scene(GeometryScene(FRAME, (prism(),)))
    with pytest.raises(ModelError):
        build_model_from_geometry(scene, window_ratios=ratios)


def test_svg_renders_actual_native_windows_without_inference(tmp_path, monkeypatch):
    import geogen.svg

    def forbidden(*args, **kwargs):
        raise AssertionError("Approximate windows are forbidden")

    monkeypatch.setattr(geogen.svg, "_window_polygon", forbidden)
    scene = make_thermal_scene(GeometryScene(FRAME, (prism(),)))
    model = build_model_from_geometry(scene, window_ratios={"building": 0.3})
    windows = windows_from_geometry_model(model)
    expected = {
        tuple((p.x(), p.y(), p.z()) for p in window.vertices()) for window in model.getSubSurfaces()
    }
    assert {ring for rings in windows.values() for ring in rings} == expected
    path = save_svg_from_geometry(scene, tmp_path / "actual-windows.svg", windows=windows)
    root = ET.parse(path).getroot()
    metadata = json.loads(root.find("{http://www.w3.org/2000/svg}metadata").text)
    assert metadata["window_count"] == 4
    assert metadata["window_source"] == "actual_model_geometry"
    assert root.attrib["data-geometry-hash"] == scene.geometry_hash
    assert root.findall(".//{http://www.w3.org/2000/svg}path[@class='window']")


def test_svg_rejects_windows_not_on_their_host_wall(tmp_path):
    scene = make_thermal_scene(GeometryScene(FRAME, (prism(),)))
    model = build_model_from_geometry(scene, window_to_wall_ratio=0.3)
    windows = windows_from_geometry_model(model)
    face_id = next(iter(windows))
    windows[face_id] = tuple(
        tuple((x + 100.0, y, z) for x, y, z in ring) for ring in windows[face_id]
    )
    with pytest.raises(SvgError, match="outside"):
        save_svg_from_geometry(scene, tmp_path / "bad-windows.svg", windows=windows)


def test_native_faces_preserve_explicit_legacy_provenance():
    solid = prism()
    solid = replace(
        solid,
        surfaces=tuple(
            replace(surface, provenance=("legacy_inferred",)) for surface in solid.surfaces
        ),
    )
    scene = make_thermal_scene(GeometryScene(FRAME, (solid,)))
    model = build_model_from_geometry(scene)
    assert all(
        json.loads(surface.additionalProperties().getFeatureAsString("geogen_provenance").get())
        == ["legacy_inferred"]
        for surface in model.getSurfaces()
    )


@pytest.mark.parametrize("output_format", ["osm", "idf"])
def test_offline_cli_exports_real_fixture_with_actual_glazing(tmp_path, monkeypatch, output_format):
    """Exercise CLI/publication/export; replay the already-captured native result."""
    import shutil

    from click.testing import CliRunner

    from geogen.bdnb import DEFAULT_BASE_URL
    from geogen.cli import main
    from geogen.models import BuildingGroup
    from geogen.reconstruction.cache import (
        artifact,
        building_snapshot,
        checked_artifact,
        content_hash,
        write_json,
    )
    from geogen.reconstruction.types import ReconstructionUnit, SourceIdentity, SourceLink

    frame = GeometryFrame("EPSG:2154", "IGN69", (700000.0, 6600000.0, 100.0))
    raw = tmp_path / "capture.city.jsonl"
    shutil.copyfile(
        Path(__file__).parent / "fixtures/reconstruction/roofer-box-courtyard.city.jsonl", raw
    )
    buildings = {
        f"FR:{name}": BuildingGroup(
            name,
            crs="EPSG:2154",
            height=10.0,
            storeys=2,
            ground_elevation=100.0,
            glazed_ratio=ratio,
            geometry=wkt,
        )
        for name, ratio, wkt in (
            (
                "box",
                0.2,
                "POLYGON ((700000 6600000,700010 6600000,700010 6600010,700000 6600010,700000 6600000))",
            ),
            (
                "courtyard",
                0.3,
                "POLYGON ((700020 6600000,700040 6600000,700040 6600020,700020 6600020,700020 6600000), (700026 6600006,700026 6600014,700034 6600014,700034 6600006,700026 6600006))",
            ),
        )
    }
    manifest = write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": "1.0",
            "request": {
                "addresses": ["1 rue Paris"],
                "country": "FR",
                "max_buildings": 10,
                "provider_endpoint_hash": content_hash(DEFAULT_BASE_URL),
            },
            "buildings": building_snapshot(buildings),
            "artifacts": {"captured_native_geometry": artifact(raw, tmp_path)},
        },
    )

    def preflight(runtime, work):
        assert runtime.offline
        return {"mode": "captured_fixture_adapter"}

    def replay(manifest, manifest_path, work, runtime, *, report, **kwargs):
        path = checked_artifact(
            manifest_path.parent, manifest["artifacts"]["captured_native_geometry"]
        )
        scene = validate_scene(import_cityjson(path, frame))
        units = tuple(
            ReconstructionUnit(
                building.code,
                SourceIdentity(bdnb_group_ids=(building.code,)),
                building.geometry,
                building.geometry,
                "captured-fixture",
                outline_type="roofprint",
                links=(
                    SourceLink(
                        building.code,
                        "bdnb_group",
                        building.code,
                        "fixture",
                        "one_to_one",
                        attribute_scope="group",
                        confidence="verified",
                    ),
                ),
            )
            for building in buildings.values()
        )
        report["units"] = [
            {
                "reconstruction_id": unit.reconstruction_id,
                "geometry_status": "fixture_validated",
                "fidelity_status": "unverified",
                "metrics": {},
                "links": [asdict(link) for link in unit.links],
            }
            for unit in units
        ]
        links = {unit.reconstruction_id: unit.links for unit in units}
        scene = replace(
            scene,
            solids=tuple(
                replace(solid, links=links[solid.reconstruction_id]) for solid in scene.solids
            ),
        )
        return scene, units

    monkeypatch.setattr("geogen.reconstruction.runtime.DockerRuntime.preflight", preflight)
    monkeypatch.setattr("geogen.reconstruction.command.reconstruct_manifest", replay)
    monkeypatch.setattr(
        "geogen.cli._download_buildings", lambda *a, **k: pytest.fail("provider called offline")
    )
    output, svg, metadata = (
        tmp_path / f"model.{output_format}",
        tmp_path / "model.svg",
        tmp_path / "sources.json",
    )
    result = CliRunner().invoke(
        main,
        [
            "1 rue Paris",
            "--country",
            "FR",
            "--roof-source",
            "roofer",
            "--reconstruction-input",
            str(manifest),
            "--reconstruction-cache",
            str(tmp_path / "cache"),
            "--window-to-wall-ratio",
            "10",
            "--roof-pitch",
            "80",
            "--max-roof-height",
            "1",
            "--output",
            str(output),
            "--output-format",
            output_format,
            "--svg-output",
            str(svg),
            "--svg-width",
            "640",
            "--svg-height",
            "480",
            "--svg-azimuth",
            "15",
            "--svg-elevation",
            "40",
            "--json-output",
            str(metadata),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(output.with_suffix(".reconstruction.json").read_text())
    assert report["outputs_published"]
    assert report["window_count"] == 24  # One window per exterior wall on each of two storeys.
    assert all(value["modeled_space_count"] == 2 for value in report["storey_enrichment"].values())
    assert {
        decision["source_ratio"]
        for decision in report["glazing_assumptions"]
        if "space_id" in decision
    } == {0.2, 0.3}
    assert sum(
        unit["metrics"]["modeled_zone_volume_m3"] for unit in report["units"]
    ) == pytest.approx(4360.0)
    assert report["benchmark_status"] == "not_evaluated_no_independent_references"
    assert "do not modify Roofer geometry" in result.output
    svg_root = ET.parse(svg).getroot()
    assert svg_root.attrib["data-geometry-hash"] == report["scene_geometry_hash"]
    assert svg_root.attrib["width"] == "640"
    assert svg_root.findall(".//{http://www.w3.org/2000/svg}path[@class='window']")
    assert report["scene_geometry_hash"] in output.read_text()
    assert json.loads(metadata.read_text())["reconstruction"]["outputs_published"]


def test_millimetre_quantized_sloped_ngon_is_not_rejected(tmp_path):
    solid = prism()
    solid = replace(
        solid,
        surfaces=tuple(
            replace(
                surface,
                exterior=tuple(
                    (x, y, round(6.0 + 0.33334 * x + 0.21664 * y, 3)) if z == 6.0 else (x, y, z)
                    for x, y, z in surface.exterior
                ),
            )
            for surface in solid.surfaces
        ),
    )
    doc = cityjson([solid])
    doc["vertices"] = [[round(v * 1000) for v in point] for point in doc["vertices"]]
    doc["transform"] = {"scale": [0.001] * 3, "translate": list(FRAME.origin)}
    scene = validate_scene(import_cityjson(write_json(tmp_path / "quantized.json", doc), FRAME))
    report = scene.solids[0].validation
    assert 1e-6 < report["max_face_deviation_m"] < 0.001
    assert report["planarity_tolerance_m"] == 0.001
    assert report["vertex_equivalence_tolerance_m"] == 0.001
    thermal = make_thermal_scene(scene)
    roofs = [
        s
        for space in thermal.spaces
        for s in space.surfaces
        if s.semantic == "RoofSurface" and not s.interface_id
    ]
    assert len(roofs) == 2
    assert all(plane_deviation(surface) < 1e-6 for surface in roofs)
    expected_vertices = {
        p
        for surface in scene.solids[0].surfaces
        if surface.semantic == "RoofSurface"
        for p in surface.exterior
    }
    actual_vertices = {p for surface in roofs for p in surface.exterior}
    assert len(actual_vertices) == len(expected_vertices)
    for actual, expected in zip(sorted(actual_vertices), sorted(expected_vertices)):
        assert actual == pytest.approx(expected, abs=1e-12)
    assert sum(space.volume for space in thermal.spaces) == pytest.approx(scene.solids[0].volume)
    model = build_model_from_geometry(thermal)
    assert save_model_from_geometry(model, tmp_path / "quantized.osm").exists()
    assert save_svg_from_geometry(thermal, tmp_path / "quantized.svg").exists()


def test_vertices_within_one_millimetre_are_equivalent():
    solid = prism()
    roof = solid.surfaces[1]
    roof = replace(roof, exterior=((0.0005, 0.0, 6.0), *roof.exterior[1:]))
    solid = replace(solid, surfaces=(solid.surfaces[0], roof, *solid.surfaces[2:]))
    checked = validate_scene(GeometryScene(FRAME, (solid,))).solids[0]
    assert checked.volume == pytest.approx(600.0)
    assert checked.validation["max_vertex_adjustment_m"] == pytest.approx(0.0005)


def test_planarity_outside_one_millimetre_has_named_rejection():
    solid = prism()
    roof = solid.surfaces[1]
    roof = replace(roof, exterior=((0.0, 0.0, 6.005), *roof.exterior[1:]))
    solid = replace(solid, surfaces=(solid.surfaces[0], roof, *solid.surfaces[2:]))
    with pytest.raises(ReconstructionError) as error:
        validate_scene(GeometryScene(FRAME, (solid,)))
    assert error.value.code == "planarity_tolerance_exceeded"


@pytest.mark.parametrize("orientations", [("south",), ("north", "east"), ()])
def test_requested_glazing_orientations_and_achieved_ratios(tmp_path, orientations):
    import math

    scene = make_thermal_scene(GeometryScene(FRAME, (prism(),)))
    model = build_model_from_geometry(
        scene,
        window_to_wall_ratio=0.3,
        window_orientations={"building": orientations},
    )
    windows = model.getSubSurfaces()
    assert len(windows) == (len(orientations) if orientations else 4)
    if orientations:
        assert all(
            cardinal_point(math.degrees(window.surface().get().azimuth())) in orientations
            for window in windows
        )
    report = json.loads(
        model.getBuilding().additionalProperties().getFeatureAsString("geogen_glazing_report").get()
    )["building"]
    assert report["requested_orientations"] == list(orientations)
    assert report["requested_wwr"] == 0.3
    assert report["eligible_wall_area_m2"] == pytest.approx(len(windows) * 60.0)
    assert report["requested_window_area_m2"] == pytest.approx(len(windows) * 18.0)
    assert report["achieved_window_area_m2"] == pytest.approx(len(windows) * 18.0)
    assert report["achieved_wwr_eligible"] == pytest.approx(0.3)
    assert report["achieved_wwr_exterior"] == pytest.approx(len(windows) * 18.0 / 240.0)
    assert save_model_from_geometry(model, tmp_path / "oriented.osm").exists()
    assert save_svg_from_geometry(
        scene,
        tmp_path / "oriented.svg",
        windows=windows_from_geometry_model(model),
    ).exists()


def test_orientation_hints_are_scoped_to_space_ids():
    scene = make_thermal_scene(
        GeometryScene(
            FRAME,
            (
                prism("a"),
                prism("b", offset=(20.0, 0.0, 0.0)),
            ),
        )
    )
    model = build_model_from_geometry(
        scene, window_to_wall_ratio=0.3, window_orientations={"a": ("south",)}
    )
    counts = {
        space.nameString(): sum(len(surface.subSurfaces()) for surface in space.surfaces())
        for space in model.getSpaces()
    }
    assert counts == {"a": 1, "b": 4}


def test_orientation_hints_never_glaze_reciprocal_party_walls():
    scene = make_thermal_scene(
        GeometryScene(
            FRAME,
            (
                prism("a"),
                prism("b", offset=(10.0, 0.0, 0.0)),
            ),
        )
    )
    model = build_model_from_geometry(
        scene,
        window_to_wall_ratio=0.3,
        window_orientations={"a": ("east",), "b": ("west",)},
    )
    assert len(model.getSubSurfaces()) == 0
    reports = json.loads(
        model.getBuilding().additionalProperties().getFeatureAsString("geogen_glazing_report").get()
    )
    assert all(report["eligible_wall_area_m2"] == 0 for report in reports.values())
    assert all(report["achieved_wwr_eligible"] is None for report in reports.values())
    assert all(
        report["eligibility_status"] == "no_eligible_exterior_walls" for report in reports.values()
    )


@pytest.mark.parametrize(
    "orientations",
    [
        {"unknown": ("north",)},
        {"building": ("N",)},
        {"building": "north"},
    ],
)
def test_invalid_glazing_orientation_hints_rejected(orientations):
    scene = make_thermal_scene(GeometryScene(FRAME, (prism(),)))
    with pytest.raises(ModelError, match="orientations"):
        build_model_from_geometry(scene, window_orientations=orientations)
