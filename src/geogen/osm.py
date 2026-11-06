"""Generation of OpenStudio models (.osm) from building footprints."""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Iterable
from pathlib import Path

import openstudio
from openstudio import model as osmodel

from geogen.envelope import CARDINAL_POINTS, DEFAULT_MAX_ROOF_HEIGHT, cardinal_point
from geogen.geometry import (
    Footprint,
    bounding_center,
    exterior_ring,
    floor_patches,
    roof_sections,
    to_wgs84,
)
from geogen.reconstruction.types import ThermalScene
from geogen.thermal_properties import apply_envelope_properties

LOGGER = logging.getLogger(__name__)

DEFAULT_BUILDING_NAME = "BDNB buildings"
_UNSAFE_FILE_NAME_RE = re.compile(r"[^\w.-]+")


class ModelError(RuntimeError):
    """Raised when the OpenStudio model cannot be built or saved."""


def model_name(codes: Iterable[str]) -> str:
    """Name of the model of the buildings identified by their BDNB ``codes``."""
    unique = list(dict.fromkeys(code for code in codes if code))
    if not unique:
        return DEFAULT_BUILDING_NAME
    if len(unique) == 1:
        return unique[0]
    return f"{unique[0]} and {len(unique) - 1} more"


def model_file_name(name: str, suffix: str = ".osm") -> str:
    """File name of the model of a building, derived from its ``name``."""
    stem = _UNSAFE_FILE_NAME_RE.sub("_", name).strip("._")
    return f"{stem or 'model'}{suffix}"


def model_file_suffix(output_format: str) -> str:
    """File suffix used for models written in ``output_format``."""
    if output_format not in OUTPUT_FORMATS:
        raise ModelError(
            f"Unknown output format {output_format!r}, expected one of {sorted(OUTPUT_FORMATS)}"
        )
    return OUTPUT_FORMATS[output_format]


def build_model(
    footprints: Iterable[Footprint],
    *,
    building_name: str = DEFAULT_BUILDING_NAME,
    max_roof_height: float = DEFAULT_MAX_ROOF_HEIGHT,
) -> osmodel.Model:
    """Build an OpenStudio model made of one space per storey of each footprint.

    The model coordinates are metres relative to the centre of the bounding box of the
    footprints, which is used as the site location of the model.
    Buildings whose envelope is known are completed with a sloped roof and with windows on
    their exterior walls.
    """
    footprints = list(footprints)
    if not footprints:
        raise ModelError("At least one footprint is required to build a model")
    coordinate_systems = {footprint.crs for footprint in footprints}
    if len(coordinate_systems) != 1:
        raise ModelError("Footprints must be transformed to one common CRS")
    crs = footprints[0].crs
    origin = bounding_center(footprints)

    model = osmodel.Model()
    model.getBuilding().setName(building_name)

    elevations = [
        footprint.ground_elevation
        for footprint in footprints
        if footprint.ground_elevation is not None
    ]
    reference_elevation = min(elevations) if elevations else 0.0
    longitude, latitude = to_wgs84(origin[0], origin[1], crs)
    site = model.getSite()
    site.setName(f"{building_name} site")
    site.setLatitude(latitude)
    site.setLongitude(longitude)
    site.setElevation(reference_elevation)

    spaces = openstudio.model.SpaceVector()
    extruded: list[tuple[Footprint, list[osmodel.Space]]] = []
    envelope_by_space = {}
    for footprint in footprints:
        base_elevation = _base_elevation(footprint, reference_elevation)
        try:
            storeys = _add_storeys(model, footprint, origin, base_elevation)
        except ModelError as error:
            LOGGER.warning("Skipping footprint %s: %s", footprint.name, error)
            continue
        extruded.append((footprint, storeys))
        for space in storeys:
            spaces.append(space)
            envelope_by_space[space.nameString()] = footprint.envelope.thermal_properties
        for attic in _add_roof(model, footprint, origin, base_elevation, max_roof_height):
            spaces.append(attic)
            envelope_by_space[attic.nameString()] = footprint.envelope.thermal_properties
    if not spaces:
        raise ModelError("None of the footprints could be extruded into a space")

    # ForwardTranslator omits spaces without a zone, including all their geometry.
    for space in spaces:
        zone = osmodel.ThermalZone(model)
        zone.setName(f"{space.nameString()} Zone")
        space.setThermalZone(zone)
    LOGGER.info("Built %d spaces from %d footprints", len(spaces), len(footprints))
    osmodel.intersectSurfaces(spaces)
    osmodel.matchSurfaces(spaces)

    windows = sum(_add_windows(footprint, storeys) for footprint, storeys in extruded)
    if windows:
        LOGGER.info("Added %d windows", windows)
    apply_envelope_properties(model, envelope_by_space)
    return model


def _base_elevation(footprint: Footprint, reference_elevation: float) -> float:
    """Height of the ground of a building above the reference elevation.

    Buildings whose ground altitude is unknown are placed at the reference elevation of the
    model.
    """
    if footprint.ground_elevation is None:
        return 0.0
    return footprint.ground_elevation - reference_elevation


def _floor_print(footprint: Footprint, origin: tuple[float, float]) -> openstudio.Point3dVector:
    """Return the ring of a footprint relative to the model origin."""
    origin_x, origin_y = origin
    points = openstudio.Point3dVector()
    for x, y in footprint.ring:
        points.append(openstudio.Point3d(x - origin_x, y - origin_y, 0.0))
    return points


def _add_storeys(
    model: osmodel.Model,
    footprint: Footprint,
    origin: tuple[float, float],
    base_elevation: float,
) -> list[osmodel.Space]:
    """Extrude a footprint into one space per storey."""
    floor_print = _floor_print(footprint, origin)
    storey_height = footprint.storey_height
    spaces: list[osmodel.Space] = []
    for level in range(footprint.storeys):
        optional_space = (
            None
            if footprint.holes
            else osmodel.Space.fromFloorPrint(floor_print, storey_height, model)
        )
        if optional_space is not None and not optional_space.is_initialized():
            _remove_spaces(spaces)
            raise ModelError(
                f"OpenStudio could not create a space for footprint {footprint.name!r}"
            )
        elevation = base_elevation + level * storey_height
        space = (
            _courtyard_space(model, footprint, origin, storey_height)
            if footprint.holes
            else optional_space.get()
        )
        space.setName(f"{footprint.name} Storey {level + 1} Space")
        space.setZOrigin(elevation)

        storey = osmodel.BuildingStory(model)
        storey.setName(f"{footprint.name} Storey {level + 1}")
        storey.setNominalZCoordinate(elevation)
        storey.setNominalFloortoFloorHeight(storey_height)
        space.setBuildingStory(storey)
        spaces.append(space)
    return spaces


def _add_roof(
    model: osmodel.Model,
    footprint: Footprint,
    origin: tuple[float, float],
    base_elevation: float,
    max_roof_height: float,
) -> list[osmodel.Space]:
    """Create one enclosed attic per inferred roof wing.

    Independent roofs meeting at zero-height eaves cannot be put into one manifold volume.
    Separate spaces keep every edge paired exactly twice.
    """
    sections = roof_sections(footprint, max_roof_height)
    attics = []
    for index, (patch, faces) in enumerate(sections, 1):
        suffix = f" {index}" if len(sections) > 1 else ""
        space = osmodel.Space(model)
        space.setName(f"{footprint.name} Attic{suffix} Space")
        space.setZOrigin(base_elevation + footprint.height)
        surface = _surface(model, space, [(x, y, 0) for x, y in exterior_ring(patch, 0)], origin)
        surface.setSurfaceType("Floor")
        for face in faces:
            surface = _surface(model, space, face, origin)
            surface.setSurfaceType("RoofCeiling")
            surface.setOutsideBoundaryCondition("Outdoors")
        roof_height = max(z for face in faces for x, y, z in face)
        storey = osmodel.BuildingStory(model)
        storey.setName(f"{footprint.name} Attic{suffix}")
        storey.setNominalZCoordinate(base_elevation + footprint.height)
        storey.setNominalFloortoFloorHeight(roof_height)
        space.setBuildingStory(storey)
        attics.append(space)
    return attics


def _add_windows(footprint: Footprint, spaces: Iterable[osmodel.Space]) -> int:
    """Glaze the exterior walls of the spaces of a building.

    Walls are left untouched when the BDNB does not know the share of glazing of the
    building, and only the glazed orientations are fenestrated when the BDNB knows them.
    """
    ratio = footprint.envelope.window_to_wall_ratio
    if not ratio:
        return 0
    orientations = footprint.envelope.glazing_orientations
    windows = 0
    for space in spaces:
        for surface in space.surfaces():
            if surface.surfaceType() != "Wall":
                continue
            if surface.outsideBoundaryCondition() != "Outdoors":
                continue
            if orientations and cardinal_point(math.degrees(surface.azimuth())) not in orientations:
                continue
            if surface.setWindowToWallRatio(ratio).is_initialized():
                windows += 1
            else:
                LOGGER.debug("No window fits on surface %s", surface.nameString())
    return windows


def _remove_spaces(spaces: list[osmodel.Space]) -> None:
    """Remove partially built spaces and their storeys from their model."""
    for space in spaces:
        storey = space.buildingStory()
        space.remove()
        if storey.is_initialized():
            storey.get().remove()


#: Output formats supported by :func:`save_model`, mapped to their file suffix.
OUTPUT_FORMATS = {"osm": ".osm", "idf": ".idf"}
DEFAULT_OUTPUT_FORMAT = "osm"


def save_model(
    model: osmodel.Model, path: str | Path, output_format: str = DEFAULT_OUTPUT_FORMAT
) -> Path:
    """Save an OpenStudio model to ``path``, overwriting any existing file.

    ``output_format`` selects how the model is written: ``"osm"`` (the default) saves the
    native OpenStudio model, while ``"idf"`` forward translates it to an EnergyPlus IDF file.
    """
    if output_format not in OUTPUT_FORMATS:
        raise ModelError(
            f"Unknown output format {output_format!r}, expected one of {sorted(OUTPUT_FORMATS)}"
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "idf":
        translator = openstudio.energyplus.ForwardTranslator()
        workspace = translator.translateModel(model)
        for error in translator.errors():
            LOGGER.error("IDF translation error: %s", error.logMessage())
        if not workspace.save(openstudio.path(str(destination)), True):
            raise ModelError(f"Could not write the IDF model to {destination}")
    elif not model.save(openstudio.path(str(destination)), True):
        raise ModelError(f"Could not write the OpenStudio model to {destination}")
    return destination


def _surface(model, space, vertices, origin):
    points = openstudio.Point3dVector(
        [openstudio.Point3d(x - origin[0], y - origin[1], z) for x, y, z in vertices]
    )
    surface = osmodel.Surface(points, model)
    surface.setSpace(space)
    return surface


def _courtyard_space(model, footprint, origin, height):
    """One closed space with tiled floors/ceilings and outward courtyard walls."""
    space = osmodel.Space(model)
    for patch in floor_patches(footprint):
        ring = exterior_ring(patch, 0)
        floor = _surface(model, space, [(x, y, 0) for x, y in ring], origin)
        floor.setSurfaceType("Floor")
        floor.setOutsideBoundaryCondition("Ground")
        ceiling = _surface(model, space, [(x, y, height) for x, y in reversed(ring)], origin)
        ceiling.setSurfaceType("RoofCeiling")
        ceiling.setOutsideBoundaryCondition("Outdoors")
    for ring in (footprint.ring, *footprint.holes):
        for start, end in zip(ring, (*ring[1:], ring[0])):
            wall = _surface(
                model, space, [(*end, 0), (*start, 0), (*start, height), (*end, height)], origin
            )
            wall.setSurfaceType("Wall")
            wall.setOutsideBoundaryCondition("Outdoors")
    return space


def build_model_from_geometry(
    scene: ThermalScene,
    *,
    building_name: str = DEFAULT_BUILDING_NAME,
    window_to_wall_ratio: float | None = None,
    window_ratios: dict[str, float] | None = None,
    window_orientations: dict[str, tuple[str, ...]] | None = None,
    envelope_properties: dict[str, dict] | None = None,
) -> osmodel.Model:
    """Export the actual thermal surfaces; never extrude or infer a roof.

    Reciprocal interfaces need a construction for OpenStudio translation. Their
    explicitly labelled R=0.1 placeholder is not a measured building property;
    this geometry-only model requires material enrichment before simulation.
    ``window_ratios`` overrides the global ratio by original thermal space ID.
    ``window_orientations`` optionally restricts each space to canonical cardinal
    directions. Empty or absent hints leave all its exterior walls eligible.
    Unrepresentable windows are omitted and reported, never by changing walls.
    ``envelope_properties`` supplies scoped classifications and optional numeric
    thermal values by exact thermal space ID. Enrichment never infers U-values
    from labels and remains partial; see :func:`apply_envelope_properties`.
    """
    from dataclasses import replace

    from geogen.reconstruction.thermal import validate_thermal_scene
    from geogen.reconstruction.validate import face_plane, node_surfaces, surface_pieces

    validate_thermal_scene(scene)
    window_ratios = window_ratios or {}
    window_orientations = window_orientations or {}
    if set(window_ratios) - {space.space_id for space in scene.spaces}:
        raise ModelError("Per-space window ratios reference unknown thermal spaces")
    if set(window_orientations) - {space.space_id for space in scene.spaces}:
        raise ModelError("Window orientations reference unknown thermal spaces")
    if envelope_properties is not None and (
        not isinstance(envelope_properties, dict)
        or set(envelope_properties) - {space.space_id for space in scene.spaces}
    ):
        raise ModelError("Envelope properties reference unknown thermal spaces")
    for orientations in window_orientations.values():
        if not isinstance(orientations, (tuple, list)) or any(
            orientation not in CARDINAL_POINTS for orientation in orientations
        ):
            raise ModelError("Window orientations must use north, east, south or west")
    for ratio in (window_to_wall_ratio, *window_ratios.values()):
        if ratio is not None and (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(ratio)
            or not 0 <= ratio < 1
        ):
            raise ModelError("Window-to-wall ratio must be finite and in [0, 1)")
    model = osmodel.Model()
    building = model.getBuilding()
    building.setName(building_name)
    building.additionalProperties().setFeature("geogen_geometry_hash", scene.geometry_hash)
    building.additionalProperties().setFeature("geogen_fidelity_status", "experimental_unverified")
    building.additionalProperties().setFeature(
        "geogen_construction_assumption",
        "Partial source-scoped envelope enrichment, not simulation-ready. Missing thermal "
        "properties remain unset. Reciprocal interfaces use an unspecified massless "
        "R=0.1 m2-K/W placeholder; replace it with sourced constructions.",
    )
    interface_construction = None
    if any(s.interface_id for space in scene.spaces for s in space.surfaces):
        material = osmodel.MasslessOpaqueMaterial(model)
        material.setName("GEOMETRY ONLY - unspecified interface material R=0.1")
        material.setThermalResistance(0.1)
        interface_construction = osmodel.Construction(model)
        interface_construction.setName("GEOMETRY ONLY - replace unspecified interface construction")
        if not interface_construction.insertLayer(0, material):
            raise ModelError("Could not create the explicit geometry-only interface placeholder")
    longitude, latitude = to_wgs84(
        scene.frame.origin[0], scene.frame.origin[1], scene.frame.projected_crs
    )
    model.getSite().setLatitude(latitude)
    model.getSite().setLongitude(longitude)
    model.getSite().setElevation(scene.frame.origin[2])
    interfaces, exported = {}, {}
    manifest = {"geometry_hash": scene.geometry_hash, "spaces": {}, "surfaces": {}}
    ratios_by_model_space = {}
    orientations_by_model_space = {}
    glazing_reports = {}
    stories = {}
    for thermal_space in scene.spaces:
        space = osmodel.Space(model)
        space.setName(thermal_space.space_id)
        zone = osmodel.ThermalZone(model)
        zone.setName(f"{thermal_space.space_id} Zone")
        space.setThermalZone(zone)
        _assign_geometry_story(model, space, thermal_space, stories)
        manifest["spaces"][space.nameString()] = thermal_space.volume
        ratios_by_model_space[space.nameString()] = window_ratios.get(
            thermal_space.space_id, window_to_wall_ratio
        )
        orientations_by_model_space[space.nameString()] = tuple(
            dict.fromkeys(window_orientations.get(thermal_space.space_id, ()))
        )
        glazing_reports[space.nameString()] = {
            "space_id": thermal_space.space_id,
            "requested_wwr": ratios_by_model_space[space.nameString()],
            "requested_orientations": orientations_by_model_space[space.nameString()],
            "exterior_wall_area_m2": 0.0,
            "eligible_wall_area_m2": 0.0,
            "achieved_window_area_m2": 0.0,
            "window_count": 0,
            "omitted_windows": [],
        }
        pieces = []
        parent_faces = {}
        for surface in thermal_space.surfaces:
            if surface.interface_id in interfaces:
                reference = interfaces[surface.interface_id]
                tiles = tuple(
                    replace(
                        surface,
                        exterior=tile.exterior[::-1],
                        interiors=(),
                        face_id=f"{surface.face_id}:paired:{i}",
                    )
                    for i, tile in enumerate(reference)
                )
            else:
                tiles = surface_pieces(surface)
                if surface.interface_id:
                    interfaces[surface.interface_id] = tiles
            if not math.isclose(
                sum(face_plane(tile)[4].area for tile in tiles),
                face_plane(surface)[4].area,
                rel_tol=1e-9,
                abs_tol=1e-7,
            ):
                raise ModelError("Hole decomposition changed a surface area")
            pieces.extend(tiles)
            parent_faces.update({tile.face_id: surface.face_id for tile in tiles})
        pieces = node_surfaces(pieces)
        for piece in pieces:
            points = openstudio.Point3dVector(
                [openstudio.Point3d(*point) for point in piece.exterior]
            )
            surface = osmodel.Surface(points, model)
            surface.setName(piece.face_id)
            if not surface.setSpace(space):
                raise ModelError(f"Could not assign {piece.face_id} to its space")
            normal = face_plane(piece)[3]
            surface_type = (
                "Wall"
                if piece.semantic == "WallSurface"
                else ("RoofCeiling" if normal[2] > 0 else "Floor")
            )
            if not surface.setSurfaceType(surface_type):
                raise ModelError(f"Could not set surface type for {piece.face_id}")
            boundary = {
                "ground": "Ground",
                "outdoors": "Outdoors",
                "adiabatic": "Adiabatic",
                "surface": "Adiabatic",
            }[piece.boundary]
            if not surface.setOutsideBoundaryCondition(boundary):
                raise ModelError(f"Could not set boundary for {piece.face_id}")
            if piece.boundary != "outdoors":
                surface.setSunExposure("NoSun")
                surface.setWindExposure("NoWind")
            if piece.interface_id and not surface.setConstruction(interface_construction):
                raise ModelError("Could not assign the geometry-only interface placeholder")
            properties = surface.additionalProperties()
            properties.setFeature("geogen_source_face_id", piece.source_face_id)
            properties.setFeature("geogen_source_object_id", piece.source_object_id)
            properties.setFeature("geogen_interface_id", piece.interface_id or "")
            properties.setFeature("geogen_thermal_face_id", parent_faces[piece.face_id])
            properties.setFeature("geogen_provenance", json.dumps(piece.provenance))
            exported[piece.face_id] = (surface, piece, space.nameString())
    pairs = {}
    for surface, piece, space_name in exported.values():
        if piece.interface_id:
            vertex_key = tuple(sorted(tuple(round(v, 8) for v in p) for p in piece.exterior))
            pairs.setdefault((piece.interface_id, vertex_key), []).append(surface)
        ratio = ratios_by_model_space[space_name]
        orientations = orientations_by_model_space[space_name]
        exterior_wall = piece.boundary == "outdoors" and surface.surfaceType() == "Wall"
        eligible = exterior_wall and (
            not orientations or cardinal_point(math.degrees(surface.azimuth())) in orientations
        )
        report = glazing_reports[space_name]
        if exterior_wall:
            report["exterior_wall_area_m2"] += surface.grossArea()
        if eligible:
            report["eligible_wall_area_m2"] += surface.grossArea()
        if ratio and eligible:
            result = surface.setWindowToWallRatio(ratio)
            if not result.is_initialized() or not math.isclose(
                sum(window.grossArea() for window in surface.subSurfaces()),
                surface.grossArea() * ratio,
                rel_tol=1e-5,
                abs_tol=1e-6,
            ):
                for window in list(surface.subSurfaces()):
                    window.remove()
                report["omitted_windows"].append(
                    {
                        "face_id": piece.face_id,
                        "requested_wwr": ratio,
                        "requested_area_m2": surface.grossArea() * ratio,
                        "reason": "requested_windows_not_representable",
                    }
                )
        report["achieved_window_area_m2"] += sum(w.grossArea() for w in surface.subSurfaces())
        report["window_count"] += len(surface.subSurfaces())
    for pair in pairs.values():
        if len(pair) != 2 or not pair[0].setAdjacentSurface(pair[1]):
            raise ModelError("Could not create a reciprocal OpenStudio boundary")
    for surface, piece, space_name in exported.values():
        manifest["surfaces"][surface.nameString()] = {
            "vertices": piece.exterior,
            "area": face_plane(piece)[4].area,
            "space": space_name,
            "boundary": surface.outsideBoundaryCondition(),
            "adjacent": (
                surface.adjacentSurface().get().nameString()
                if surface.adjacentSurface().is_initialized()
                else None
            ),
            "window_area": sum(s.grossArea() for s in surface.subSurfaces()),
        }
    building.additionalProperties().setFeature(
        "geogen_geometry_manifest", json.dumps(manifest, sort_keys=True)
    )
    for space in model.getSpaces():
        report = glazing_reports[space.nameString()]
        eligible_area = report["eligible_wall_area_m2"]
        exterior_area = report["exterior_wall_area_m2"]
        report["requested_window_area_m2"] = (
            report["requested_wwr"] * eligible_area if report["requested_wwr"] is not None else None
        )
        report["achieved_wwr_eligible"] = (
            report["achieved_window_area_m2"] / eligible_area if eligible_area else None
        )
        report["achieved_wwr_exterior"] = (
            report["achieved_window_area_m2"] / exterior_area if exterior_area else None
        )
        report["eligibility_status"] = (
            "eligible_walls_present" if eligible_area else "no_eligible_exterior_walls"
        )
        space.additionalProperties().setFeature(
            "geogen_glazing_report", json.dumps(report, sort_keys=True)
        )
    building.additionalProperties().setFeature(
        "geogen_glazing_report",
        json.dumps(
            {report["space_id"]: report for report in glazing_reports.values()}, sort_keys=True
        ),
    )
    apply_envelope_properties(model, envelope_properties)
    _check_geometry_model(model, manifest)
    return model


def _assign_geometry_story(model, space, thermal_space, stories):
    """Attach storey metadata in the shared local frame, without moving geometry."""
    number = getattr(thermal_space, "storey_number", None)
    nominal_z = getattr(thermal_space, "nominal_z", None)
    if number is None and nominal_z is None:
        return
    if number is not None and (isinstance(number, bool) or not isinstance(number, int)):
        raise ModelError("Thermal storey number must be an integer")
    if nominal_z is None:
        nominal_z = min(p[2] for face in thermal_space.surfaces for p in face.exterior)
    if (
        isinstance(nominal_z, bool)
        or not isinstance(nominal_z, (int, float))
        or not math.isfinite(nominal_z)
    ):
        raise ModelError("Thermal storey elevation must be finite")
    key = (thermal_space.solid_id, number if number is not None else thermal_space.space_id)
    if key not in stories:
        story = osmodel.BuildingStory(model)
        story.setName(
            f"{thermal_space.solid_id} Storey {number}"
            if number is not None
            else f"{thermal_space.space_id} Storey"
        )
        if not story.setNominalZCoordinate(nominal_z):
            raise ModelError("Could not set the local thermal storey elevation")
        story.additionalProperties().setFeature("geogen_source_solid_id", thermal_space.solid_id)
        if number is not None:
            story.additionalProperties().setFeature("geogen_storey_number", number)
        stories[key] = story
    story = stories[key]
    if not math.isclose(story.nominalZCoordinate().get(), nominal_z, abs_tol=1e-8):
        raise ModelError("Thermal spaces in one solid/storey disagree on their elevation")
    if not space.setBuildingStory(story):
        raise ModelError("Could not associate a thermal space with its storey")


def windows_from_geometry_model(model: osmodel.Model) -> dict[str, tuple]:
    """Extract actual native glazing in the shared frame, keyed by thermal face ID.

    Pass the returned mapping to ``save_svg_from_geometry(..., windows=...)``.
    It includes no proxy windows and preserves local-frame coordinates after
    applying each OpenStudio space transformation.
    """
    properties = model.getBuilding().additionalProperties()
    manifest = properties.getFeatureAsString("geogen_geometry_manifest")
    if not manifest.is_initialized():
        raise ModelError("Window extraction requires a solid-based model")
    _check_geometry_model(model, json.loads(manifest.get()))
    windows = {}
    for surface in model.getSurfaces():
        if not surface.subSurfaces():
            continue
        face = surface.additionalProperties().getFeatureAsString("geogen_thermal_face_id")
        if not face.is_initialized():
            raise ModelError("Glazed surface has no thermal face provenance")
        transformation = surface.space().get().transformation()
        for window in surface.subSurfaces():
            ring = tuple((p.x(), p.y(), p.z()) for p in transformation * window.vertices())
            windows.setdefault(face.get(), []).append(ring)
    return {face: tuple(rings) for face, rings in windows.items()}


def _check_geometry_model(model, manifest):
    """Check geometry after native construction/reload, including tiled holes."""
    from shapely.geometry import Polygon

    from geogen.reconstruction.types import Surface
    from geogen.reconstruction.validate import face_plane, project

    actual_spaces = {space.nameString(): space for space in model.getSpaces()}
    if set(actual_spaces) != set(manifest["spaces"]):
        raise ModelError("OpenStudio changed the thermal space inventory")
    for name, volume in manifest["spaces"].items():
        if not math.isclose(actual_spaces[name].volume(), volume, rel_tol=1e-7, abs_tol=1e-6):
            raise ModelError(f"OpenStudio changed the enclosed volume of {name}")
    actual = {surface.nameString(): surface for surface in model.getSurfaces()}
    if set(actual) != set(manifest["surfaces"]):
        raise ModelError("OpenStudio changed the thermal surface inventory")
    for name, expected in manifest["surfaces"].items():
        surface = actual[name]
        if not surface.space().is_initialized():
            raise ModelError(f"OpenStudio removed space ownership: {name}")
        plane = face_plane(Surface(name, tuple(tuple(p) for p in expected["vertices"])))
        world_vertices = surface.space().get().transformation() * surface.vertices()
        vertices = [(p.x(), p.y(), p.z()) for p in world_vertices]
        actual_plane = face_plane(Surface(name, tuple(vertices)))
        polygon = Polygon([project(point, plane) for point in vertices])
        if (
            polygon.symmetric_difference(plane[4]).area > 1e-6
            or sum(a * b for a, b in zip(plane[3], actual_plane[3])) < 1 - 1e-8
            or any(
                abs(sum((p[i] - plane[0][i]) * plane[3][i] for i in range(3))) > 1e-6
                for p in vertices
            )
            or not math.isclose(surface.grossArea(), expected["area"], rel_tol=1e-8, abs_tol=1e-6)
        ):
            raise ModelError(f"OpenStudio changed face geometry/winding: {name}")
        if (
            not surface.space().is_initialized()
            or surface.space().get().nameString() != expected["space"]
        ):
            raise ModelError(f"OpenStudio changed space ownership: {name}")
        if surface.outsideBoundaryCondition() != expected["boundary"]:
            raise ModelError(f"OpenStudio changed boundary condition: {name}")
        adjacent = surface.adjacentSurface()
        adjacent_name = adjacent.get().nameString() if adjacent.is_initialized() else None
        if adjacent_name != expected["adjacent"]:
            raise ModelError(f"OpenStudio changed reciprocal adjacency: {name}")
        if not math.isclose(
            sum(s.grossArea() for s in surface.subSurfaces()),
            expected["window_area"],
            rel_tol=1e-8,
            abs_tol=1e-6,
        ):
            raise ModelError(f"OpenStudio changed window area: {name}")


def _check_translated_geometry(workspace, manifest):
    """Check IDF coordinates, holes-as-tiles and reciprocal boundary references."""
    from shapely.geometry import Polygon

    from geogen.reconstruction.types import Surface
    from geogen.reconstruction.validate import face_plane, project

    objects = workspace.getObjectsByType(openstudio.IddObjectType("BuildingSurface:Detailed"))
    actual = {obj.getString(0).get(): obj for obj in objects}
    expected = {name.casefold(): values for name, values in manifest["surfaces"].items()}
    if {name.casefold() for name in actual} != set(expected):
        raise ModelError("EnergyPlus translation changed the surface inventory")
    for name, obj in actual.items():
        values = expected[name.casefold()]
        vertices = []
        for group in obj.extensibleGroups():
            coords = [group.getDouble(i) for i in range(3)]
            if any(not coord.is_initialized() for coord in coords):
                raise ModelError(f"IDF contains invalid vertices on {name}")
            vertices.append(tuple(coord.get() for coord in coords))
        plane = face_plane(Surface(name, tuple(tuple(p) for p in values["vertices"])))
        actual_plane = face_plane(Surface(name, tuple(vertices)))
        polygon = Polygon([project(point, plane) for point in vertices])
        if (
            polygon.symmetric_difference(plane[4]).area > 1e-6
            or sum(a * b for a, b in zip(plane[3], actual_plane[3])) < 1 - 1e-8
            or any(
                abs(sum((p[i] - plane[0][i]) * plane[3][i] for i in range(3))) > 1e-6
                for p in vertices
            )
        ):
            raise ModelError(f"EnergyPlus translation changed geometry/winding: {name}")
        boundary = obj.getString(5)
        adjacent = obj.getString(6)
        if (
            not boundary.is_initialized()
            or boundary.get().casefold() != values["boundary"].casefold()
        ):
            raise ModelError(f"EnergyPlus translation changed boundary condition: {name}")
        if values["adjacent"] and (
            not adjacent.is_initialized()
            or adjacent.get().casefold() != values["adjacent"].casefold()
        ):
            raise ModelError(f"EnergyPlus translation changed adjacency: {name}")
    window_areas = {name: 0.0 for name in expected}
    for window in workspace.getObjectsByType(
        openstudio.IddObjectType("FenestrationSurface:Detailed")
    ):
        parent = window.getString(3)
        count = window.getInt(8)
        vertex_count = count.get() if count.is_initialized() else (window.numFields() - 9) // 3
        if (
            not parent.is_initialized()
            or parent.get().casefold() not in expected
            or not 3 <= vertex_count <= 4
        ):
            raise ModelError("EnergyPlus translation produced unsupported glazing")
        coordinates = [window.getDouble(9 + i) for i in range(3 * vertex_count)]
        if any(not coord.is_initialized() for coord in coordinates):
            raise ModelError("EnergyPlus translation lost window vertices")
        vertices = tuple(
            tuple(coord.get() for coord in coordinates[i : i + 3])
            for i in range(0, len(coordinates), 3)
        )
        window_areas[parent.get().casefold()] += face_plane(Surface("window", vertices))[4].area
    if any(
        not math.isclose(window_areas[name], values["window_area"], rel_tol=1e-8, abs_tol=1e-6)
        for name, values in expected.items()
    ):
        raise ModelError("EnergyPlus translation changed window area")


def save_model_from_geometry(
    model: osmodel.Model, path: str | Path, output_format: str = DEFAULT_OUTPUT_FORMAT
) -> Path:
    """Fail closed on translation errors or geometry loss, for either format."""
    if output_format not in OUTPUT_FORMATS:
        raise ModelError(f"Unknown output format {output_format!r}")
    properties = model.getBuilding().additionalProperties()
    manifest_value = properties.getFeatureAsString("geogen_geometry_manifest")
    hash_value = properties.getFeatureAsString("geogen_geometry_hash")
    if not manifest_value.is_initialized() or not hash_value.is_initialized():
        raise ModelError("Strict geometry export requires a solid-based model")
    manifest = json.loads(manifest_value.get())
    if manifest.get("geometry_hash") != hash_value.get():
        raise ModelError("The model geometry hash no longer matches its manifest")
    _check_geometry_model(model, manifest)
    translator = openstudio.energyplus.ForwardTranslator()
    workspace = translator.translateModel(model)
    errors = list(translator.errors())
    if errors:
        raise ModelError(
            "EnergyPlus translation failed: " + "; ".join(e.logMessage() for e in errors)
        )
    expected_count = len(manifest["surfaces"])
    surface_type = openstudio.IddObjectType("BuildingSurface:Detailed")
    if len(workspace.getObjectsByType(surface_type)) != expected_count:
        raise ModelError("EnergyPlus translation lost thermal surfaces")
    _check_translated_geometry(workspace, manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "idf":
        if not workspace.save(openstudio.path(str(destination)), True):
            raise ModelError(f"Could not write IDF to {destination}")
        destination.write_text(
            f"!- geogen_geometry_hash: {hash_value.get()}\n"
            "!- Experimental geometry only: fidelity unverified; material enrichment required.\n"
            + destination.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        reloaded = openstudio.IdfFile.load(openstudio.path(str(destination)))
        if (
            not reloaded.is_initialized()
            or len(reloaded.get().getObjectsByType(surface_type)) != expected_count
        ):
            destination.unlink(missing_ok=True)
            raise ModelError("IDF reload lost thermal surfaces")
        try:
            _check_translated_geometry(reloaded.get(), manifest)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
    else:
        if not model.save(openstudio.path(str(destination)), True):
            raise ModelError(f"Could not write OSM to {destination}")
        reloaded = openstudio.osversion.VersionTranslator().loadModel(
            openstudio.path(str(destination))
        )
        try:
            if not reloaded.is_initialized():
                raise ModelError("Saved OSM could not be reloaded")
            _check_geometry_model(reloaded.get(), manifest)
            reloaded_hash = (
                reloaded.get()
                .getBuilding()
                .additionalProperties()
                .getFeatureAsString("geogen_geometry_hash")
            )
            if not reloaded_hash.is_initialized() or reloaded_hash.get() != hash_value.get():
                raise ModelError("OSM reload lost the shared geometry hash")
        except Exception:
            destination.unlink(missing_ok=True)
            raise
    return destination
