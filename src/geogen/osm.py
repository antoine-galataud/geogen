"""Generation of OpenStudio models (.osm) from building footprints."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable
from pathlib import Path

import openstudio
from openstudio import model as osmodel

from geogen.envelope import DEFAULT_MAX_ROOF_HEIGHT, cardinal_point
from geogen.geometry import LAMBERT_93, Footprint, bounding_center, roof_apex, to_wgs84

LOGGER = logging.getLogger(__name__)

DEFAULT_BUILDING_NAME = "BDNB buildings"
#: Roof faces smaller than this area (m²) cannot be modelled.
MIN_ROOF_FACE_AREA = 1e-3
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
    origin: tuple[float, float] | None = None,
    building_name: str = DEFAULT_BUILDING_NAME,
    max_roof_height: float = DEFAULT_MAX_ROOF_HEIGHT,
    crs: str = LAMBERT_93,
) -> osmodel.Model:
    """Build an OpenStudio model made of one space per storey of each footprint.

    The model coordinates are metres relative to ``origin`` (the centre of the bounding box
    of the footprints by default), which is used as the site location of the model.
    Buildings whose envelope is known are completed with a sloped roof and with windows on
    their exterior walls.
    """
    footprints = list(footprints)
    if not footprints:
        raise ModelError("At least one footprint is required to build a model")
    if origin is None:
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
        attic = _add_roof(model, footprint, origin, base_elevation, max_roof_height)
        if attic is not None:
            spaces.append(attic)
    if not spaces:
        raise ModelError("None of the footprints could be extruded into a space")

    LOGGER.info("Built %d spaces from %d footprints", len(spaces), len(footprints))
    osmodel.intersectSurfaces(spaces)
    osmodel.matchSurfaces(spaces)

    windows = sum(_add_windows(footprint, storeys) for footprint, storeys in extruded)
    if windows:
        LOGGER.info("Added %d windows", windows)
    return model


def _base_elevation(footprint: Footprint, reference_elevation: float) -> float:
    """Height of the ground of a building above the reference elevation.

    Buildings whose ground altitude is unknown are placed at the reference elevation of the
    model.
    """
    if footprint.ground_elevation is None:
        return 0.0
    return footprint.ground_elevation - reference_elevation


def _floor_print(
    footprint: Footprint, origin: tuple[float, float], elevation: float = 0.0
) -> openstudio.Point3dVector:
    """Return the ring of a footprint relative to the model origin."""
    origin_x, origin_y = origin
    points = openstudio.Point3dVector()
    for x, y in footprint.ring:
        points.append(openstudio.Point3d(x - origin_x, y - origin_y, elevation))
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
        optional_space = osmodel.Space.fromFloorPrint(floor_print, storey_height, model)
        if not optional_space.is_initialized():
            _remove_spaces(spaces)
            raise ModelError(
                f"OpenStudio could not create a space for footprint {footprint.name!r}"
            )
        elevation = base_elevation + level * storey_height
        space = optional_space.get()
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
) -> osmodel.Space | None:
    """Cover a footprint with a hip roof, as an attic space above its storeys.

    Nothing is added when the BDNB does not describe a sloped roof, or when the roof cannot
    be built, the flat roof of the top storey is then left as is.
    """
    pitch = footprint.envelope.roof_pitch
    if not pitch:
        return None
    apex = roof_apex(footprint.ring, pitch, max_roof_height)
    if apex is None:
        LOGGER.debug("No sloped roof could be built for footprint %s", footprint.name)
        return None
    origin_x, origin_y = origin
    apex_point = openstudio.Point3d(apex[0] - origin_x, apex[1] - origin_y, apex[2])

    space = osmodel.Space(model)
    space.setName(f"{footprint.name} Attic Space")
    space.setZOrigin(base_elevation + footprint.height)
    ring = _floor_print(footprint, origin)
    surfaces: list[osmodel.Surface] = []
    try:
        surfaces.append(osmodel.Surface(ring, model))
        for index, start in enumerate(ring):
            end = ring[(index + 1) % len(ring)]
            face = openstudio.Point3dVector()
            face.append(end)
            face.append(start)
            face.append(apex_point)
            surface = osmodel.Surface(face, model)
            surfaces.append(surface)
            if surface.grossArea() < MIN_ROOF_FACE_AREA:
                raise ModelError(f"degenerate roof face of footprint {footprint.name!r}")
        for surface in surfaces:
            surface.setSpace(space)
    except (ModelError, RuntimeError) as error:
        LOGGER.warning("Skipping the roof of footprint %s: %s", footprint.name, error)
        for surface in surfaces:
            surface.remove()
        space.remove()
        return None

    storey = osmodel.BuildingStory(model)
    storey.setName(f"{footprint.name} Attic")
    storey.setNominalZCoordinate(base_elevation + footprint.height)
    storey.setNominalFloortoFloorHeight(apex[2])
    space.setBuildingStory(storey)
    return space


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
