"""Approximation of the building envelope (fenestration and roof) from the BDNB.

The BDNB does not describe the geometry of the windows nor the shape of the roofs, but it
publishes enough attributes to approximate them: the share of glazing of the exterior walls
and the orientations of the glazed bays for the fenestration, the inclination of the roof
faces (``wall_dict``), the roof material and the construction of the upper floor for the
roof shape.

Every helper of this module returns ``None`` when the BDNB does not know, so that the
corresponding part of the envelope is left as is, unless an estimate is given.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from geogen.models import Building

LOGGER = logging.getLogger(__name__)

#: Buildings with less glazing than this ratio are modelled without windows.
MIN_WINDOW_TO_WALL_RATIO = 0.01
#: OpenStudio cannot fit a window on a wall above this ratio.
MAX_WINDOW_TO_WALL_RATIO = 0.9
#: Pitch (degrees) used when the BDNB only knows the material of a sloped roof.
DEFAULT_ROOF_PITCH = 30.0
#: Roofs flatter than this pitch (degrees) are kept flat.
MIN_ROOF_PITCH = 5.0
MAX_ROOF_PITCH = 70.0
#: Maximum height (m) of a roof, large footprints would get an unrealistic one.
DEFAULT_MAX_ROOF_HEIGHT = 6.0

#: Cardinal directions of a wall, used to orient the fenestration.
CARDINAL_POINTS = ("north", "east", "south", "west")

_CARDINAL_BY_KEYWORD = {
    "nord": "north",
    "north": "north",
    "est": "east",
    "east": "east",
    "sud": "south",
    "south": "south",
    "ouest": "west",
    "west": "west",
}

#: Roof materials that are only used on sloped roofs.
_PITCHED_ROOF_MATERIALS = (
    "tuile",
    "ardoise",
    "zinc",
    "alumin",
    "chaume",
    "bardeau",
    "shingle",
    "tole",
    "bac acier",
    "fibro",
    "amiante",
    "comble",
    "rampant",
    "charpente",
)
#: Roof materials that are typical of flat roofs.
_FLAT_ROOF_MATERIALS = ("beton", "terrasse", "gravier", "bitume", "etanch", "membrane")

_WORD_SEPARATOR_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Envelope:
    """Envelope attributes of a building, all optional."""

    #: Share of the exterior walls covered by windows, ``None`` when unknown.
    window_to_wall_ratio: float | None = None
    #: Cardinal directions of the glazed facades, empty when all are glazed.
    glazing_orientations: tuple[str, ...] = ()
    #: Pitch of the roof in degrees, ``None`` for a flat or unknown roof.
    roof_pitch: float | None = None


def _normalize(text: Any) -> str:
    """Return ``text`` lowercased and without accents."""
    if not isinstance(text, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char)).lower()


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def cardinal_point(azimuth: float) -> str:
    """Return the cardinal direction of a wall azimuth, in degrees from north."""
    index = int(((azimuth % 360.0) + 45.0) % 360.0 // 90.0)
    return CARDINAL_POINTS[index]


def window_to_wall_ratio(group: Building, default_ratio: float | None = None) -> float | None:
    """Return the share of the exterior walls covered by windows.

    The BDNB publishes the percentage of glazing of the exterior walls, and falls back on
    the glazed and exterior wall areas of the representative dwelling.  ``default_ratio`` is
    the estimate to use for the buildings the BDNB knows nothing about, ``None`` leaves them
    without any window.
    """
    ratio = _as_ratio(group.glazed_ratio)
    if ratio is None:
        glazed = sum(area for _, area in group.glazed_areas)
        wall = group.wall_area
        if glazed > 0 and wall is not None and wall > 0:
            ratio = glazed / wall
    if ratio is None:
        ratio = _as_ratio(default_ratio)
    if ratio is None or ratio < MIN_WINDOW_TO_WALL_RATIO:
        return None
    return min(ratio, MAX_WINDOW_TO_WALL_RATIO)


def _as_ratio(value: float | None) -> float | None:
    """Read a share, given either as a percentage or as a fraction."""
    if value is None or value <= 0:
        return None
    return value / 100.0 if value > 1.0 else value


def glazing_orientations(group: Building) -> tuple[str, ...]:
    """Return the cardinal directions of the glazed facades.

    An empty tuple means that the orientations are unknown, in which case every exterior
    wall is glazed.
    """
    found: list[str] = []
    for orientation in group.glazing_orientations:
        for token in _WORD_SEPARATOR_RE.split(_normalize(orientation)):
            cardinal = _CARDINAL_BY_KEYWORD.get(token)
            if cardinal is not None and cardinal not in found:
                found.append(cardinal)
    if found:
        return tuple(found)
    return tuple(cardinal for cardinal, area in group.glazed_areas if area > 0)


def roof_pitch_from_walls(walls: Iterable[Any]) -> float | None:
    """Return the mean pitch (degrees) of the roof faces of a ``wall_dict``.

    The inclination of a BDNB wall is 90 for a vertical wall and 180 for a horizontal
    ceiling, a flat roof therefore has a pitch of zero.
    """
    faces: list[tuple[float, float]] = []
    for wall in walls:
        if not isinstance(wall, Mapping):
            continue
        if _normalize(wall.get("wall_type")).strip() != "roof":
            continue
        inclination = _as_float(wall.get("inclination"))
        if inclination is None or not 90.0 <= inclination <= 180.0:
            continue
        area = _as_float(wall.get("area")) or 0.0
        faces.append((180.0 - inclination, max(area, 0.0)))
    if not faces:
        return None
    total_area = sum(area for _, area in faces)
    if total_area > 0:
        pitch = sum(pitch * area for pitch, area in faces) / total_area
    else:
        pitch = sum(pitch for pitch, _ in faces) / len(faces)
    return min(pitch, MAX_ROOF_PITCH)


def roof_pitch_from_material(material: str | None, default_pitch: float) -> float | None:
    """Guess the pitch (degrees) of a roof from the material covering it."""
    normalized = _normalize(material)
    if not normalized:
        return None
    if any(keyword in normalized for keyword in _PITCHED_ROOF_MATERIALS):
        return default_pitch
    if any(keyword in normalized for keyword in _FLAT_ROOF_MATERIALS):
        return 0.0
    return None


def envelope_from_group(
    group: Building,
    *,
    default_roof_pitch: float = DEFAULT_ROOF_PITCH,
    default_window_to_wall_ratio: float | None = None,
) -> Envelope:
    """Approximate the envelope of a BDNB building group."""
    shape = getattr(group, "roof_shape", None)
    pitch = (
        0.0
        if shape == "flat"
        else (default_roof_pitch if shape == "pitched" else roof_pitch_from_walls(group.walls))
    )
    if pitch is None:
        for description in (group.roof_material, group.roof_type):
            pitch = roof_pitch_from_material(description, default_roof_pitch)
            if pitch is not None:
                break
    if pitch is not None and pitch < MIN_ROOF_PITCH:
        pitch = None
    return Envelope(
        window_to_wall_ratio=window_to_wall_ratio(group, default_window_to_wall_ratio),
        glazing_orientations=glazing_orientations(group),
        roof_pitch=pitch,
    )
