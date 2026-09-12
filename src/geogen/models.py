"""Provider-independent building data used by all geometry exporters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class ProviderError(RuntimeError):
    """A provider request or address resolution failed."""


class AddressNotFoundError(ProviderError):
    """No address or building matched the query."""


@dataclass(frozen=True)
class Building:
    """Normalized building; geometry coordinates use the explicitly declared CRS.

    Heights and elevations are metres. Unknown envelope attributes remain None.
    """

    code: str
    country: str = "FR"
    crs: str = "EPSG:2154"
    roof_shape: str | None = None
    geometry: Any = None
    height: float | None = None
    storeys: int | None = None
    ground_elevation: float | None = None
    footprint_area: float | None = None
    fictitious_geometry: bool | None = None
    address: str | None = None
    city: str | None = None
    #: Share of the exterior walls covered by glazing, as published by the provider.
    glazed_ratio: float | None = None
    #: Glazed area (m²) of the representative dwelling, per cardinal direction.
    glazed_areas: tuple[tuple[str, float], ...] = ()
    #: Exterior wall area (m²) of the representative dwelling.
    wall_area: float | None = None
    #: Orientations of the glazed bays, in the source wording.
    glazing_orientations: tuple[str, ...] = ()
    #: Main material covering the roof.
    roof_material: str | None = None
    #: Construction principle of the upper floor, which hints at the roof shape.
    roof_type: str | None = None
    #: Exterior walls of the building group, as described by ``wall_dict``.
    walls: tuple[Any, ...] = ()


class BuildingProvider(Protocol):
    def buildings_for_address(self, query: str, *, max_buildings: int = 10) -> list[Building]: ...
    def close(self) -> None: ...
