"""Provider-independent building data used by all geometry exporters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

NOT_PROVIDED = "not provided"


class ProviderError(RuntimeError):
    """A provider request or address resolution failed."""


class AddressNotFoundError(ProviderError):
    """No address or building matched the query."""


@dataclass(frozen=True, kw_only=True)
class Building:
    """Normalized building; geometry coordinates use the explicitly declared CRS.

    Heights and elevations are metres. Unknown measurements remain None. Envelope
    classifications unavailable from a provider use ``NOT_PROVIDED``; explicit nulls remain None.
    """

    code: str = field(kw_only=False)
    provider: ClassVar[str | None] = None
    country: str = "FR"
    crs: str | None = "EPSG:2154"
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
    #: Provider classification of exterior-wall insulation.
    wall_insulation: str | None = NOT_PROVIDED
    #: Provider classification of upper-floor, attic, or roof insulation.
    upper_floor_insulation: str | None = NOT_PROVIDED
    #: Provider classification of lowest-floor insulation.
    lower_floor_insulation: str | None = NOT_PROVIDED
    #: Provider classification of window glazing (for example single, double, or triple).
    glazing_type: str | None = NOT_PROVIDED
    #: Exterior walls of the building group, as described by ``wall_dict``.
    walls: tuple[Any, ...] = ()


@dataclass(frozen=True)
class Address:
    """A BDNB postal address identified by its BAN interop key."""

    cle_interop_adr: str
    label: str | None = None
    code_commune_insee: str | None = None


@dataclass(frozen=True)
class BuildingGroup(Building):
    """BDNB building group (or standalone construction) with shared attributes.

    An unspecified CRS preserves BDNB's historical coordinate auto-detection. The provider
    adapter resolves it before returning normalized buildings.
    """

    code: str = field(init=False)
    provider: ClassVar[str] = "bdnb"
    crs: str | None = field(default=None, kw_only=True)
    batiment_groupe_id: str
    batiment_construction_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "code", self.batiment_groupe_id or self.batiment_construction_id or ""
        )

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> BuildingGroup:
        """Compatibility factory; BDNB schema parsing stays in the provider."""
        from geogen.bdnb import building_group_from_row

        return building_group_from_row(row, building_class=cls)


@dataclass(frozen=True)
class OsBuilding(Building):
    """Ordnance Survey building retaining its unprefixed source identifier."""

    code: str = field(init=False)
    provider: ClassVar[str] = "ordnance_survey"
    country: str = field(default="UK", kw_only=True)
    crs: str | None = field(default="EPSG:27700", kw_only=True)
    os_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", f"os-{self.os_id}")


class BuildingProvider(Protocol):
    def buildings_for_address(self, query: str, *, max_buildings: int = 10) -> list[Building]: ...
    def close(self) -> None: ...
