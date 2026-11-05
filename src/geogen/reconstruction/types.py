"""Immutable contracts for sources, reconstructed exteriors and thermal scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Point3 = tuple[float, float, float]
Ring3 = tuple[Point3, ...]
SCHEMA_VERSION = "1.0"
GEOGEN_BASE_COMMIT = "86b45c5ea4a25054c4eb024659e62699a22f7246"


class ReconstructionError(ValueError):
    """A failed reconstruction gate, not evidence that a target does not exist."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SourceIdentity:
    bdnb_group_ids: tuple[str, ...] = ()
    bdnb_construction_ids: tuple[str, ...] = ()
    bdtopo_ids: tuple[str, ...] = ()
    rnb_ids: tuple[str, ...] = ()
    dataset_versions: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class SourceLink:
    reconstruction_id: str
    source_type: str
    source_id: str
    method: str
    relationship: str
    intersection_area: float | None = None
    source_coverage: float | None = None
    physical_coverage: float | None = None
    iou: float | None = None
    attribute_scope: str = "unknown"
    confidence: str = "unresolved"
    reasons: tuple[str, ...] = ()
    source_dates: tuple[str, ...] = ()
    attribution_polygon: str | None = None


@dataclass(frozen=True)
class ReconstructionUnit:
    reconstruction_id: str
    identity: SourceIdentity
    original_wkt: str
    working_wkt: str
    geometry_revision: str
    outline_type: str = "unknown"
    role: str = "target"
    component_index: int = 0
    crs: str = "EPSG:2154"
    decisions: tuple[str, ...] = ()
    links: tuple[SourceLink, ...] = ()


@dataclass(frozen=True)
class GeometryFrame:
    projected_crs: str
    vertical_reference: str
    origin: Point3

    def to_local(self, point: Point3) -> Point3:
        return tuple(float(v - o) for v, o in zip(point, self.origin))


@dataclass(frozen=True)
class Surface:
    face_id: str
    exterior: Ring3
    interiors: tuple[Ring3, ...] = ()
    semantic: str = "Unknown"
    source_object_id: str = ""
    source_face_id: str = ""
    boundary: str = "unresolved"
    adjacent_space_id: str | None = None
    interface_id: str | None = None
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True)
class Solid:
    solid_id: str
    reconstruction_id: str
    surfaces: tuple[Surface, ...]
    links: tuple[SourceLink, ...] = ()
    interior_shells: tuple[tuple[Surface, ...], ...] = ()
    volume: float | None = None
    validation: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeometryScene:
    frame: GeometryFrame
    solids: tuple[Solid, ...]


@dataclass(frozen=True)
class ThermalSpace:
    space_id: str
    solid_id: str
    surfaces: tuple[Surface, ...]
    volume: float
    conditioned: bool | None = None
    storey_number: int | None = None
    nominal_z: float | None = None


@dataclass(frozen=True)
class ThermalScene:
    frame: GeometryFrame
    spaces: tuple[ThermalSpace, ...]
    geometry_hash: str
    assumptions: tuple[str, ...] = ()
