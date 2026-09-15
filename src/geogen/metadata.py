"""Source-building metadata for BDNB, Ordnance Survey, and shared models."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from geogen.models import Building, BuildingGroup, OsBuilding


def _round_if_number(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(float(value), digits)


def _sum_optional(values: Iterable[float | None], digits: int = 2) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return round(sum(valid), digits) if valid else None


def building_metadata(building: Building) -> dict[str, Any]:
    """Describe source measurements, without substituting geometry defaults."""
    estimated_floor_area = None
    if building.footprint_area is not None and building.storeys is not None:
        estimated_floor_area = building.footprint_area * building.storeys

    data = {
        "building_id": building.code,
        "provider": building.provider,
        "country": building.country,
        "crs": building.crs,
        "address": building.address,
        "city": building.city,
        "footprint_area_m2": _round_if_number(building.footprint_area),
        "number_of_storeys": building.storeys,
        "height_m": _round_if_number(building.height),
        "ground_elevation_m": _round_if_number(building.ground_elevation),
        "estimated_floor_area_m2": _round_if_number(estimated_floor_area),
        "fictitious_geometry": building.fictitious_geometry,
        "glazing_ratio": _round_if_number(building.glazed_ratio),
        "glazing_type": building.glazing_type,
        "wall_insulation": building.wall_insulation,
        "upper_floor_insulation": building.upper_floor_insulation,
        "lower_floor_insulation": building.lower_floor_insulation,
        "roof_material": building.roof_material,
        "roof_type": building.roof_type,
        "roof_shape": building.roof_shape,
    }
    if isinstance(building, BuildingGroup):
        data["bdnb_id"] = building.code
    elif isinstance(building, OsBuilding):
        data["os_id"] = building.os_id
    return data


def _street_address(address: str | None) -> str | None:
    """Remove the postcode and locality from a French address label."""
    if not address:
        return None
    street = re.split(r"\s+\d{5}\b", address, maxsplit=1)[0].strip()
    return street or address.strip()


def building_description(building: Building) -> str:
    """Generate a description using only known source attributes."""
    description = (
        f"{building.storeys}-storey building" if building.storeys is not None else "Building"
    )
    street = _street_address(building.address) if building.country == "FR" else building.address
    if street and building.city:
        description += f" located at {street}, in {building.city}"
    elif street:
        description += f" located at {street}"
    elif building.city:
        description += f" located in {building.city}"

    details: list[str] = []
    if building.footprint_area is not None:
        details.append(f"an approximate footprint of {building.footprint_area:.0f} m²")
    if building.footprint_area is not None and building.storeys is not None:
        floor_area = building.footprint_area * building.storeys
        details.append(f"an estimated floor area of {floor_area:.0f} m²")
    if building.height is not None:
        details.append(f"a height of {building.height:.1f} m")
    if details:
        text = details[0] if len(details) == 1 else ", ".join(details[:-1]) + f", and {details[-1]}"
        description += f", with {text}"
    return description + "."


def building_group_metadata(group: Building) -> dict[str, Any]:
    """Backward-compatible name for :func:`building_metadata`."""
    return building_metadata(group)


def building_group_description(group: Building) -> str:
    """Backward-compatible name for :func:`building_description`."""
    return building_description(group)


def portfolio_metadata(
    groups: Mapping[str, Building] | Iterable[Building],
    *,
    source_addresses: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Summarize source buildings from any provider; unknown totals stay null."""
    group_list = groups.values() if isinstance(groups, Mapping) else groups
    buildings = []
    for building in group_list:
        item = building_metadata(building)
        item["description"] = building_description(building)
        buildings.append(item)
    return {
        "source_addresses": list(source_addresses or []),
        "building_count": len(buildings),
        "total_footprint_area_m2": _sum_optional(
            building["footprint_area_m2"] for building in buildings
        ),
        "total_estimated_floor_area_m2": _sum_optional(
            building["estimated_floor_area_m2"] for building in buildings
        ),
        "buildings": buildings,
    }


def save_metadata_json(data: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination
