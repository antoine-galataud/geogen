from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from geogen.bdnb import BuildingGroup


def _round_if_number(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _sum_optional(values: Iterable[float | None], digits: int = 2) -> float | None:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return round(sum(valid), digits)


def building_group_metadata(group: BuildingGroup) -> dict[str, Any]:
    estimated_floor_area = None
    if group.footprint_area is not None and group.storeys is not None:
        estimated_floor_area = group.footprint_area * group.storeys

    return {
        "bdnb_id": group.code,
        "address": group.address,
        "city": group.city,
        "footprint_area_m2": _round_if_number(group.footprint_area),
        "number_of_storeys": group.storeys,
        "height_m": _round_if_number(group.height),
        "ground_elevation_m": _round_if_number(group.ground_elevation),
        "estimated_floor_area_m2": _round_if_number(estimated_floor_area),
        "fictitious_geometry": group.fictitious_geometry,
        "glazing_ratio": _round_if_number(group.glazed_ratio),
        "roof_material": group.roof_material,
        "roof_type": group.roof_type,
    }


import re


def _street_address(address: str | None) -> str | None:
    """Extract the street part from a BDNB address label.

    Example:
    "122 Rue Amelot 75011 Paris 11e Arrondissement"
    becomes:
    "122 Rue Amelot"
    """
    if not address:
        return None

    # Remove everything starting from a French 5-digit postal code.
    street = re.split(r"\s+\d{5}\b", address, maxsplit=1)[0].strip()

    return street or address.strip()


def building_group_description(group: BuildingGroup) -> str:
    # Building type / number of storeys
    if group.storeys is not None:
        description = f"{group.storeys}-storey building"
    else:
        description = "Building"

    # Location
    street = _street_address(group.address)

    if street and group.city:
        description += f" located at {street}, in {group.city}"
    elif street:
        description += f" located at {street}"
    elif group.city:
        description += f" located in {group.city}"

    # Building characteristics
    details: list[str] = []

    if group.footprint_area is not None:
        details.append(f"an approximate footprint of {group.footprint_area:.0f} m²")

    if group.footprint_area is not None and group.storeys is not None:
        estimated_floor_area = group.footprint_area * group.storeys
        details.append(f"an estimated floor area of {estimated_floor_area:.0f} m²")

    if group.height is not None:
        details.append(f"a height of {group.height:.1f} m")

    if details:
        if len(details) == 1:
            details_text = details[0]
        else:
            details_text = ", ".join(details[:-1]) + f", and {details[-1]}"

        description += f", with {details_text}"

    return description + "."


def portfolio_metadata(
    groups: dict[str, BuildingGroup] | Iterable[BuildingGroup],
    *,
    source_addresses: Iterable[str] | None = None,
) -> dict[str, Any]:
    if isinstance(groups, dict):
        group_list = list(groups.values())
    else:
        group_list = list(groups)

    buildings = []
    for group in group_list:
        item = building_group_metadata(group)
        item["description"] = building_group_description(group)
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
