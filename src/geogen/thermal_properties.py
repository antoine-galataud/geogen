"""Scoped, partial thermal enrichment, without inferring assemblies from labels.

Opaque U-values are interpreted as air-to-air W/(m² K). Equivalent massless
layers subtract explicit conventional inside/outside films from 1/U, not the
temperature-dependent films used during simulation. This reproduces a nominal
resistance only: it does not recover thermal mass, an actual material assembly,
or a ground heat-transfer model. Glazing requires both U and SHGC; visible
transmittance remains unset. None of these enrichments makes a model simulation-ready.
"""

from __future__ import annotations

import json
import math
import unicodedata

from openstudio import model as osmodel

from geogen.bdnb import THERMAL_COLUMNS

_OPAQUE_FIELDS = {
    "Wall": ("wall", ("wall_insulation",), "wall_u_value", 0.13),
    "RoofCeiling": (
        "roof",
        ("upper_floor_insulation", "roof_material"),
        "roof_u_value",
        0.10,
    ),
    "Floor": ("floor", ("lower_floor_insulation",), "floor_u_value", 0.17),
}
_CLASSIFICATIONS = (
    "wall_insulation",
    "upper_floor_insulation",
    "lower_floor_insulation",
    "glazing_type",
    "roof_material",
)
ENVELOPE_FIELDS = (*_CLASSIFICATIONS, *THERMAL_COLUMNS)
_UNKNOWN = {
    "",
    "-",
    "unknown",
    "inconnu",
    "inconnue",
    "non renseigne",
    "non renseignee",
    "non disponible",
    "not provided",
    "not_available",
    "none",
    "null",
    "n/a",
    "na",
    "nc",
    "nr",
    "conflict",
    "conflicting",
}
_EXTERIOR_FILM = 0.04


def _label(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    normalized = "".join(
        c for c in unicodedata.normalize("NFKD", value.casefold()) if not unicodedata.combining(c)
    )
    return value if normalized not in _UNKNOWN else None


def _number(value, *, solar=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
    except OverflowError:
        return None
    if not math.isfinite(value) or (not 0 <= value <= 1 if solar else value <= 0):
        return None
    return value


def _report_data(value):
    """Keep source evidence JSON-safe even when a rejected value is nonfinite."""
    if isinstance(value, dict):
        return {key: _report_data(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_report_data(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_numeric": str(value)}
    return value


def _set_provenance(obj, source, values):
    properties = obj.additionalProperties()
    properties.setFeature("geogen_thermal_source_group_ids", json.dumps(source["source_group_ids"]))
    properties.setFeature("geogen_thermal_scope", source["scope"])
    properties.setFeature(
        "geogen_envelope_properties",
        json.dumps({**source, "values": values}, sort_keys=True),
    )
    for field, value in values.items():
        properties.setFeature(f"geogen_{field}", value)


def _construction(model, name, values, field, *, resistance=None):
    """Return a checked construction/material pair, or None if the SDK rejects it."""
    material = (
        osmodel.SimpleGlazing(model)
        if resistance is None
        else osmodel.MasslessOpaqueMaterial(model)
    )
    construction = osmodel.Construction(model)
    material.setName(f"{name} material")
    construction.setName(name)
    if resistance is None:
        accepted = material.setUFactor(values[field]) and material.setSolarHeatGainCoefficient(
            values["window_solar_factor"]
        )
        material.resetVisibleTransmittance()
    else:
        accepted = material.setThermalResistance(resistance)
    if not accepted or not construction.insertLayer(0, material):
        construction.remove()
        material.remove()
        return None
    return construction, material


def apply_envelope_properties(
    model: osmodel.Model, mapping: dict[str, dict] | None = None
) -> dict[str, dict]:
    """Enrich exterior/ground faces after glazing, keyed by exact model space ID.

    Each record has ``source_group_ids``, ``scope`` and a ``values`` dictionary.
    Optional ``conflicts`` (field names or a field-keyed dictionary) and source
    ``decisions`` with a status other than ``available`` suppress those fields.
    Missing, invalid and conflicting values remain unset and are recorded,
    alongside applied counts, in Building AdditionalProperties.
    Legacy models can use the same helper with their exact space names.
    """
    from geogen.osm import ModelError

    if mapping is None:
        mapping = {}
    spaces = {space.nameString(): space for space in model.getSpaces()}
    if not isinstance(mapping, dict) or set(mapping) - set(spaces):
        raise ModelError("Envelope properties reference unknown thermal spaces")
    for record in mapping.values():
        if not isinstance(record, dict) or not isinstance(record.get("values", {}), dict):
            raise ModelError("Envelope properties must contain a values dictionary")
        group_ids = record.get("source_group_ids", [])
        if (
            not isinstance(group_ids, (list, tuple))
            or any(not isinstance(value, str) or not value.strip() for value in group_ids)
            or not isinstance(record.get("scope", "unknown"), str)
            or not isinstance(record.get("conflicts", ()), (list, tuple, set, dict))
            or not isinstance(record.get("decisions", {}), dict)
            or any(
                not isinstance(decision, dict) for decision in record.get("decisions", {}).values()
            )
            or not isinstance(record.get("links", []), (list, tuple))
            or any(not isinstance(link, dict) for link in record.get("links", []))
        ):
            raise ModelError("Envelope properties have invalid source scope or conflicts")

    reports = {}
    for space_id, space in spaces.items():
        record = mapping.get(space_id, {})
        source = {
            "source_group_ids": list(dict.fromkeys(record.get("source_group_ids", []))),
            "scope": record.get("scope", "unknown"),
        }
        counts = {
            "classified_surfaces": 0,
            "classified_windows": 0,
            "wall_constructions": 0,
            "roof_constructions": 0,
            "floor_constructions": 0,
            "window_constructions": 0,
        }
        report = {
            **source,
            "decisions": _report_data(record.get("decisions", {})),
            "links": _report_data(record.get("links", [])),
            "bdnb_numeric_field_mapping": dict(THERMAL_COLUMNS),
            "source_limitation": (
                "BDNB numeric fields are representative DPE group values, not face measurements. "
                "Raw lower-floor U excludes ground adjustment; window Uw excludes shutter Ujn."
            ),
            "applied_counts": counts,
            "omissions": [],
            "assumptions": [],
            "values": {},
        }
        reports[space_id] = report
        values = report["values"]
        raw = record.get("values", {})
        conflicts = record.get("conflicts", ())
        decisions = record.get("decisions", {})
        for field in ENVELOPE_FIELDS:
            decision = decisions.get(field, {})
            if field in conflicts:
                reason, value = "conflicting", None
            elif decision.get("status", "available") != "available":
                reason, value = decision["status"], None
            elif field in _CLASSIFICATIONS:
                value = _label(raw.get(field))
                reason = "missing_or_unknown" if value is None else None
            else:
                value = _number(raw.get(field), solar=field == "window_solar_factor")
                reason = (
                    ("missing" if raw.get(field) is None else "invalid") if value is None else None
                )
            if value is None:
                report["omissions"].append({"field": field, "reason": reason})
            else:
                values[field] = value
        eligible = {"Wall": [], "RoofCeiling": [], "Floor": []}
        windows = []
        for surface in space.surfaces():
            if surface.outsideBoundaryCondition() not in ("Outdoors", "Ground"):
                continue
            interface = surface.additionalProperties().getFeatureAsString("geogen_interface_id")
            if interface.is_initialized() and interface.get():
                continue
            if surface.surfaceType() in eligible:
                eligible[surface.surfaceType()].append(surface)
            for window in surface.subSurfaces():
                if (
                    surface.outsideBoundaryCondition() == "Outdoors"
                    and window.subSurfaceType() in ("FixedWindow", "OperableWindow", "Skylight")
                    and window.outsideBoundaryCondition() == "Outdoors"
                ):
                    windows.append(window)
                else:
                    report["omissions"].append(
                        {"object": window.nameString(), "reason": "unsupported_subsurface"}
                    )

        report["eligible_surface_counts"] = {key: len(faces) for key, faces in eligible.items()}
        report["eligible_window_count"] = len(windows)
        for surface_type, (kind, labels, u_field, inside_film) in _OPAQUE_FIELDS.items():
            surfaces = eligible[surface_type]
            relevant = {field: values[field] for field in (*labels, u_field) if field in values}
            for surface in surfaces:
                if relevant:
                    _set_provenance(surface, source, relevant)
                if any(field in relevant for field in labels):
                    counts["classified_surfaces"] += 1
            if u_field not in values:
                continue
            if not surfaces:
                report["omissions"].append({"field": u_field, "reason": "no_eligible_surfaces"})
                continue
            resistance = 1.0 / values[u_field] - inside_film - _EXTERIOR_FILM
            if not math.isfinite(resistance) or resistance <= 0:
                report["omissions"].append(
                    {
                        "field": u_field,
                        "reason": (
                            "nonpositive_resistance_after_films"
                            if math.isfinite(resistance)
                            else "nonfinite_equivalent_resistance"
                        ),
                    }
                )
                continue
            result = _construction(
                model,
                f"{space_id} sourced {kind} equivalent R",
                values,
                u_field,
                resistance=resistance,
            )
            if result is None:
                report["omissions"].append({"field": u_field, "reason": "sdk_rejected_material"})
                continue
            construction, material = result
            assumption = {
                "field": u_field,
                "bdnb_source_field": THERMAL_COLUMNS[u_field],
                "source_scope": source["scope"],
                "source_u_value_w_m2k": values[u_field],
                "inside_film_m2k_w": inside_film,
                "outside_film_m2k_w": _EXTERIOR_FILM,
                "equivalent_layer_resistance_m2k_w": material.thermalResistance(),
                "limitation": (
                    "Air-to-air U assumed; equivalent massless resistance only, not an actual "
                    "assembly. SDK default roughness/absorptances retained; thermal mass unknown."
                ),
            }
            if kind == "roof":
                assumption["mapping_assumption"] = (
                    "Representative upper-loss-floor mapped onto roof, no attic exclusion."
                )
            if any(surface.outsideBoundaryCondition() == "Ground" for surface in surfaces):
                assumption["ground_limitation"] = (
                    "Raw lower-floor U converted to a nominal layer only, not a ground-adjusted "
                    "coefficient; ground coupling is not resolved by this construction."
                )
            report["assumptions"].append(assumption)
            for obj in (construction, material):
                _set_provenance(obj, source, relevant)
                obj.additionalProperties().setFeature(
                    "geogen_thermal_assumption", json.dumps(assumption, sort_keys=True)
                )
            for surface in surfaces:
                if surface.setConstruction(construction):
                    counts[f"{kind}_constructions"] += 1
                else:
                    report["omissions"].append(
                        {"object": surface.nameString(), "reason": "sdk_rejected_construction"}
                    )
            if not counts[f"{kind}_constructions"]:
                construction.remove()
                material.remove()

        relevant = {
            field: values[field]
            for field in ("glazing_type", "window_u_value", "window_solar_factor")
            if field in values
        }
        for window in windows:
            if relevant:
                _set_provenance(window, source, relevant)
            if "glazing_type" in relevant:
                counts["classified_windows"] += 1
        if "window_u_value" in values and "window_solar_factor" in values:
            if not windows:
                report["omissions"].append(
                    {"field": "window_construction", "reason": "no_eligible_windows"}
                )
            else:
                result = _construction(
                    model, f"{space_id} sourced simple glazing", values, "window_u_value"
                )
                if result is None:
                    report["omissions"].append(
                        {"field": "window_construction", "reason": "sdk_rejected_material"}
                    )
                else:
                    construction, material = result
                    assumption = {
                        "field": "window_construction",
                        "bdnb_source_fields": {
                            field: THERMAL_COLUMNS[field]
                            for field in ("window_u_value", "window_solar_factor")
                        },
                        "source_scope": source["scope"],
                        "limitation": (
                            "Equivalent SimpleGlazing from sourced U and SHGC, not an actual "
                            "glazing assembly. Visible transmittance unset; engine defaults apply."
                        ),
                        "visible_transmittance": None,
                    }
                    report["assumptions"].append(assumption)
                    for obj in (construction, material):
                        _set_provenance(obj, source, relevant)
                        obj.additionalProperties().setFeature(
                            "geogen_thermal_assumption", json.dumps(assumption, sort_keys=True)
                        )
                    for window in windows:
                        if window.setConstruction(construction):
                            counts["window_constructions"] += 1
                        else:
                            report["omissions"].append(
                                {
                                    "object": window.nameString(),
                                    "reason": "sdk_rejected_construction",
                                }
                            )
                    if not counts["window_constructions"]:
                        construction.remove()
                        material.remove()
        elif windows:
            report["omissions"].append(
                {"field": "window_construction", "reason": "requires_both_u_value_and_solar_factor"}
            )
        report["status"] = "partial" if any(counts.values()) else "unavailable"
        space.additionalProperties().setFeature(
            "geogen_thermal_enrichment", json.dumps(report, sort_keys=True)
        )
    model.getBuilding().additionalProperties().setFeature(
        "geogen_thermal_enrichment", json.dumps(reports, sort_keys=True)
    )
    return reports
