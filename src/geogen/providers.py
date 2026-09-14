"""Country routing and the adapter between BDNB and the common data model."""

from __future__ import annotations

import re
from dataclasses import replace

from geogen.bdnb import BdnbClient
from geogen.models import Building, ProviderError

UK_POSTCODE = re.compile(r"\b(?:GIR\s*0AA|[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})\b", re.I)
_COUNTRY_SUFFIX = re.compile(
    r"(?:^|[,\s])(?P<country>france|fr|uk|gb|united kingdom|great britain|england|scotland|wales|royaume[- ]uni)\s*$",
    re.I,
)


def discover_country(address: str, override: str = "auto") -> str:
    """Route without network calls; decline ambiguous/conflicting input.

    Country suffixes and postcodes are preferred. A small French vocabulary preserves
    existing postcode-free addresses; no city-name guessing is used.
    """
    if not address.strip():
        raise ProviderError("Address must not be empty")
    postcode = UK_POSTCODE.search(address)
    if (postcode and postcode.group().upper().startswith(("BT", "JE", "GY", "IM"))) or re.search(
        r"\bnorthern ireland\b", address, re.I
    ):
        raise ProviderError(
            "OS Building Features covers Great Britain only; Northern Ireland and Crown Dependencies are unsupported"
        )
    if override.lower() != "auto":
        country = override.upper()
        if country == "GB":
            country = "UK"
        if country not in {"FR", "UK"}:
            raise ProviderError(f"Unsupported country: {override}")
        return country
    suffix = _COUNTRY_SUFFIX.search(address)
    explicit = None
    if suffix:
        explicit = "FR" if suffix["country"].lower() in {"fr", "france"} else "UK"
    postal = "UK" if postcode else ("FR" if re.search(r"\b\d{5}\b", address) else None)
    if explicit and postal and explicit != postal:
        raise ProviderError(
            "Conflicting country and postcode; correct the address or use --country"
        )
    if explicit or postal:
        return explicit or postal
    if re.search(r"\b(?:rue|chemin|impasse|allee|allée)\b", address, re.I):
        return "FR"
    raise ProviderError(
        f"Cannot determine country for {address!r}; include France/UK or a postcode, or use --country FR/UK"
    )


class BdnbProvider:
    """Preserve BDNB's public client while returning provider-neutral buildings."""

    def __init__(self, api_key: str | None, **kwargs) -> None:
        self.client = BdnbClient(api_key, **kwargs)
        self._cache: dict[str, list[Building]] = {}

    def close(self) -> None:
        self.client.close()

    def buildings_for_address(self, query: str, *, max_buildings: int = 10) -> list[Building]:
        key = " ".join(query.casefold().split())
        cache_key = f"{max_buildings}:{key}"
        if cache_key not in self._cache:
            groups = self.client.buildings_for_address(query, max_buildings=max_buildings)
            self._cache[cache_key] = [
                replace(group, crs=group.crs or _bdnb_crs(group.geometry)) for group in groups
            ]
        return self._cache[cache_key][:max_buildings]


def _bdnb_crs(raw) -> str:
    from geogen.geometry import GeometryError, is_geographic, parse_geometry

    try:
        return "EPSG:4326" if is_geographic(parse_geometry(raw)) else "EPSG:2154"
    except (GeometryError, ValueError):
        return "EPSG:2154"
