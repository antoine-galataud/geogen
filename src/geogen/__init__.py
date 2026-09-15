"""Generate OpenStudio geometry models from provider-derived building data."""

from geogen.bdnb import BdnbClient, BdnbError
from geogen.envelope import Envelope, envelope_from_group
from geogen.geometry import Footprint, footprints_from_group
from geogen.metadata import (
    building_description,
    building_group_description,
    building_group_metadata,
    building_metadata,
    portfolio_metadata,
    save_metadata_json,
)
from geogen.models import Address, Building, BuildingGroup, BuildingProvider, OsBuilding
from geogen.osm import build_model, save_model

__version__ = "0.1.0"

__all__ = [
    "Address",
    "BdnbClient",
    "BdnbError",
    "Building",
    "BuildingGroup",
    "BuildingProvider",
    "OsBuilding",
    "Envelope",
    "Footprint",
    "build_model",
    "envelope_from_group",
    "footprints_from_group",
    "save_model",
    "building_description",
    "building_metadata",
    "building_group_description",
    "building_group_metadata",
    "portfolio_metadata",
    "save_metadata_json",
    "__version__",
]
