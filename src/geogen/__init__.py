"""Generate OpenStudio geometry models from BDNB building open data."""

from geogen.bdnb import Address, BdnbClient, BdnbError, BuildingGroup
from geogen.envelope import Envelope, envelope_from_group
from geogen.geometry import Footprint, footprints_from_group
from geogen.osm import build_model, save_model

__version__ = "0.1.0"

__all__ = [
    "Address",
    "BdnbClient",
    "BdnbError",
    "BuildingGroup",
    "Envelope",
    "Footprint",
    "build_model",
    "envelope_from_group",
    "footprints_from_group",
    "save_model",
    "__version__",
]
