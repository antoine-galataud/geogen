"""Command line interface of geogen."""

from __future__ import annotations

import logging
from contextlib import ExitStack
from pathlib import Path

import click

from geogen import __version__
from geogen.bdnb import API_KEY_ENV_VAR, DEFAULT_BASE_URL, DEFAULT_TIMEOUT
from geogen.envelope import (
    DEFAULT_MAX_ROOF_HEIGHT,
    DEFAULT_ROOF_PITCH,
    MAX_WINDOW_TO_WALL_RATIO,
)
from geogen.geometry import (
    DEFAULT_SIMPLIFY_TOLERANCE,
    DEFAULT_STOREY_HEIGHT,
    GeometryError,
    footprints_from_group,
)
from geogen.models import AddressNotFoundError, Building, ProviderError
from geogen.ordnance_survey import OsClient
from geogen.osm import (
    DEFAULT_OUTPUT_FORMAT,
    OUTPUT_FORMATS,
    ModelError,
    build_model,
    model_file_name,
    model_file_suffix,
    model_name,
    save_model,
)
from geogen.providers import BdnbProvider, discover_country
from geogen.svg import (
    DEFAULT_CAMERA_AZIMUTH,
    DEFAULT_CAMERA_ELEVATION,
    DEFAULT_SVG_HEIGHT,
    DEFAULT_SVG_WIDTH,
    SvgError,
    save_svg,
)

LOGGER = logging.getLogger(__name__)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("addresses", nargs=-1, required=True)
@click.option(
    "-k",
    "--api-key",
    envvar=API_KEY_ENV_VAR,
    required=False,
    help=f"BDNB API key (can also be set with ${API_KEY_ENV_VAR}). When not provided, the BDNB API is queried anonymously, which is rate limited.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Path of the model to write. Defaults to the name of the building, "
    "that is the provider identifier, in the current directory.",
)
@click.option(
    "--name",
    default=None,
    help="Name of the building in the model. Defaults to the provider identifier.",
)
@click.option(
    "--svg-output",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Optional path of an SVG 3D preview generated from the same building footprints.",
)
@click.option(
    "--svg-width",
    type=click.IntRange(min=100),
    default=DEFAULT_SVG_WIDTH,
    show_default=True,
    help="Width in pixels of the SVG preview canvas.",
)
@click.option(
    "--svg-height",
    type=click.IntRange(min=100),
    default=DEFAULT_SVG_HEIGHT,
    show_default=True,
    help="Height in pixels of the SVG preview canvas.",
)
@click.option(
    "--svg-azimuth",
    type=float,
    default=DEFAULT_CAMERA_AZIMUTH,
    show_default=True,
    help="Azimuth in degrees of the orthographic SVG camera.",
)
@click.option(
    "--svg-elevation",
    type=click.FloatRange(min=0, max=90, max_open=True),
    default=DEFAULT_CAMERA_ELEVATION,
    show_default=True,
    help="Elevation in degrees of the orthographic SVG camera.",
)
@click.option(
    "--output-format",
    type=click.Choice(sorted(OUTPUT_FORMATS)),
    default=DEFAULT_OUTPUT_FORMAT,
    show_default=True,
    help="Format of the model to write: an OpenStudio model (osm) or an EnergyPlus IDF file "
    "(idf), the latter obtained by forward translating the OpenStudio model.",
)
@click.option(
    "--storey-height",
    type=click.FloatRange(min=0, min_open=True),
    default=DEFAULT_STOREY_HEIGHT,
    show_default=True,
    help="Floor to floor height used when the building height or storey count is missing.",
)
@click.option(
    "--simplify-tolerance",
    type=click.FloatRange(min=0),
    default=DEFAULT_SIMPLIFY_TOLERANCE,
    show_default=True,
    help="Tolerance in metres used to simplify the footprints.",
)
@click.option(
    "--window-to-wall-ratio",
    type=click.FloatRange(min=0, max=100 * MAX_WINDOW_TO_WALL_RATIO),
    default=None,
    help="Estimated share of the exterior walls covered by windows, used for the buildings "
    "whose fenestration is unknown to the data provider. Given either as a fraction (0.2) or as a "
    "percentage (20). Those buildings are left without any window when it is not given.",
)
@click.option(
    "--roof-pitch",
    type=click.FloatRange(min=0, max=80),
    default=DEFAULT_ROOF_PITCH,
    show_default=True,
    help="Pitch in degrees of the sloped roofs whose inclination is unknown to the data provider. "
    "Set to 0 to only slope the roofs whose inclination is known.",
)
@click.option(
    "--max-roof-height",
    type=click.FloatRange(min=0),
    default=DEFAULT_MAX_ROOF_HEIGHT,
    show_default=True,
    help="Maximum height in metres of a sloped roof.",
)
@click.option(
    "--max-buildings",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
    help="Maximum number of building groups downloaded per address.",
)
@click.option(
    "--base-url",
    envvar="BDNB_BASE_URL",
    default=DEFAULT_BASE_URL,
    show_default=True,
    help="Base URL of the BDNB API.",
)
@click.option(
    "--timeout",
    type=click.FloatRange(min=0, min_open=True),
    default=DEFAULT_TIMEOUT,
    show_default=True,
    help="Timeout in seconds of API requests.",
)
@click.option(
    "--country",
    type=click.Choice(["auto", "FR", "UK", "GB"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Country override for all addresses.",
)
@click.option(
    "--os-api-key", envvar="OS_API_KEY", help="OS key with Places and NGD Features access."
)
@click.option(
    "--os-building-collection",
    envvar="OS_BUILDING_COLLECTION",
    help="NGD collection ID (supported/default: bld-fts-building-4).",
)
@click.option("-v", "--verbose", is_flag=True, help="Print debug information.")
@click.version_option(__version__)
def main(
    addresses: tuple[str, ...],
    api_key: str,
    output: Path | None,
    name: str | None,
    svg_output: Path | None,
    svg_width: int,
    svg_height: int,
    svg_azimuth: float,
    svg_elevation: float,
    output_format: str,
    storey_height: float,
    simplify_tolerance: float,
    window_to_wall_ratio: float | None,
    roof_pitch: float,
    max_roof_height: float,
    max_buildings: int,
    base_url: str,
    timeout: float,
    verbose: bool,
    country: str,
    os_api_key: str | None,
    os_building_collection: str | None,
) -> None:
    """Generate an OpenStudio geometry model (.osm) of the buildings at ADDRESSES.

    Geometry is downloaded from BDNB (France) or Ordnance Survey (Great Britain) and
    extruded into one space per storey.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s: %(message)s"
    )

    buildings = _download_buildings(
        addresses,
        api_key,
        base_url,
        timeout,
        max_buildings,
        country=country,
        os_api_key=os_api_key,
        os_building_collection=os_building_collection,
    )
    if not buildings:
        raise click.ClickException("No building found for the given addresses")

    footprints = _build_footprints(
        buildings, storey_height, simplify_tolerance, roof_pitch, window_to_wall_ratio
    )
    if not footprints:
        raise click.ClickException("None of the buildings found has a usable geometry")

    building_name = name or model_name(building.code for building in buildings.values())
    try:
        model = build_model(
            footprints, building_name=building_name, max_roof_height=max_roof_height
        )
        destination = save_model(
            model,
            output or Path(model_file_name(building_name, model_file_suffix(output_format))),
            output_format=output_format,
        )
        svg_destination = None
        if svg_output is not None:
            svg_destination = save_svg(
                footprints,
                svg_output,
                width=svg_width,
                height=svg_height,
                azimuth=svg_azimuth,
                elevation=svg_elevation,
                max_roof_height=max_roof_height,
            )
    except (ModelError, SvgError) as error:
        raise click.ClickException(str(error)) from error

    click.echo(
        f"Wrote {destination} with {len(buildings)} building(s), "
        f"{len(footprints)} footprint(s), {len(model.getSpaces())} space(s) and "
        f"{len(model.getSubSurfaces())} window(s)"
    )
    if svg_destination is not None:
        click.echo(f"Wrote SVG preview {svg_destination}")


def _download_buildings(
    addresses: tuple[str, ...],
    api_key: str | None,
    base_url: str,
    timeout: float,
    max_buildings: int,
    *,
    country: str = "auto",
    os_api_key: str | None = None,
    os_building_collection: str | None = None,
) -> dict[str, Building]:
    """Resolve countries before network I/O and reuse each country's provider."""
    buildings: dict[str, Building] = {}
    try:
        routes = [(address, discover_country(address, country)) for address in addresses]
        with ExitStack() as stack:
            clients = {}
            for address, selected in routes:
                if selected not in clients:
                    client = (
                        BdnbProvider(api_key, base_url=base_url, timeout=timeout)
                        if selected == "FR"
                        else OsClient(
                            os_api_key, timeout=timeout, collection=os_building_collection
                        )
                    )
                    stack.callback(client.close)
                    clients[selected] = client
                try:
                    found = clients[selected].buildings_for_address(
                        address, max_buildings=max_buildings
                    )
                except AddressNotFoundError as error:
                    click.echo(f"Warning: {error}", err=True)
                    continue
                for building in found:
                    buildings.setdefault(f"{selected}:{building.code}", building)
    except ProviderError as error:
        raise click.ClickException(str(error)) from error
    return buildings


def _build_footprints(
    buildings: dict[str, Building],
    storey_height: float,
    simplify_tolerance: float,
    roof_pitch: float,
    window_to_wall_ratio: float | None,
) -> list:
    """Convert the downloaded building groups into footprints."""
    footprints = []
    # Use one projected CRS for the entire model, including mixed-country input.
    target_crs = "EPSG:27700" if all(b.country == "UK" for b in buildings.values()) else "EPSG:2154"
    for building in buildings.values():
        if building.fictitious_geometry:
            click.echo(
                f"Warning: building {building.code} has a fictitious geometry in the BDNB",
                err=True,
            )
        try:
            footprints.extend(
                footprints_from_group(
                    building,
                    crs=target_crs,
                    default_storey_height=storey_height,
                    simplify_tolerance=simplify_tolerance,
                    default_roof_pitch=roof_pitch,
                    default_window_to_wall_ratio=window_to_wall_ratio,
                )
            )
        except GeometryError as error:
            click.echo(
                f"Warning: skipping building {building.code}: {error}",
                err=True,
            )
    return footprints


if __name__ == "__main__":  # pragma: no cover
    main()
