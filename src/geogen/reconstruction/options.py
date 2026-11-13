"""LiDAR/Roofer CLI flags; importing this module never starts native tools."""

from pathlib import Path

import click

from geogen.reconstruction.cache import DEFAULT_CACHE


def reconstruction_options(function):
    options = [
        click.option(
            "--roof-source",
            type=click.Choice(["legacy", "roofer"]),
            default="legacy",
            show_default=True,
            help="legacy: extrude BDNB/Ordnance Survey data. roofer: reconstruct French roofs "
            "from IGN LiDAR HD with Roofer (requires Docker).",
        ),
        click.option(
            "--roofer-missing-roof-height",
            type=click.FloatRange(min=0, max=300, min_open=True),
            default=None,
            help="Assumed height in metres above observed ground for buildings without usable "
            "LiDAR roof points. Defaults to the BDNB height when available; otherwise such "
            "buildings make the run fail.",
        ),
        click.option(
            "--reconstruction-input",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            help="Replay a previous acquisition manifest offline (no downloads).",
        ),
        click.option(
            "--reconstruction-cache",
            type=click.Path(file_okay=False, path_type=Path),
            default=DEFAULT_CACHE,
            show_default=True,
            help="Directory of acquisitions, Roofer results and run diagnostics.",
        ),
    ]
    for option in reversed(options):
        function = option(function)
    return function

