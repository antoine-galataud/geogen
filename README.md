# geogen

Generate OpenStudio and EnergyPlus building geometry from addresses in France and Great Britain.

`geogen` retrieves building footprints and attributes from:

- [BDNB](https://bdnb.io) in France;
- [Ordnance Survey Building Features](https://www.ordnancesurvey.co.uk/products/os-building-features) in England, Scotland, and Wales.

It generates an OpenStudio model (`.osm`) or EnergyPlus IDF (`.idf`), with optional SVG previews and JSON metadata. Multiple addresses, including mixed French and British portfolios, can be combined in one model.

## Features

- Polygon and multipart building footprints, including courtyards
- One space and thermal zone per storey and footprint part
- Building elevations and matched adjoining surfaces
- Approximate windows and pitched roofs when source data or fallback options allow
- Static, orthographic SVG previews for reports and dashboards
- Provider-aware JSON metadata for individual buildings and portfolios

The generated model contains geometry and thermal zones only. It does not add constructions, schedules, loads, occupancy, HVAC, or other simulation inputs.

## Requirements and installation

- Python 3.12
- [Poetry](https://python-poetry.org/)

OpenStudio 3.11 is installed as a Python dependency.

```bash
poetry install
```

## Quick start

### France: BDNB

A BDNB API key is optional. Anonymous requests may be rate-limited.

```bash
export BDNB_API_KEY="your-api-key"  # optional
poetry run geogen "122 Rue Amelot, 75011 Paris, France"
```

The key can also be supplied with `--api-key`.

### Great Britain: Ordnance Survey

Create an [OS Data Hub API project](https://docs.os.uk/os-apis/core-concepts/getting-started-with-an-api-project) with access to both:

- OS Places API;
- OS NGD API – Features, using Building Features collection `bld-fts-building-4`.

OS APIs don't have fenestration info.

```bash
export OS_API_KEY="your-project-key"
poetry run geogen \
  "10 Downing Street, London, SW1A 2AA, UK" \
  -o london.osm
```

OS access is subject to Data Hub entitlements, quotas, licensing, and attribution requirements. Coverage is limited to England, Scotland, and Wales; Northern Ireland and Crown Dependencies are not supported.

#### How to register for a free OS Data Hub API key

1. Sign in or create an account at [OS Data Hub](https://osdatahub.os.uk/).
1. Create a new project in the [API Projects](https://osdatahub.os.uk/projects) section.
1. Enable the following APIs for your project:
   1. OS Places API: a free trial is available for testing (credit card info filling can be skipped).
   1. OS NGD API - Features
1. Copy the generated API key for your project.

## Outputs

### OpenStudio and EnergyPlus

OpenStudio (`osm`) is the default output format. Use `--output-format idf` to forward-translate the generated model to EnergyPlus IDF.

```bash
poetry run geogen \
  "122 Rue Amelot, 75011 Paris, France" \
  --output-format idf \
  -o building.idf
```

Without `--output`, the filename is derived from the provider building identifier and selected format.

### SVG preview

`--svg-output` writes a scalable, orthographic preview from the same processed footprints used by the model. It supports multiple buildings, courtyards, elevations, storey lines, roofs, and approximate windows.

```bash
poetry run geogen \
  "122 Rue Amelot, 75011 Paris, France" \
  -o building.osm \
  --svg-output building.svg
```

The view can be adjusted with `--svg-width`, `--svg-height`, `--svg-azimuth`, and `--svg-elevation`. The SVG is a static presentation asset, not an interactive viewer or simulation-authoritative geometry.

### JSON metadata

`--json-output` writes UTF-8 portfolio metadata without making another provider request.

```bash
poetry run geogen \
  "122 Rue Amelot, 75011 Paris, France" \
  -o building.osm \
  --json-output building.json
```

The JSON includes source addresses, building count, aggregate footprint and estimated floor areas, and one object per building. Building fields include:

- provider, country, CRS, and provider identifier (`bdnb_id` or `os_id`);
- address and city;
- footprint area, storeys, height, elevation, and estimated floor area;
- glazing and roof attributes when available;
- fictitious-geometry status and a short description.

Example:

```json
{
  "source_addresses": [
    "122 Rue Amelot, 75011 Paris, France",
    "120 Rue Amelot, 75011 Paris, France"
  ],
  "building_count": 2,
  "total_footprint_area_m2": 614.0,
  "total_estimated_floor_area_m2": 3070.0,
  "buildings": [
    {
      "building_id": "bdnb-bg-4K9H-BPAU-3CHX",
      "provider": "bdnb",
      "country": "FR",
      "crs": "EPSG:2154",
      "address": "122 Rue Amelot 75011 Paris 11e Arrondissement",
      "city": "Paris 11e Arrondissement",
      "footprint_area_m2": 115.0,
      "number_of_storeys": 5,
      "height_m": 13.0,
      "ground_elevation_m": 35.0,
      "estimated_floor_area_m2": 575.0,
      "fictitious_geometry": false,
      "glazing_ratio": 0.31,
      "roof_material": "ZINC ALUMINIUM",
      "roof_type": null,
      "roof_shape": null,
      "bdnb_id": "bdnb-bg-4K9H-BPAU-3CHX",
      "description": "5-storey building located at 122 Rue Amelot, in Paris 11e Arrondissement, with an approximate footprint of 115 m², an estimated floor area of 575 m², and a height of 13.0 m."
    },
    {
      "building_id": "bdnb-bg-8R6K-ZAC9-EFXR",
      "provider": "bdnb",
      "country": "FR",
      "crs": "EPSG:2154",
      "address": "5 Rue de Crussol 75011 Paris 11e Arrondissement",
      "city": "Paris 11e Arrondissement",
      "footprint_area_m2": 499.0,
      "number_of_storeys": 5,
      "height_m": 13.0,
      "ground_elevation_m": 35.0,
      "estimated_floor_area_m2": 2495.0,
      "fictitious_geometry": false,
      "glazing_ratio": 0.24,
      "roof_material": "ZINC ALUMINIUM",
      "roof_type": "inconnu",
      "roof_shape": null,
      "bdnb_id": "bdnb-bg-8R6K-ZAC9-EFXR",
      "description": "5-storey building located at 5 Rue de Crussol, in Paris 11e Arrondissement, with an approximate footprint of 499 m², an estimated floor area of 2495 m², and a height of 13.0 m."
    }
  ]
}
```

Unknown source measurements remain `null`; geometry-generation defaults are not substituted into metadata. Estimated floor area is `footprint area × storey count` and is not a certified floor-area measurement.

### Generate all outputs together

```bash
poetry run geogen \
  "122 Rue Amelot, 75011 Paris, France" \
  -o building.osm \
  --svg-output building.svg \
  --json-output building.json
```

## Multiple addresses and country routing

Pass multiple addresses to create one model:

```bash
poetry run geogen \
  "120 Rue Amelot, 75011 Paris, France" \
  "122 Rue Amelot, 75011 Paris, France" \
  -o portfolio.osm \
  --svg-output portfolio.svg \
  --json-output portfolio.json
```

In the default `--country auto` mode, each address is routed independently using its country suffix or postcode. Use complete addresses, or override routing for the entire command with `--country FR`, `--country UK`, or `--country GB`.

Pure British portfolios use British National Grid (EPSG:27700). French and mixed portfolios use Lambert-93 (EPSG:2154). Because a model has one site/weather location and provider elevations use different vertical datums, geographically distant or mixed-country buildings should generally be generated separately.

## Main options

| Option                         | Description                                                       |
| ------------------------------ | ----------------------------------------------------------------- |
| `-k, --api-key TEXT`           | BDNB key; defaults to `$BDNB_API_KEY`                             |
| `--os-api-key TEXT`            | OS Data Hub key; defaults to `$OS_API_KEY`                        |
| `-o, --output PATH`            | Model output path; defaults to the provider-derived name          |
| `--output-format osm\|idf`     | Model format (default: `osm`)                                     |
| `--name TEXT`                  | Override the model name                                           |
| `--svg-output PATH`            | Write an SVG preview                                              |
| `--svg-width INTEGER`          | SVG canvas width (default `1200`)                                 |
| `--svg-height INTEGER`         | SVG canvas height (default `900`)                                 |
| `--svg-azimuth INTEGER`        | Orthographic camera azimuth (default `45` degrees)                |
| `--svg-elevation INTEGER`      | Orthographic camera elevation (default `28` degrees)              |
| `--json-output PATH`           | Write building and portfolio metadata                             |
| `--country auto\|FR\|UK\|GB`   | Select or automatically discover the provider                     |
| `--storey-height FLOAT`        | Fallback floor-to-floor height (default: `3.0` m)                 |
| `--window-to-wall-ratio FLOAT` | Fallback glazing ratio as a fraction (`0.2`) or percentage (`20`) |
| `--roof-pitch FLOAT`           | Fallback pitch for roofs inferred as pitched (default: `30°`)     |
| `--max-roof-height FLOAT`      | Cap generated pitched-roof height (default: `6.0` m)              |
| `--simplify-tolerance FLOAT`   | Footprint simplification tolerance (default: `0.1` m)             |
| `--max-buildings INTEGER`      | Maximum matched buildings per address (default: `10`)             |
| `--timeout FLOAT`              | Provider request timeout (default: `30` seconds)                  |
| `-v, --verbose`                | Enable debug logging                                              |

Run `poetry run geogen --help` for all options, including SVG camera settings and BDNB/OS endpoint configuration.

## Provider and geometry notes

- **BDNB:** accepts anonymous or authenticated requests and supplements building records with available envelope data. Geometry flagged as fictitious is warned about but still modeled.
- **Ordnance Survey:** requires one API key with Places and NGD Features access. The address point must fall inside a returned building footprint; there is no nearest-building fallback. OS data does not provide fenestration through this integration, so use `--window-to-wall-ratio` to add estimated windows.
- Missing height or storey data is inferred using `--storey-height`.
- Provider glazing data takes precedence over `--window-to-wall-ratio`; otherwise walls remain unglazed when no fallback is supplied.
- Roofs and windows are approximations. Actual roof faces, varying-height parts, basements, and detailed window geometry are not reconstructed.
- Very small or unusable footprint parts may be repaired or skipped. A run fails if no usable geometry remains.

## Development

```bash
poetry install
poetry run pytest
```
