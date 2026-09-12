# geogen

Generate OpenStudio and EnergyPlus geometry from addresses in France and Great Britain.

`geogen` is a command line tool that locates one or several buildings from their
postal addresses using [BDNB](https://bdnb.io) in France or
[OS Building Features](https://www.ordnancesurvey.co.uk/products/os-building-features)
in Great Britain, downloads their footprint, height and storey count, and writes an
OpenStudio model (`.osm`) containing the corresponding geometry. It can also produce an
EnergyPlus IDF file (`.idf`) instead, by forward translating the OpenStudio model. An
optional vector SVG preview can be generated from the same geometry for dashboards and
report generation.

The envelope is completed with windows and with a sloped roof when the provider
describes them, as an approximation. Nothing else is generated: no construction
or envelope properties, no usage, no occupancy and no HVAC.

## Requirements

- Python 3.12
- [Poetry](https://python-poetry.org/)

Optionally, a valid BDNB API key, obtained from the
[BDNB API portal](https://api-portail.bdnb.io/catalog/api/f4905edc-db58-3a3b-a8e5-c5dfc6692ee5)

The OpenStudio 3.11 SDK is installed as a Python dependency, no separate
OpenStudio installation is needed.

## Installation

```bash
poetry install
```

## Usage

If you have a valid BDNB API key, set it in the environment variable `BDNB_API_KEY`. It is optional; anonymous access is subject to BDNB rate limits and quotas.

```bash
export BDNB_API_KEY=<your-api-key>
poetry run geogen "1 rue de la Paix, Paris"
# Wrote bdnb-bg-1234.osm with 1 building(s), 1 footprint(s), 3 space(s) and 6 window(s)

# Generate the OSM and a report-ready SVG preview in the same run
poetry run geogen "1 rue de la Paix, Paris" -o building.osm --svg-output building.svg
```

The model is named after the BDNB code of the building group (or of the building
itself when it belongs to no group), and so is the generated file unless
`--output` is given.

Several addresses can be passed at once to generate a single model containing a
group of buildings:

```bash
poetry run geogen "1 rue de la Paix, Paris" "3 rue de la Paix, Paris" -o block.osm
```

The API key can also be passed with `--api-key`. Run `geogen --help` for the
full list of options:

| Option                   | Description                                                             |
| ------------------------ | ----------------------------------------------------------------------- |
| `-o, --output`           | Path of the generated model file (default: the name of the building)    |
| `-k, --api-key`          | BDNB API key, defaults to `$BDNB_API_KEY`                               |
| `--name`                 | Name of the building of the model (default: its BDNB code)              |
| `--svg-output`           | Optional path of a vector SVG 3D preview                                |
| `--svg-width`            | SVG canvas width (default `1200`)                                       |
| `--svg-height`           | SVG canvas height (default `900`)                                       |
| `--svg-azimuth`          | Orthographic camera azimuth (default `45` degrees)                      |
| `--svg-elevation`        | Orthographic camera elevation (default `28` degrees)                    |
| `--output-format`        | Format of the generated model, `osm` or `idf` (default `osm`)           |
| `--storey-height`        | Storey height used when the BDNB data is incomplete (default `3.0` m)   |
| `--simplify-tolerance`   | Footprint simplification tolerance in metres (default `0.1`)            |
| `--window-to-wall-ratio` | Glazing estimate for the buildings unknown to the BDNB (no default)     |
| `--roof-pitch`           | Pitch of the sloped roofs unknown to the BDNB (default `30` degrees)    |
| `--max-roof-height`      | Maximum height of a sloped roof (default `6.0` m)                       |
| `--max-buildings`        | Maximum number of building groups downloaded per address (default `10`) |
| `--base-url`             | Base URL of the BDNB API, defaults to `$BDNB_BASE_URL`                  |
| `--timeout`              | Timeout of the API requests in seconds                                  |
| `-v, --verbose`          | Print debug information                                                 |

### Great Britain: setup and examples

Create an [OS Data Hub API project](https://docs.os.uk/os-apis/core-concepts/getting-started-with-an-api-project)
with **both OS Places API and OS NGD API – Features** enabled, and access to
OS Building Features (Building v4).

#### How to register for an OS Data Hub API key

1. Sign in or create an account at [OS Data Hub](https://osdatahub.os.uk/).
1. Create a new project in the [API Projects](https://osdatahub.os.uk/projects) section.
1. Enable the following APIs for your project:
   - OS Places API: a free trial is available for testing, without a credit card.
   - OS NGD API - Features
1. Copy the generated API key for your project.

Set its key:

```bash
export OS_API_KEY="your-project-key"
export ADDRESS="PRIME MINISTER & FIRST LORD OF THE TREASURY, 10, DOWNING STREET, LONDON, SW1A 2AA"
poetry run geogen "$ADDRESS" -o london.osm
poetry run geogen "$ADDRESS" \
  --output-format idf -o london.idf --svg-output london.svg \
  --window-to-wall-ratio 0.2
```

OS access is authenticated and subject to your OS plan, entitlements and licence;
it is not the anonymous French open-data service. Consult the OS Data Hub for
current terms and retain any attribution required when using or distributing
models and previews derived from OS data. A single key is used for both APIs.
The existing `--api-key` / `BDNB_API_KEY` still applies only to France.

**Coverage:** `UK` is the CLI country code, but the geometry service covers only
**England, Scotland and Wales**. Northern Ireland and the Crown Dependencies
are unsupported, even though OS Places can return their addresses.

### Country discovery

Country routing performs **zero API calls**. An explicit country suffix (France,
FR, UK, GB, United Kingdom, Great Britain, England, Scotland or Wales) or a full
postcode selects the provider. French five-digit and British alphanumeric
postcodes are recognized, including British postcodes without a space. In the
absence of those hints, French street words such as `rue`, `chemin`, `impasse`
and `allée` preserve the existing French workflow.

This is a two-country classifier, not a worldwide geocoder: a five-digit code
is treated as French within that scope. Use complete addresses. Contradictory
country/postcode hints and ambiguous inputs fail with a useful error before
any requests. For an address such as `10 High Street, Oxford`, either append
`UK` or pass `--country UK`. `--country FR|UK|GB` overrides discovery for every
address in that invocation; unsupported coverage is still rejected.

Each address is routed independently in automatic mode. Both providers share
the same generation options. Mixed FR/UK inputs are transformed into one
Lambert-93 coordinate frame; pure UK inputs use British National Grid. A model
has only one site and weather location, so distant buildings should normally
be generated in separate invocations. Ground elevations use each provider's
native vertical datum; no cross-country vertical-datum conversion is performed.

| Additional option          | Description                                                                                             |
| -------------------------- | ------------------------------------------------------------------------------------------------------- |
| \`--country auto           | FR                                                                                                      |
| `--os-api-key`             | OS key, defaults to `OS_API_KEY`                                                                        |
| `--os-building-collection` | `bld-fts-building-4`, also configurable as `OS_BUILDING_COLLECTION`; other schema versions are rejected |

### SVG preview

`--svg-output` generates a lightweight orthographic vector preview from the same
provider-derived footprints used to create the OpenStudio model. The SVG is intended as a
presentation/report asset; the `.osm` remains the authoritative simulation geometry.
Walls, roofs, storey separators and windows are emitted with CSS classes (`wall`, `roof`,
`window`, `storey-line`) so a downstream report generator can restyle them without
regenerating the geometry.

Example:

```bash
poetry run geogen "122 Rue Amelot, 75011 Paris, France" \
  -o 122_rue_amelot.osm \
  --svg-output 122_rue_amelot.svg
```

## How it works

`models.py` defines the common `Building` data contract and `BuildingProvider`
protocol. `providers.py` handles country routing and adapts the existing BDNB
client. `ordnance_survey.py` implements the independent OS provider. Shared
`geometry.py`, `envelope.py`, `osm.py` and `svg.py` implement all generation.
The existing `geogen.bdnb.BdnbClient` and `BuildingGroup` API remains available.

### UK retrieval and attribute mapping

1. `GET https://api.os.uk/search/places/v1/find` requests up to two DPA address
   candidates in EPSG:27700. The best match must score at least 0.8 and lead a
   different UPRN by at least 0.05; otherwise supply a more precise address.
1. `GET https://api.os.uk/features/ngd/ofa/v1/collections/bld-fts-building-4/items`
   requests footprints within a 2 m square around the address point, explicitly
   setting `bbox-crs` and output `crs` to EPSG:27700. Local point-in-polygon
   matching excludes neighbouring footprints returned by the bounding box.
   Pagination is followed when present, with pages capped at 100 features.
1. A normal address needs **two requests**. The v4 collection is pinned, avoiding
   discovery calls. Repeated normalized addresses and repeated UPRNs reuse
   in-memory results. Buildings are deduplicated by provider and identifier.
   `--max-buildings` caps the unique matched buildings returned per address.

| OS Building v4 field           | Generated model                                                        |
| ------------------------------ | ---------------------------------------------------------------------- |
| `osid` (or GeoJSON `id`)       | Stable model/space identifier, prefixed `os-`                          |
| GeoJSON geometry               | Polygon/MultiPolygon footprints in EPSG:27700                          |
| `height_relativeroofbase_m`    | Wall extrusion height in metres                                        |
| `height_relativemax_m`         | Fallback height; keeps the roof flat to avoid counting it twice        |
| `numberoffloors`               | Storeys above ground; missing values use shared height/storey defaults |
| `height_absolutemin_m`         | Ground elevation in metres                                             |
| `geometry_area_m2`             | Source footprint area metadata                                         |
| `roofshapeaspect_shape`        | Flat/pitched envelope hint; takes precedence over material             |
| `roofmaterial_primarymaterial` | Roof material hint when roof shape is unknown                          |

UK window locations and glazing ratios are not supplied by this integration.
Use `--window-to-wall-ratio` for estimated windows, or omit it for blind walls.
Pitched roofs use the shared approximate hip-roof algorithm and `--roof-pitch`
/ `--max-roof-height`; actual roof faces, roof aspect, building parts with
varying heights and basements are not reconstructed. Unknown and non-finite
numeric attributes fall back to the shared defaults.

Address matching is intentionally conservative: an address point outside the
building (for example in a courtyard or at a site entrance) produces a warning,
not a nearest-neighbour guess. This spatial lookup does not enumerate every
building on a campus associated with one postal address. Supply individual
building addresses for those cases. Weak/ambiguous address matches, missing
credentials, HTTP errors and malformed responses are distinguished from no
building found. Authentication/quota errors abort the run rather than switching
countries. As with France, unmatched addresses warn and the remaining buildings
can still be exported; no usable buildings means failure.

API references: [Places Find](https://docs.os.uk/os-apis/accessing-os-apis/os-places-api/technical-specification/find),
[NGD Features](https://docs.os.uk/os-apis/accessing-os-apis/os-ngd-api-features),
[Building schema](https://docs.os.uk/osngd/data-structure/buildings/building-features/building).

### France retrieval

For each address, `geogen` calls the BDNB API (`https://api.bdnb.io/v1/bdnb`):

1. `GET /geocodage` resolves the address to a BAN interoperability key
   (`cle_interop_adr`). The `donnees/adresse` table is used as a fallback.
1. `GET /donnees/rel_batiment_groupe_adresse` lists the building groups
   (`batiment_groupe_id`) located at this address.
1. `GET /donnees/batiment_groupe_complet` downloads the footprint
   (`geom_groupe`), the mean height (`hauteur_mean`), the storey count
   (`nb_niveau`) and the ground altitude (`altitude_sol_mean`) of each group,
   together with the envelope attributes described below.
1. `GET /donnees/batiment_groupe_wall_dict` downloads the description of the
   exterior surfaces of the building, which gives the inclination of its roof
   faces. This table is only published by the complete BDNB; the roof material
   is used instead when it is missing.
1. `GET /donnees/batiment_groupe_dpe_representatif_logement` and
   `GET /donnees/batiment_groupe_ffo_bat` complete the envelope attributes that
   `batiment_groupe_complet` does not publish. They are only called for the
   buildings that need them.

The footprints are then extruded into an OpenStudio model:

- BDNB geometries are expressed in Lambert-93 (EPSG:2154), whose unit is the
  metre; they are simply translated so that the centre of the modelled area
  becomes the origin of the model, which is also used to set the latitude and
  longitude of the site.
- Every building group is extruded into one `Space` per storey, each space being
  assigned to its own `BuildingStory`. The storey height is `hauteur_mean` /
  `nb_niveau`; missing values are replaced by `--storey-height`.
- Buildings are placed at their own ground altitude, relative to the lowest
  building of the model, which defines the site elevation; buildings whose
  altitude is unknown sit at that reference elevation.
- Surfaces are intersected and matched, so that adjacent storeys and adjoining
  buildings share interior surfaces.
- The building of the model, its storeys and its spaces are named after the BDNB
  code of the building group (`batiment_groupe_id`), so that the model can be
  traced back to the database. A model holding several groups is named after the
  first one, and the generated file takes the name of the building.

### Envelope

The BDNB does not publish the geometry of the windows nor the shape of the
roofs, so both are approximated from its attributes. **When the information is
missing, the envelope is left as is**: a building without glazing data gets no
window, and a building without roof data keeps the flat roof of its top storey.
`--window-to-wall-ratio` gives an estimate to use for the buildings whose
fenestration the BDNB does not describe.

- **Fenestration**: the share of the exterior walls covered by windows is
  `pourcentage_surface_baie_vitree_exterieur`, or the ratio between the glazed
  and exterior wall areas of the representative dwelling (`surface_vitree_nord`
  and friends, over `surface_mur_exterieur`). It is applied to every exterior
  wall left after the surface matching, so party walls stay blind. Only the
  walls facing a glazed orientation are glazed, taken from
  `l_orientation_baie_vitree` or from the facades whose glazed area is not zero.
  When the BDNB knows neither of these shares, the ratio given with
  `--window-to-wall-ratio` is used instead, as a fraction (`0.2`) or as a
  percentage (`20`); without it the building keeps its blind walls.
- **Roof shape**: the pitch of the roof is the mean inclination of the roof
  faces of `wall_dict`, weighted by their area. When that description is not
  available, the roof material (`materiaux_toiture_simplifie`, `mat_toit_txt`)
  and the construction of the upper floor (`type_plancher_haut_deperditif`) are
  used instead: a roof that can only be sloped (tiles, slate, zinc, attic...)
  gives a roof of `--roof-pitch` degrees, while a flat one (concrete, bitumen,
  terrace roof...) keeps the roof flat. A sloped roof is modelled as an attic
  space above the top storey. OpenStudio’s hip-roof generator constructs ridges
  and sloped faces from the footprint, replacing the former single-apex pyramid.

The BDNB is published in several editions, and the open one serves fewer tables
and fewer columns than the complete data model. A column or a table the API
rejects is dropped from the request, which is retried, and geogen remembers it
so it is only asked once; the envelope attributes missing from
`batiment_groupe_complet` are looked up in the table they come from
(`batiment_groupe_dpe_representatif_logement`, `batiment_groupe_ffo_bat`).

## Development

```bash
poetry install
poetry run pytest
```
