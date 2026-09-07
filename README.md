# geogen

Generate geometry model in OpenStudio format from building open data (BDNB)

`geogen` is a command line tool that locates one or several buildings from their
postal addresses in the [BDNB](https://bdnb.io) (*Base de Données Nationale des
Bâtiments*), downloads their footprint, height and storey count, and writes an
OpenStudio model (`.osm`) containing the corresponding geometry. It can also produce an
EnergyPlus IDF file (`.idf`) instead, by forward translating the OpenStudio model.

The envelope is completed with windows and with a sloped roof when the BDNB
describes them, as an approximation. Nothing else is generated: no construction
or envelope properties, no usage, no occupancy and no HVAC.

## Requirements

- Python 3.12
- [Poetry](https://python-poetry.org/)

Optionnally, a valid BDNB API key, obtained from the
[BDNB API portal](https://api-portail.bdnb.io/catalog/api/f4905edc-db58-3a3b-a8e5-c5dfc6692ee5)

The OpenStudio 3.11 SDK is installed as a Python dependency, no separate
OpenStudio installation is needed.

## Installation

```bash
poetry install
```

## Usage

If you have a valid BDNB API key, set it in the environment variable `BDNB_API_KEY`. It's optional, without this
you'll get a quota of 10000 requests per month.

```bash
export BDNB_API_KEY=<your-api-key>
poetry run geogen "1 rue de la Paix, Paris"
# Wrote bdnb-bg-1234.osm with 1 building(s), 1 footprint(s), 3 space(s) and 6 window(s)
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

## How it works

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
  space above the top storey, covered by a hip roof whose apex stands above the
  point of the footprint that is the farthest from its boundary.

The BDNB is published in several editions, and the open one serves fewer tables
and fewer columns than the complete data model. A column or a table the API
rejects is dropped from the request, which is retried, and geogen remembers it
so it is only asked once; the envelope attributes missing from
`batiment_groupe_complet` are looked up in the table they come from
(`batiment_groupe_dpe_representatif_logement`, `batiment_groupe_ffo_bat`).

Known limitations: interior rings (courtyards) of the BDNB footprints are
ignored, a building group whose geometry is flagged as fictitious in the BDNB
(`contient_fictive_geom_groupe`) is reported but still modelled, sloped roofs
are added on top of `hauteur_mean` and their apex is capped by
`--max-roof-height`, which flattens the roofs of large footprints.

## Development

```bash
poetry install
poetry run pytest
```
