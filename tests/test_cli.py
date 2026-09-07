"""Tests of the command line interface."""

from __future__ import annotations

from pathlib import Path

import openstudio
import pytest
import responses
from click.testing import CliRunner
from openstudio import osversion

from geogen.cli import main
from tests.conftest import BASE_URL, unknown_column_error, unknown_table_error


def _building_name(path: Path) -> str:
    """Name of the building of a saved model."""
    model = osversion.VersionTranslator().loadModel(openstudio.path(str(path)))
    return model.get().getBuilding().nameString()


def _window_to_wall_ratio(path: Path) -> float:
    """Share of the glazed walls of a saved model that is covered by windows."""
    model = osversion.VersionTranslator().loadModel(openstudio.path(str(path))).get()
    walls = [surface for surface in model.getSurfaces() if surface.subSurfaces()]
    glazing = sum(
        sub_surface.grossArea() for surface in walls for sub_surface in surface.subSurfaces()
    )
    return glazing / sum(surface.grossArea() for surface in walls)


def _translated(geometry: dict, offset: float) -> dict:
    """Move a GeoJSON multipolygon eastwards, to get a neighbouring building."""
    return {
        **geometry,
        "coordinates": [
            [[[x + offset, y] for x, y in ring] for ring in polygon]
            for polygon in geometry["coordinates"]
        ],
    }


def _register_api(building_row: dict, wall_dict_row: dict | None = None) -> None:
    responses.get(
        f"{BASE_URL}/geocodage",
        json={"features": [{"properties": {"id": "75102_7043_00001", "label": "1 rue"}}]},
    )
    responses.get(
        f"{BASE_URL}/donnees/rel_batiment_groupe_adresse",
        json=[{"batiment_groupe_id": "bdnb-0001"}],
    )
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_complet", json=[building_row])
    responses.get(
        f"{BASE_URL}/donnees/batiment_groupe_wall_dict",
        json=[wall_dict_row] if wall_dict_row else [],
    )


@responses.activate
def test_generates_a_model(building_row: dict, tmp_path: Path) -> None:
    _register_api(building_row)
    output = tmp_path / "model.osm"

    result = CliRunner().invoke(
        main,
        [
            "1 rue de la Paix Paris",
            "--api-key",
            "secret-key",
            "--base-url",
            BASE_URL,
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert "4 space(s)" in result.output
    assert "Storey 3" in output.read_text()


@responses.activate
def test_generates_an_idf_model(building_row: dict, tmp_path: Path) -> None:
    _register_api(building_row)
    output = tmp_path / "model.idf"

    result = CliRunner().invoke(
        main,
        [
            "1 rue de la Paix Paris",
            "--api-key",
            "secret-key",
            "--base-url",
            BASE_URL,
            "--output-format",
            "idf",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    idf_file = openstudio.IdfFile.load(openstudio.path(str(output)))
    assert idf_file.is_initialized()


@responses.activate
def test_reports_unknown_addresses(tmp_path: Path) -> None:
    responses.get(f"{BASE_URL}/geocodage", json={"features": []})
    responses.get(f"{BASE_URL}/donnees/adresse", json=[])

    result = CliRunner().invoke(
        main,
        [
            "nowhere",
            "--api-key",
            "secret-key",
            "--base-url",
            BASE_URL,
            "--output",
            str(tmp_path / "model.osm"),
        ],
    )

    assert result.exit_code == 1
    assert "No building found" in result.output


@responses.activate
def test_reports_api_errors(tmp_path: Path) -> None:
    responses.get(f"{BASE_URL}/geocodage", status=500)
    responses.get(f"{BASE_URL}/donnees/adresse", status=500)

    result = CliRunner().invoke(
        main,
        [
            "1 rue de la Paix Paris",
            "--api-key",
            "secret-key",
            "--base-url",
            BASE_URL,
            "--output",
            str(tmp_path / "model.osm"),
        ],
    )

    assert result.exit_code == 1
    assert "failed" in result.output


def test_doesnt_require_an_api_key(tmp_path: Path) -> None:
    result = CliRunner(env={"BDNB_API_KEY": ""}).invoke(
        main, ["1 rue de la Paix Paris", "--output", str(tmp_path / "model.osm")]
    )

    assert result.exit_code == 0
    assert "--api-key" not in result.output


@responses.activate
def test_takes_the_api_key_from_the_environment(building_row: dict, tmp_path: Path) -> None:
    _register_api(building_row)
    output = tmp_path / "model.osm"

    result = CliRunner(env={"BDNB_API_KEY": "secret-key"}).invoke(
        main, ["1 rue de la Paix Paris", "--base-url", BASE_URL, "--output", str(output)]
    )

    assert result.exit_code == 0, result.output
    assert responses.calls[0].request.headers["X-Gravitee-Api-Key"] == "secret-key"


@responses.activate
def test_deduplicates_buildings_shared_by_several_addresses(
    building_row: dict, tmp_path: Path
) -> None:
    _register_api(building_row)
    output = tmp_path / "model.osm"

    result = CliRunner().invoke(
        main,
        [
            "1 rue de la Paix Paris",
            "3 rue de la Paix Paris",
            "--api-key",
            "secret-key",
            "--base-url",
            BASE_URL,
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "1 building(s)" in result.output


@responses.activate
def test_completes_the_envelope_of_the_buildings(
    building_row: dict, wall_dict_row: dict, tmp_path: Path
) -> None:
    _register_api(building_row, wall_dict_row)
    output = tmp_path / "model.osm"

    result = CliRunner().invoke(
        main,
        [
            "1 rue de la Paix Paris",
            "--api-key",
            "secret-key",
            "--base-url",
            BASE_URL,
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "6 window(s)" in result.output
    assert "bdnb-0001 Attic Space" in output.read_text()


@responses.activate
def test_keeps_the_envelope_as_is_without_bdnb_data(geom_groupe: dict, tmp_path: Path) -> None:
    _register_api({"batiment_groupe_id": "bdnb-0001", "geom_groupe": geom_groupe})
    output = tmp_path / "model.osm"

    result = CliRunner().invoke(
        main,
        [
            "1 rue de la Paix Paris",
            "--api-key",
            "secret-key",
            "--base-url",
            BASE_URL,
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "0 window(s)" in result.output
    assert "Attic" not in output.read_text()


@responses.activate
def test_completes_the_envelope_from_the_open_bdnb(
    building_row: dict, dpe_row: dict, tmp_path: Path
) -> None:
    """The open BDNB has neither the wall_dict table nor every envelope column."""
    published = dict(
        building_row,
        pourcentage_surface_baie_vitree_exterieur=None,
        l_orientation_baie_vitree=None,
        mat_toit_txt=None,
        type_plancher_haut_deperditif=None,
    )
    published.pop("materiaux_toiture_simplifie")
    responses.get(
        f"{BASE_URL}/geocodage",
        json={"features": [{"properties": {"id": "75102_7043_00001", "label": "1 rue"}}]},
    )
    responses.get(
        f"{BASE_URL}/donnees/rel_batiment_groupe_adresse",
        json=[{"batiment_groupe_id": "bdnb-0001"}],
    )
    responses.get(
        f"{BASE_URL}/donnees/batiment_groupe_complet",
        json=unknown_column_error("batiment_groupe_complet", "materiaux_toiture_simplifie"),
        status=400,
    )
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_complet", json=[published])
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_dpe_representatif_logement", json=[dpe_row])
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_ffo_bat", json=[{"mat_toit_txt": "Tuiles"}])
    responses.get(
        f"{BASE_URL}/donnees/batiment_groupe_wall_dict",
        json=unknown_table_error("batiment_groupe_wall_dict"),
        status=404,
    )
    output = tmp_path / "model.osm"

    result = CliRunner().invoke(
        main,
        [
            "1 rue de la Paix Paris",
            "--api-key",
            "secret-key",
            "--base-url",
            BASE_URL,
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "6 window(s)" in result.output
    assert "bdnb-0001 Attic Space" in output.read_text()


@responses.activate
def test_names_the_model_after_the_building_code(
    building_row: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register_api(building_row)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        main, ["1 rue de la Paix Paris", "--api-key", "secret-key", "--base-url", BASE_URL]
    )
    output = tmp_path / "bdnb-0001.osm"

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert "Wrote bdnb-0001.osm" in result.output
    assert _building_name(output) == "bdnb-0001"


@responses.activate
def test_default_output_file_uses_the_output_format_suffix(
    building_row: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register_api(building_row)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "1 rue de la Paix Paris",
            "--api-key",
            "secret-key",
            "--base-url",
            BASE_URL,
            "--output-format",
            "idf",
        ],
    )
    output = tmp_path / "bdnb-0001.idf"

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert "Wrote bdnb-0001.idf" in result.output


@responses.activate
def test_names_the_model_after_the_first_of_several_buildings(
    building_row: dict, geom_groupe: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses.get(
        f"{BASE_URL}/geocodage",
        json={"features": [{"properties": {"id": "75102_7043_00001", "label": "1 rue"}}]},
    )
    responses.get(
        f"{BASE_URL}/donnees/rel_batiment_groupe_adresse",
        json=[{"batiment_groupe_id": "bdnb-0001"}, {"batiment_groupe_id": "bdnb-0002"}],
    )
    neighbour = {
        **building_row,
        "batiment_groupe_id": "bdnb-0002",
        "geom_groupe": _translated(geom_groupe, 100.0),
    }
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_complet", json=[building_row])
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_complet", json=[neighbour])
    responses.get(f"{BASE_URL}/donnees/batiment_groupe_wall_dict", json=[])
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        main, ["1 rue de la Paix Paris", "--api-key", "secret-key", "--base-url", BASE_URL]
    )
    output = tmp_path / "bdnb-0001_and_1_more.osm"

    assert result.exit_code == 0, result.output
    assert "2 building(s)" in result.output
    assert output.exists()
    assert _building_name(output) == "bdnb-0001 and 1 more"


@responses.activate
def test_names_the_model_after_the_given_name(
    building_row: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register_api(building_row)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "1 rue de la Paix Paris",
            "--api-key",
            "secret-key",
            "--base-url",
            BASE_URL,
            "--name",
            "Hotel de ville",
        ],
    )
    output = tmp_path / "Hotel_de_ville.osm"

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert _building_name(output) == "Hotel de ville"


@responses.activate
def test_estimates_the_fenestration_missing_from_the_bdnb(
    geom_groupe: dict, tmp_path: Path
) -> None:
    _register_api({"batiment_groupe_id": "bdnb-0001", "geom_groupe": geom_groupe})
    output = tmp_path / "model.osm"

    result = CliRunner().invoke(
        main,
        [
            "1 rue de la Paix Paris",
            "--api-key",
            "secret-key",
            "--base-url",
            BASE_URL,
            "--window-to-wall-ratio",
            "25",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "4 window(s)" in result.output
    assert _window_to_wall_ratio(output) == pytest.approx(0.25, abs=1e-2)
    assert "Attic" not in output.read_text()


@responses.activate
def test_the_bdnb_wins_over_the_estimated_fenestration(building_row: dict, tmp_path: Path) -> None:
    _register_api(building_row)
    output = tmp_path / "model.osm"

    result = CliRunner().invoke(
        main,
        [
            "1 rue de la Paix Paris",
            "--api-key",
            "secret-key",
            "--base-url",
            BASE_URL,
            "--window-to-wall-ratio",
            "25",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert _window_to_wall_ratio(output) == pytest.approx(0.2, abs=1e-2)


@responses.activate
def test_rejects_an_unusable_window_to_wall_ratio(building_row: dict, tmp_path: Path) -> None:
    _register_api(building_row)

    result = CliRunner().invoke(
        main,
        [
            "1 rue de la Paix Paris",
            "--api-key",
            "secret-key",
            "--base-url",
            BASE_URL,
            "--window-to-wall-ratio",
            "120",
            "--output",
            str(tmp_path / "model.osm"),
        ],
    )

    assert result.exit_code == 2
    assert "120" in result.output
