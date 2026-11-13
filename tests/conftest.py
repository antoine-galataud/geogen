"""Shared fixtures for the geogen test suite."""

from __future__ import annotations

import copy
import json

import pytest

BASE_URL = "https://api.bdnb.test/v1/bdnb"

#: A 20 m x 10 m rectangle in Lambert-93, in the middle of Paris.
RECTANGLE = [
    [
        [
            [652000.0, 6862000.0],
            [652020.0, 6862000.0],
            [652020.0, 6862010.0],
            [652000.0, 6862010.0],
            [652000.0, 6862000.0],
        ]
    ]
]


@pytest.fixture
def geom_groupe() -> dict:
    """A GeoJSON multipolygon as returned by the BDNB API."""
    return {"type": "MultiPolygon", "coordinates": RECTANGLE}


@pytest.fixture
def building_row(geom_groupe: dict) -> dict:
    """A row of the ``batiment_groupe_complet`` endpoint."""
    return {
        "batiment_groupe_id": "bdnb-0001",
        "geom_groupe": geom_groupe,
        "hauteur_mean": 9,
        "nb_niveau": 3,
        "altitude_sol_mean": 35,
        "s_geom_groupe": 200,
        "contient_fictive_geom_groupe": False,
        "libelle_adr_principale_ban": "1 rue de la Paix 75002 Paris",
        "libelle_commune_insee": "Paris",
        "pourcentage_surface_baie_vitree_exterieur": 20.0,
        "l_orientation_baie_vitree": ["nord", "sud"],
        "materiaux_toiture_simplifie": "tuiles",
        "mat_toit_txt": "Tuiles",
        "type_plancher_haut_deperditif": "Combles perdus",
        "type_isolation_mur_exterieur": "ITI",
        "type_isolation_plancher_haut": "isolé",
        "type_isolation_plancher_bas": "non isolé",
        "type_vitrage": "double vitrage",
    }


@pytest.fixture
def dpe_row() -> dict:
    """A row of the ``batiment_groupe_dpe_representatif_logement`` endpoint."""
    return {
        "surface_mur_exterieur": 200.0,
        "surface_vitree_nord": 10.0,
        "surface_vitree_est": 0.0,
        "surface_vitree_sud": 20.0,
        "surface_vitree_ouest": None,
        "type_isolation_mur_exterieur": "ITI",
        "type_isolation_plancher_haut": "isolé",
        "type_isolation_plancher_bas": "non isolé",
        "type_vitrage": "double vitrage",
    }


@pytest.fixture
def wall_dict_row() -> dict:
    """A row of the ``batiment_groupe_wall_dict`` endpoint."""
    return {
        "batiment_groupe_id": "bdnb-0001",
        "wall_dict": [
            {"wall_type": "roof", "inclination": 145.0, "area": 120.0, "azimuth": 0.0},
            {"wall_type": "vertical", "inclination": 90.0, "area": 60.0, "azimuth": 90.0},
            {"wall_type": "floor", "inclination": 0.0, "area": 200.0},
        ],
    }


def unknown_column_error(table: str, column: str) -> dict:
    """The body PostgREST returns when a column is not part of a table."""
    return {
        "code": "42703",
        "details": None,
        "hint": None,
        "message": f"column {table}.{column} does not exist",
    }


def unknown_table_error(table: str) -> dict:
    """The body PostgREST returns when a table is not part of the published schema."""
    return {
        "code": "PGRST205",
        "details": None,
        "hint": "Perhaps you meant the table 'api_open_202602a_1.batiment_groupe_ffo_bat'",
        "message": (f"Could not find the table 'api_open_202602a_1.{table}' in the schema cache"),
    }


@pytest.fixture
def offline_runtime(monkeypatch, tmp_path):
    import requests

    from geogen.reconstruction.runtime import DEFAULT_IMAGE, DockerRuntime

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    def no_http(*args, **kwargs):
        raise AssertionError("Offline CLI attempted a Python HTTP request")

    monkeypatch.setattr(requests.Session, "send", no_http)
    real_run = DockerRuntime.run
    calls = []

    def observed_run(self, args, work_dir, *, network=False):
        assert self.offline is True
        assert network is False
        result = real_run(self, args, work_dir, network=network)
        command = result.command
        assert command[command.index("--network") + 1] == "none"
        assert command[command.index("--pull") + 1] == "never"
        assert self.image == DEFAULT_IMAGE
        calls.append({"args": args, "result": result})
        return result

    monkeypatch.setattr(DockerRuntime, "run", observed_run)
    return calls


@pytest.fixture
def reconstruction(monkeypatch, tmp_path):
    from geogen.reconstruction.cache import ReconstructionCache
    from geogen.reconstruction.pipeline import reconstruct_manifest
    from tests.test_reconstruction_policy import EXAMPLE, Validator, captured_reconstruction

    runtime = Validator()
    manifest = json.loads(EXAMPLE.read_text())
    evidence = {"fidelity_status": "evidence_passed", "metrics": {}, "reasons": []}
    monkeypatch.setattr("geogen.reconstruction.roofer.reconstruct", captured_reconstruction)
    monkeypatch.setattr(
        "geogen.reconstruction.evidence.assess_evidence",
        lambda scene, units, path: {
            unit.reconstruction_id: copy.deepcopy(evidence) for unit in units
        },
    )
    calls = []

    def run(**policies):
        work = tmp_path / str(len(calls))
        work.mkdir()
        report = {"units": []}
        calls.append(report)
        return reconstruct_manifest(
            manifest,
            EXAMPLE,
            work,
            runtime,
            cache=ReconstructionCache(tmp_path / "cache"),
            report=report,
            **policies,
        )

    return run, runtime, manifest, evidence, calls
