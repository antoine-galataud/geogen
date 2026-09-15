"""Tests of the OpenStudio model generation."""

from __future__ import annotations

import math
from pathlib import Path

import openstudio
import pytest

from geogen.envelope import Envelope
from geogen.geometry import Footprint
from geogen.osm import (
    DEFAULT_BUILDING_NAME,
    ModelError,
    build_model,
    model_file_name,
    model_file_suffix,
    model_name,
    save_model,
)

RING = ((652000.0, 6862000.0), (652000.0, 6862010.0), (652020.0, 6862010.0), (652020.0, 6862000.0))


@pytest.fixture
def footprint() -> Footprint:
    return Footprint(name="bdnb-0001", ring=RING, height=9.0, storeys=3, ground_elevation=35.0)


def test_build_model_creates_one_space_per_storey(footprint: Footprint) -> None:
    model = build_model([footprint])

    spaces = model.getSpaces()
    assert len(spaces) == 3
    assert len(model.getBuildingStorys()) == 3
    assert sorted(space.nameString() for space in spaces) == [
        "bdnb-0001 Storey 1 Space",
        "bdnb-0001 Storey 2 Space",
        "bdnb-0001 Storey 3 Space",
    ]
    assert sorted(space.zOrigin() for space in spaces) == [0.0, 3.0, 6.0]
    for space in spaces:
        assert space.floorArea() == pytest.approx(200.0)
        assert space.volume() == pytest.approx(600.0)


def test_build_model_centers_the_geometry_on_the_site(footprint: Footprint) -> None:
    model = build_model([footprint])

    vertices = [
        (round(point.x(), 3), round(point.y(), 3))
        for surface in model.getSpaces()[0].surfaces()
        if surface.surfaceType() == "Floor"
        for point in surface.vertices()
    ]
    assert sorted(vertices) == [(-10.0, -5.0), (-10.0, 5.0), (10.0, -5.0), (10.0, 5.0)]

    site = model.getSite()
    assert site.latitude() == pytest.approx(48.856, abs=1e-2)
    assert site.longitude() == pytest.approx(2.346, abs=1e-2)
    assert site.elevation() == pytest.approx(35.0)


def test_build_model_matches_the_surfaces_between_storeys(footprint: Footprint) -> None:
    model = build_model([footprint])

    boundaries = sorted(
        surface.outsideBoundaryCondition()
        for space in model.getSpaces()
        for surface in space.surfaces()
    )
    assert boundaries.count("Ground") == 1
    assert boundaries.count("Surface") == 4
    assert boundaries.count("Outdoors") == 13


def test_build_model_stacks_buildings_on_their_own_ground_level() -> None:
    low = Footprint(name="low", ring=RING, height=3.0, storeys=1, ground_elevation=35.0)
    shifted = tuple((x + 100.0, y) for x, y in RING)
    high = Footprint(name="high", ring=shifted, height=3.0, storeys=1, ground_elevation=45.0)

    model = build_model([low, high])

    elevations = {space.nameString(): space.zOrigin() for space in model.getSpaces()}
    assert elevations["low Storey 1 Space"] == pytest.approx(0.0)
    assert elevations["high Storey 1 Space"] == pytest.approx(10.0)
    assert model.getSite().elevation() == pytest.approx(35.0)


def test_build_model_names_the_building(footprint: Footprint) -> None:
    model = build_model([footprint], building_name="1 rue de la Paix")

    assert model.getBuilding().nameString() == "1 rue de la Paix"


def test_build_model_without_footprint() -> None:
    with pytest.raises(ModelError):
        build_model([])


def test_save_model_writes_a_loadable_osm(footprint: Footprint, tmp_path: Path) -> None:
    destination = save_model(build_model([footprint]), tmp_path / "out" / "model.osm")

    assert destination.exists()
    reloaded = openstudio.osversion.VersionTranslator().loadModel(openstudio.path(str(destination)))
    assert reloaded.is_initialized()
    assert len(reloaded.get().getSpaces()) == 3


def test_save_model_writes_a_loadable_idf(footprint: Footprint, tmp_path: Path) -> None:
    destination = save_model(
        build_model([footprint]), tmp_path / "out" / "model.idf", output_format="idf"
    )

    assert destination.exists()
    idf_file = openstudio.IdfFile.load(openstudio.path(str(destination)))
    assert idf_file.is_initialized()


def test_save_model_rejects_an_unknown_output_format(footprint: Footprint, tmp_path: Path) -> None:
    with pytest.raises(ModelError):
        save_model(build_model([footprint]), tmp_path / "model.xyz", output_format="xyz")


def test_build_model_skips_footprints_rejected_by_openstudio(footprint: Footprint) -> None:
    counter_clockwise = Footprint(name="invalid", ring=tuple(reversed(RING)), height=3.0, storeys=1)

    model = build_model([footprint, counter_clockwise])

    assert sorted(space.nameString() for space in model.getSpaces()) == [
        "bdnb-0001 Storey 1 Space",
        "bdnb-0001 Storey 2 Space",
        "bdnb-0001 Storey 3 Space",
    ]
    assert len(model.getBuildingStorys()) == 3


def test_build_model_without_any_usable_footprint() -> None:
    counter_clockwise = Footprint(name="invalid", ring=tuple(reversed(RING)), height=3.0, storeys=1)

    with pytest.raises(ModelError):
        build_model([counter_clockwise])


def test_build_model_ignores_unknown_ground_elevations() -> None:
    known = Footprint(name="known", ring=RING, height=3.0, storeys=1, ground_elevation=300.0)
    shifted = tuple((x + 100.0, y) for x, y in RING)
    unknown = Footprint(name="unknown", ring=shifted, height=3.0, storeys=1)

    model = build_model([known, unknown])

    elevations = {space.nameString(): space.zOrigin() for space in model.getSpaces()}
    assert elevations["known Storey 1 Space"] == pytest.approx(0.0)
    assert elevations["unknown Storey 1 Space"] == pytest.approx(0.0)
    assert model.getSite().elevation() == pytest.approx(300.0)


def test_build_model_covers_a_building_with_a_sloped_roof() -> None:
    sloped = Footprint(
        name="bdnb-0001",
        ring=RING,
        height=3.0,
        storeys=1,
        envelope=Envelope(roof_pitch=30.0),
    )

    model = build_model([sloped])

    attic = next(
        space for space in model.getSpaces() if space.nameString() == "bdnb-0001 Attic Space"
    )
    assert attic.zOrigin() == pytest.approx(3.0)
    assert attic.floorArea() == pytest.approx(200.0)
    # A hip roof of 30 degrees over a 20 m x 10 m footprint is 5 * tan(30) high.
    assert attic.volume() == pytest.approx((200.0 / 2 - 100.0 / 6) * 2.8868, rel=1e-3)
    roofs = [surface for surface in attic.surfaces() if surface.surfaceType() == "RoofCeiling"]
    assert len(roofs) == len(RING)
    assert all(surface.outwardNormal().z() > 0 for surface in roofs)
    assert all(surface.outsideBoundaryCondition() == "Outdoors" for surface in roofs)
    floor = next(surface for surface in attic.surfaces() if surface.surfaceType() == "Floor")
    assert floor.outsideBoundaryCondition() == "Surface"


def test_build_model_caps_the_height_of_the_roofs() -> None:
    sloped = Footprint(
        name="bdnb-0001", ring=RING, height=3.0, storeys=1, envelope=Envelope(roof_pitch=60.0)
    )

    model = build_model([sloped], max_roof_height=2.0)

    attic = next(
        space for space in model.getSpaces() if space.nameString() == "bdnb-0001 Attic Space"
    )
    assert attic.volume() == pytest.approx((200.0 / 2 - 100.0 / 6) * 2.0, rel=1e-3)


def test_build_model_leaves_flat_roofs_alone(footprint: Footprint) -> None:
    model = build_model([footprint])

    assert all("Attic" not in space.nameString() for space in model.getSpaces())
    assert len(model.getSpaces()) == 3


def test_build_model_glazes_the_exterior_walls() -> None:
    glazed = Footprint(
        name="bdnb-0001",
        ring=RING,
        height=6.0,
        storeys=2,
        envelope=Envelope(window_to_wall_ratio=0.25),
    )

    model = build_model([glazed])

    windows = model.getSubSurfaces()
    assert len(windows) == 8
    assert {window.subSurfaceType() for window in windows} == {"FixedWindow"}
    walls = [
        surface
        for space in model.getSpaces()
        for surface in space.surfaces()
        if surface.surfaceType() == "Wall"
    ]
    assert all(surface.windowToWallRatio() == pytest.approx(0.25) for surface in walls)


def test_build_model_glazes_the_known_orientations_only() -> None:
    glazed = Footprint(
        name="bdnb-0001",
        ring=RING,
        height=3.0,
        storeys=1,
        envelope=Envelope(window_to_wall_ratio=0.3, glazing_orientations=("south",)),
    )

    model = build_model([glazed])

    windows = model.getSubSurfaces()
    assert len(windows) == 1
    surface = windows[0].surface().get()
    assert surface.azimuth() == pytest.approx(math.pi)
    assert surface.grossArea() == pytest.approx(60.0)


def test_build_model_does_not_glaze_party_walls() -> None:
    envelope = Envelope(window_to_wall_ratio=0.3)
    left = Footprint(name="left", ring=RING, height=3.0, storeys=1, envelope=envelope)
    adjoining = tuple((x + 20.0, y) for x, y in RING)
    right = Footprint(name="right", ring=adjoining, height=3.0, storeys=1, envelope=envelope)

    model = build_model([left, right])

    assert len(model.getSubSurfaces()) == 6
    glazed = {
        window.surface().get().outsideBoundaryCondition() for window in model.getSubSurfaces()
    }
    assert glazed == {"Outdoors"}


def test_build_model_without_envelope_data_leaves_the_building_as_is(
    footprint: Footprint,
) -> None:
    model = build_model([footprint])

    assert not model.getSubSurfaces()
    assert len(model.getSpaces()) == 3


def test_build_model_skips_a_roof_it_cannot_build() -> None:
    flat = Footprint(
        name="flat",
        ring=((0.0, 0.0), (0.0, 0.5), (2.0, 0.5), (2.0, 0.0)),
        height=3.0,
        storeys=1,
        envelope=Envelope(roof_pitch=30.0),
    )

    model = build_model([flat])

    assert [space.nameString() for space in model.getSpaces()] == ["flat Storey 1 Space"]


def test_model_name_is_the_code_of_the_building() -> None:
    assert model_name(["bdnb-0001"]) == "bdnb-0001"


def test_model_name_of_several_buildings_counts_them() -> None:
    assert model_name(["bdnb-0001", "bdnb-0002", "bdnb-0003"]) == "bdnb-0001 and 2 more"


def test_model_name_ignores_the_buildings_without_a_code() -> None:
    assert model_name(["", "bdnb-0001", "bdnb-0001"]) == "bdnb-0001"
    assert model_name([]) == DEFAULT_BUILDING_NAME


def test_model_file_name_is_the_name_of_the_building() -> None:
    assert model_file_name("bdnb-0001") == "bdnb-0001.osm"


def test_model_file_name_is_a_usable_file_name() -> None:
    assert model_file_name("bdnb-0001 and 2 more") == "bdnb-0001_and_2_more.osm"
    assert model_file_name("1 rue de la Paix, Paris") == "1_rue_de_la_Paix_Paris.osm"
    assert model_file_name("../etc/passwd") == "etc_passwd.osm"
    assert model_file_name("/") == "model.osm"


def test_model_file_suffix_depends_on_the_output_format() -> None:
    assert model_file_suffix("osm") == ".osm"
    assert model_file_suffix("idf") == ".idf"


def test_model_file_suffix_rejects_an_unknown_output_format() -> None:
    with pytest.raises(ModelError):
        model_file_suffix("xyz")
