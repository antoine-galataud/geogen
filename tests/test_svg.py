from shapely.geometry import Point

from geogen.envelope import Envelope
from geogen.geometry import Footprint
from geogen.svg import (
    _build_scene,
    _projected_face,
    _Projection,
    _visible_regions,
    save_svg,
)


def test_save_svg_creates_vector_preview(tmp_path):
    footprint = Footprint(
        name="test-building",
        ring=((0.0, 0.0), (0.0, 10.0), (20.0, 10.0), (20.0, 0.0)),
        height=12.0,
        storeys=4,
        envelope=Envelope(
            window_to_wall_ratio=0.25,
            glazing_orientations=(),
            roof_pitch=30.0,
        ),
    )

    destination = save_svg([footprint], path=tmp_path / "building.svg")

    content = destination.read_text(encoding="utf-8")
    assert destination.exists()
    assert "<svg" in content
    assert 'class="wall"' in content
    assert 'class="window"' in content
    assert "pitched-roof" in content


def test_foreground_roof_is_drawn_after_hidden_rear_facade():
    rear = Footprint("rear-tall", ((0.0, -10.0), (0.0, 0.0), (10.0, 0.0), (10.0, -10.0)), 20.0, 5)
    front = Footprint("front-low", ((0.0, 1.0), (0.0, 10.0), (10.0, 10.0), (10.0, 1.0)), 3.0, 1)
    projection = _Projection(origin=(0.0, 0.0), azimuth=0.0, elevation=45.0)

    drawables = _build_scene([rear, front], projection, max_roof_height=3.0)
    rear_wall = next(
        drawable
        for drawable in drawables
        if drawable.primitives[0][2] == "wall"
        and {point[1] for point in drawable.primitives[0][1]} == {0.0}
        and max(point[2] for point in drawable.primitives[0][1]) == 20.0
    )
    front_roof = next(
        drawable
        for drawable in drawables
        if drawable.primitives[0][2] == "roof flat-roof"
        and {point[2] for point in drawable.primitives[0][1]} == {3.0}
    )

    # Mean depth gives the opposite order, but the roof is nearer everywhere
    # that these two faces overlap and must therefore be painted last.
    assert front_roof.depth < rear_wall.depth
    assert drawables.index(rear_wall) < drawables.index(front_roof)

    overlap_point = Point(projection.project((5.0, 0.0, 1.0)))
    assert _projected_face(rear_wall, projection).polygon.covers(overlap_point)
    assert _projected_face(front_roof, projection).polygon.covers(overlap_point)

    visible = _visible_regions(drawables, projection)
    assert not visible[drawables.index(rear_wall)].covers(overlap_point)
    assert visible[drawables.index(front_roof)].covers(overlap_point)


def _camera_face(points, css_class):
    from geogen.svg import _Drawable

    projection = _Projection((0.0, 0.0), 0.0, 0.0)
    world = [(x, depth, y) for x, y, depth in points]
    drawable = _Drawable(projection.depth(world))
    drawable.add_polygon(world, css_class)
    return drawable


def test_touching_depths_still_constrain_painter_order():
    from geogen.svg import _sort_drawables

    wall = _camera_face([(0, 0, 20), (1, 0, 20), (1, 100, 120), (0, 100, 120)], "wall")
    roof = _camera_face([(0, 0, 21), (1, 0, 21), (1, 1, 21), (0, 1, 21)], "roof")
    projection = _Projection((0, 0), 0, 0)
    assert wall.depth > roof.depth
    assert _sort_drawables([roof, wall], projection) == [wall, roof]


def test_crossing_depths_have_complementary_visible_regions():
    from shapely.geometry import Polygon

    first = _camera_face([(0, 0, 0), (2, 0, 2), (2, 2, 2), (0, 2, 0)], "wall")
    second = _camera_face([(0, 0, 2), (2, 0, 0), (2, 2, 0), (0, 2, 2)], "roof")
    projection = _Projection((0, 0), 0, 0)
    for drawables in ([first, second], [second, first]):
        visible = _visible_regions(drawables, projection)
        assert visible[0].intersection(visible[1]).area < 1e-12
        assert visible[0].union(visible[1]).equals(Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]))
        for drawable, region in zip(drawables, visible):
            assert region.covers(Point(1.5 if drawable is first else 0.5, 1))


def test_svg_bakes_occlusion_into_fills_and_decorations(tmp_path, monkeypatch):
    import re
    import xml.etree.ElementTree as ET

    from shapely.geometry import LineString, Polygon

    import geogen.svg as svg

    wall = _camera_face([(0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0)], "wall")
    wall.add_polygon([(0.5, 0, 0.5), (3.5, 0, 0.5), (3.5, 0, 3.5), (0.5, 0, 3.5)], "window")
    wall.add_polyline([(0, 0, 2), (4, 0, 2)], "storey-line")
    roof = _camera_face([(1, 1, 1), (3, 1, 1), (3, 3, 1), (1, 3, 1)], "roof")
    # Deliberately put the facade last and give its visible region a hole.
    monkeypatch.setattr(svg, "_build_scene", lambda *args: [roof, wall])
    footprint = Footprint("test", ((0, 0), (0, 4), (4, 4), (4, 0)), 4, 1)
    path = save_svg(
        [footprint],
        tmp_path / "occlusion.svg",
        width=400,
        height=400,
        margin=0,
        azimuth=0,
        elevation=0,
    )
    root = ET.parse(path).getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}
    assert not root.findall(".//svg:clipPath", ns)
    hidden = Polygon([(100, 100), (300, 100), (300, 300), (100, 300)])
    for element in root.findall(".//svg:path", ns):
        if element.get("class") == "roof":
            continue
        parts = []
        for part in element.get("d").split("M ")[1:]:
            coords = [
                tuple(map(float, pair)) for pair in re.findall(r"(-?[\d.]+),(-?[\d.]+)", part)
            ]
            parts.append(Polygon(coords) if "Z" in part else LineString(coords))
        if element.get("style") == "stroke: none":
            geometry = Polygon()
            for part in parts:
                geometry = geometry.symmetric_difference(part)
            assert geometry.intersection(hidden).area < 1e-10
            assert geometry.area > 0
        else:
            assert all(part.intersection(hidden.buffer(-1e-6)).is_empty for part in parts)
