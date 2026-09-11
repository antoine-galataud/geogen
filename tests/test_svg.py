from geogen.envelope import Envelope
from geogen.geometry import Footprint
from geogen.svg import save_svg


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
