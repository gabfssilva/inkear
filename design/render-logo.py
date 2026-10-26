# /// script
# requires-python = ">=3.13"
# dependencies = ["fonttools>=4.58", "resvg-py>=0.5"]
# ///
"""Renders the inkear mark, the app icon and the launch screen logo.

The mark is the closing double quotation mark of Newsreader, the app's text face: the spoken
text, and two ink blots shaped like ears. Run from the repository root with `uv run`.
"""

from pathlib import Path

import resvg_py
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

ROOT = Path(__file__).resolve().parent.parent
FONT = ROOT / "inkear/Resources/Fonts/Newsreader.ttf"
ASSETS = ROOT / "inkear/Resources/Assets.xcassets"

PAPER = "#F4F1EA"
INK = "#2F342B"
SAGE = "#A3B18A"

WEIGHT = 700
OPTICAL_SIZE = 24


def mark_path() -> tuple[str, tuple[float, float, float, float]]:
    """The glyph outline as an SVG path in font units, y down, and its bounds."""
    font = instantiateVariableFont(TTFont(FONT), {"wght": WEIGHT, "opsz": OPTICAL_SIZE})
    glyph = font.getGlyphSet()["quotedblright"]
    pen = SVGPathPen(font.getGlyphSet())
    glyph.draw(pen)
    bounds = BoundsPen(font.getGlyphSet())
    glyph.draw(bounds)
    x_min, y_min, x_max, y_max = bounds.bounds
    return pen.getCommands(), (x_min, -y_max, x_max, -y_min)


def composition(size: float, mark_width: float, fill: str, background: str | None) -> str:
    """A square of `size` with the mark `mark_width` wide centred on it."""
    path, (x0, y0, x1, y1) = mark_path()
    scale = mark_width / (x1 - x0)
    width, height = (x1 - x0) * scale, (y1 - y0) * scale
    # Quotation marks carry their weight at the top, so the mark sits a touch above centre.
    x = (size - width) / 2 - x0 * scale
    y = (size - height) / 2 - y0 * scale - size * 0.015
    rect = f'<rect width="{size}" height="{size}" fill="{background}"/>' if background else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}">'
        f"{rect}"
        f'<path transform="translate({x:.3f} {y:.3f}) scale({scale:.6f} {-scale:.6f})" '
        f'fill="{fill}" d="{path}"/></svg>'
    )


def mark_svg(fill: str) -> str:
    """The mark alone, cropped to its bounds, 1000 units wide."""
    path, (x0, y0, x1, y1) = mark_path()
    scale = 1000 / (x1 - x0)
    width, height = 1000, (y1 - y0) * scale
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}">'
        f'<path transform="translate({-x0 * scale:.3f} {-y0 * scale:.3f}) '
        f'scale({scale:.6f} {-scale:.6f})" fill="{fill}" d="{path}"/></svg>'
    )


def render(svg: str, destination: Path, pixels: int) -> None:
    destination.write_bytes(resvg_py.svg_to_bytes(svg_string=svg, width=pixels, height=pixels))
    print(destination.relative_to(ROOT))


def main() -> None:
    (ROOT / "design/logo.svg").write_text(mark_svg(INK))

    icons = ASSETS / "AppIcon.appiconset"
    render(composition(1024, 560, INK, PAPER), icons / "AppIcon-light.png", 1024)
    render(composition(1024, 560, SAGE, INK), icons / "AppIcon-dark.png", 1024)
    # The tinted icon is grayscale on a transparent ground; iOS supplies the background.
    render(composition(1024, 560, "#FFFFFF", None), icons / "AppIcon-tinted.png", 1024)

    launch = ASSETS / "LaunchLogo.imageset"
    launch.mkdir(exist_ok=True)
    points = 96
    for scale in (1, 2, 3):
        pixels = points * scale
        light, dark = (
            composition(points, points * 0.78, INK, None),
            composition(points, points * 0.78, SAGE, None),
        )
        render(light, launch / f"LaunchLogo-light@{scale}x.png", pixels)
        render(dark, launch / f"LaunchLogo-dark@{scale}x.png", pixels)


if __name__ == "__main__":
    main()
