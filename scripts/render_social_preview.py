"""Render the documentation's share card from its existing scientific illustrations.

Run with Pillow, Matplotlib (for its bundled fonts), and resvg-py installed:
    python scripts/render_social_preview.py

The surfaces come from render_landing_illustration.py, which evaluates real
SplineOps operators. This script lays them out with the existing vector logo.
The checked-in PNG needs no renderer or extra dependencies during a docs build.
"""

import base64
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET

from matplotlib import get_data_path
from PIL import Image
import resvg_py

STATIC = Path(__file__).resolve().parents[1] / "docs" / "_static"


def embedded_panel(name):
    """Use PNG inside the SVG so raster decoding is portable across renderers."""
    buffer = BytesIO()
    with Image.open(STATIC / "landing" / f"{name}.webp") as panel:
        panel.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def main():
    logo = ET.parse(STATIC / "logo.svg").getroot()
    logo.set("x", "52")
    logo.set("y", "34")
    logo.set("width", "76")
    logo.set("height", "76")
    panels = []
    for index, (name, label, color) in enumerate(
        [
            ("reconstruct", "Reconstruct", "#71dfcf"),
            ("resample", "Resample", "#f3d698"),
            ("differentiate", "Differentiate", "#d9a3e5"),
        ]
    ):
        x = 34 + index * 386
        panels.append(
            f'<image x="{x}" y="242" width="360" height="245" '
            f'href="{embedded_panel(name)}"/>'
            f'<text x="{x + 22}" y="528" fill="{color}" font-size="17">'
            f"0{index + 1}</text>"
            f'<text x="{x + 59}" y="528" fill="#f1f5fa" font-size="25" '
            f'font-weight="bold">{label}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630"
                   viewBox="0 0 1200 630">
      <rect width="1200" height="630" fill="#0c1729"/>
      {ET.tostring(logo, encoding="unicode")}
      <g font-family="DejaVu Sans">
        <text x="145" y="91" fill="#f4f7fb" font-size="51" font-weight="bold">SplineOps</text>
        <text x="1144" y="80" text-anchor="end" fill="#9eb6cc" font-size="17"
              letter-spacing="1.5">SCIENTIFIC COMPUTING IN PYTHON</text>
        <text x="56" y="183" fill="#f4f7fb" font-size="43" font-weight="bold"
              letter-spacing="-1">From samples to continuous operations.</text>
        <text x="56" y="225" fill="#a9bed3" font-size="23">Interpolation · Antialiased resizing · Derivatives</text>
        {''.join(panels)}
        <path d="M56 571H1144" stroke="#29405b"/>
        <text x="56" y="609" fill="#80d6e4" font-size="20">splineops.github.io</text>
        <text x="1144" y="609" text-anchor="end" fill="#a9bed3" font-size="18">Open source. Built for N-D data.</text>
      </g>
    </svg>"""
    fonts = Path(get_data_path()) / "fonts" / "ttf"
    png = resvg_py.svg_to_bytes(
        svg_string=svg,
        skip_system_fonts=True,
        font_files=[
            str(fonts / name) for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
        ],
    )
    destination = STATIC / "social" / "splineops-preview.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(png)
    print(f"Rendered {destination} ({len(png):,} bytes)")


if __name__ == "__main__":
    main()
