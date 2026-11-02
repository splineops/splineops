"""Render the share card with the historical SIAM Cover Page artwork.

Run with Pillow, Matplotlib (for its bundled fonts), and resvg-py installed:
    python scripts/render_social_preview.py

The existing feature_01.jpg is Annette Unser's SIAM Cover Page (1999), listed at
https://bigwww.epfl.ch/art/splines/index.html. Preserve its complete composition
and credit while laying it out with the vector logo. The checked-in JPEG needs
no renderer or extra dependencies during a docs build. The previous PNG remains
available so cached cards using its old URL do not lose their image.
"""

import base64
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET

from matplotlib import get_data_path
from PIL import Image
import resvg_py

STATIC = Path(__file__).resolve().parents[1] / "docs" / "_static"


def main():
    logo = ET.parse(STATIC / "logo.svg").getroot()
    logo.set("x", "52")
    logo.set("y", "44")
    logo.set("width", "76")
    logo.set("height", "76")
    artwork = base64.b64encode((STATIC / "feature_01.jpg").read_bytes()).decode()
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630"
                   viewBox="0 0 1200 630">
      <rect width="1200" height="630" fill="#0c1729"/>
      {ET.tostring(logo, encoding="unicode")}
      <g font-family="DejaVu Sans">
        <text x="145" y="101" fill="#f4f7fb" font-size="51" font-weight="bold">SplineOps</text>
        <text x="56" y="176" fill="#80d6e4" font-size="17"
              letter-spacing="1.2">SAMPLED DATA. CONTINUOUS MODELS.</text>
        <text x="56" y="267" fill="#f4f7fb" font-size="41" font-weight="bold"
              letter-spacing="-1">From samples to</text>
        <text x="56" y="322" fill="#f4f7fb" font-size="41" font-weight="bold"
              letter-spacing="-1">continuous models.</text>
        <text x="56" y="386" fill="#a9bed3" font-size="23">Interpolate. Resize with antialiasing.</text>
        <text x="56" y="421" fill="#a9bed3" font-size="23">Transform. Differentiate. In N-D.</text>
        <path d="M56 521H536" stroke="#29405b"/>
        <text x="56" y="562" fill="#80d6e4" font-size="22">splineops.github.io</text>
        <image x="600" y="44" width="550" height="508" preserveAspectRatio="xMidYMid meet"
               href="data:image/jpeg;base64,{artwork}"/>
        <text x="875" y="589" text-anchor="middle" fill="#a9bed3" font-size="16">SIAM Cover Page · Annette Unser, 1999</text>
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
    destination = STATIC / "social" / "splineops-spline-art.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    # JPEG suits the watercolor texture; full chroma keeps text and curves crisp.
    with Image.open(BytesIO(png)) as card:
        card.convert("RGB").save(destination, quality=93, subsampling=0, optimize=True)
    print(f"Rendered {destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
