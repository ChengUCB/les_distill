"""Losslessly embed the final 400 dpi PNGs at exactly 147 x 76 mm.

The accepted figure preparation used a single opaque RGB image per PDF page.
This exporter reproduces that representation without an optional PDF library.
"""
import argparse
import json
from pathlib import Path
import zlib

from PIL import Image

PACKAGE_DIR = Path(__file__).resolve().parents[1]
WIDTH_MM, HEIGHT_MM = 147, 76


def write_rgb_pdf(image, output_path):
    width_pt, height_pt = (v / 25.4 * 72 for v in (WIDTH_MM, HEIGHT_MM))
    pixels = image.tobytes()
    compressed = zlib.compress(pixels)
    content = f"q\n{width_pt:.9f} 0 0 {height_pt:.9f} 0 0 cm\n/Im0 Do\nQ\n".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width_pt:.9f} {height_pt:.9f}] "
         "/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>").encode("ascii"),
        (f"<< /Type /XObject /Subtype /Image /Width {image.width} /Height {image.height} "
         f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length {len(compressed)} >>\nstream\n").encode("ascii")
        + compressed + b"\nendstream",
        f"<< /Length {len(content)} >>\nstream\n".encode("ascii") + content + b"endstream",
    ]
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, body in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii") + body + b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend((f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                   f"startxref\n{xref_offset}\n%%EOF\n").encode("ascii"))
    output_path.write_bytes(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PACKAGE_DIR / "reproduced")
    parser.add_argument("--output-dir", type=Path,
                        help="PDF directory; defaults to --input-dir.")
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("energy", "force"):
        stem = f"pbe_teacher_{kind}_parity"
        metadata = json.loads((args.input_dir / f"{stem}_metrics.json").read_text())
        if metadata["font_size_pt"] != 11 or metadata["figure_size_mm"] != [WIDTH_MM, HEIGHT_MM]:
            raise ValueError(f"Unexpected figure dimensions or font size: {stem}")
        with Image.open(args.input_dir / f"{stem}.png") as original:
            image = original.convert("RGB")
        if image.size != (2314, 1196):
            raise ValueError(f"Unexpected PNG dimensions: {image.size}")
        output_path = output_dir / f"{stem}.pdf"
        write_rgb_pdf(image, output_path)
        print(f"Created {output_path}: 147 x 76 mm, opaque RGB image")


if __name__ == "__main__":
    main()
