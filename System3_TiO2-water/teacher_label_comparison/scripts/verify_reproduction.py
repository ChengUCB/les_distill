"""Check regenerated S9/S10 numerical results against the preserved references."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PACKAGE_DIR = Path(__file__).resolve().parents[1]


def check_value(actual, expected, path=""):
    if isinstance(expected, dict):
        for key, value in expected.items():
            # The original metadata described a vector PDF, but the final PDFs
            # were opaque PNG embeds. This description was corrected here.
            if key != "rendering":
                check_value(actual[key], value, f"{path}/{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"Length mismatch at {path}")
        for index, (a, e) in enumerate(zip(actual, expected)):
            check_value(a, e, f"{path}/{index}")
    elif isinstance(expected, (float, int)):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10,
                                   err_msg=path)
    elif actual != expected:
        raise AssertionError(f"Mismatch at {path}: {actual!r} != {expected!r}")


def check_checksums():
    checked = 0
    for line in (PACKAGE_DIR / "checksums.sha256").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        checksum = hashlib.sha256()
        with (PACKAGE_DIR / relative).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
        if checksum.hexdigest() != digest:
            raise AssertionError(f"Checksum mismatch: {relative}; check Git LFS downloads.")
        checked += 1
    return checked


def check_geometry():
    data_dir = PACKAGE_DIR / "data"
    with np.load(data_dir / "cache_SCAN.npz", allow_pickle=False) as reference:
        counts = reference["n"]
        frames = np.flatnonzero(counts == 426)
        if len(frames) != 3168:
            raise AssertionError("Expected 3168 full-interface frames")
        offsets = reference["offsets"]
        positions, numbers = reference["pos"], reference["z"]
        ref_positions = np.concatenate([positions[offsets[i]:offsets[i + 1]] for i in frames])
        ref_numbers = np.concatenate([numbers[offsets[i]:offsets[i + 1]] for i in frames])
        ref_cells, ref_pbc = reference["cell"][frames], reference["pbc"][frames]
    result = {"canonical_frames": len(counts), "selected_frames": len(frames), "models": {}}
    with np.load(data_dir / "pairing.npz", allow_pickle=False) as pairing:
        for name in ("GemNet", "MACE_MH", "MACE_MP"):
            order = pairing[name]
            np.testing.assert_array_equal(np.sort(order), np.arange(len(counts)))
            selected = order[frames]
            with np.load(data_dir / f"cache_{name}.npz", allow_pickle=False) as cache:
                np.testing.assert_array_equal(cache["n"][selected], counts[frames])
                offsets = cache["offsets"]
                positions, numbers = cache["pos"], cache["z"]
                pos = np.concatenate([positions[offsets[i]:offsets[i + 1]] for i in selected])
                z = np.concatenate([numbers[offsets[i]:offsets[i + 1]] for i in selected])
                np.testing.assert_array_equal(z, ref_numbers)
                np.testing.assert_allclose(pos, ref_positions, rtol=0, atol=1e-8)
                np.testing.assert_allclose(cache["cell"][selected], ref_cells, rtol=0, atol=1e-8)
                np.testing.assert_array_equal(cache["pbc"][selected], ref_pbc)
                result["models"][name] = {
                    "unique_paired_frames": len(np.unique(selected)),
                    "species_and_pbc_match": True,
                    "max_position_difference_A": float(np.max(np.abs(pos - ref_positions))),
                    "max_cell_difference_A": float(np.max(np.abs(cache["cell"][selected] - ref_cells))),
                }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR / "reproduced")
    parser.add_argument("--require-pixel-equality", action="store_true",
                        help="Fail if PNG pixels differ; exact rendering may depend on library versions.")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Check package checksums and geometry mapping before running the plots.")
    args = parser.parse_args()
    reference = PACKAGE_DIR / "reference"
    result = {"checksummed_files": check_checksums(), "geometry": check_geometry(),
              "figures": {}, "numerical_tolerance": {"rtol": 1e-12, "atol": 1e-10}}
    if args.skip_figures:
        print(json.dumps(result, indent=2))
        return
    for kind in ("energy", "force"):
        stem = f"pbe_teacher_{kind}_parity"
        actual = json.loads((args.output_dir / f"{stem}_metrics.json").read_text())
        expected = json.loads((reference / f"{stem}_metrics.json").read_text())
        check_value(actual, expected)
        with Image.open(args.output_dir / f"{stem}.png") as im:
            image = np.array(im.convert("RGB"))
        with Image.open(reference / f"{stem}.png") as im:
            original = np.array(im.convert("RGB"))
        pixels_equal = np.array_equal(image, original)
        if args.require_pixel_equality and not pixels_equal:
            raise AssertionError(f"PNG pixels differ: {stem}")
        result["figures"][kind] = {"metrics_match": True, "png_pixels_equal": pixels_equal,
                                   "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest()}
    actual_table = pd.read_csv(args.output_dir / "pbe_teacher_energy_parity_data.csv")
    expected_table = pd.read_csv(reference / "pbe_teacher_energy_parity_data.csv")
    if list(actual_table.columns) != list(expected_table.columns):
        raise AssertionError("Energy CSV column mismatch")
    np.testing.assert_allclose(actual_table.to_numpy(), expected_table.to_numpy(), rtol=1e-12, atol=1e-10)
    result["energy_csv_matches"] = True
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
