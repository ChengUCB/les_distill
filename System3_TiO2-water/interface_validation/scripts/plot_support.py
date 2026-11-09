"""Small-table inputs for accepted-figure redraws; no trajectory processing."""
from pathlib import Path
import argparse
import csv
import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_DIR / "data"

def output_directory():
    parser = argparse.ArgumentParser(description="Redraw from the archived final curves.")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR / "reproduced")
    result = parser.parse_args().output_dir.resolve()
    result.mkdir(parents=True, exist_ok=True)
    return result

def load_named_table(path):
    headers = [line[1:].strip() for line in path.read_text().splitlines() if line.startswith("#")]
    labels = headers[-1].split()
    values = np.loadtxt(path, ndmin=2)
    if values.shape[1] != len(labels):
        raise ValueError(f"Column-header mismatch: {path}")
    return {label: values[:, index] for index, label in enumerate(labels)}

def load_orientation_csv():
    results = {}
    with (DATA_DIR / "all_caceles_student_water_orientation.csv").open() as handle:
        for row in csv.DictReader(handle):
            result = results.setdefault(row["model_key"], {"z": [], "orientation": []})
            result["z"].append(float(row["distance_A"]))
            result["orientation"].append(float(row["mean_cos_theta"]) if row["mean_cos_theta"] else np.nan)
    return {key: {name: np.asarray(values) for name, values in result.items()} for key, result in results.items()}

def load_paired_profiles(filename, model_groups, stems, observable):
    table = load_named_table(DATA_DIR / filename)
    results = {}
    for title, _, foundation, student, _, _ in model_groups:
        stem = stems[title]
        for key, suffix in [(foundation, "mid_foundation"), (student, "large_CACELES")]:
            z = table[f"{stem}_{suffix}_z_A"]
            value = table[f"{stem}_{suffix}_{observable}"]
            # Preserve NaN bins: filtering would bridge an excluded interval.
            results[key] = (z, value) if observable == "density_g_mL" else (z, value, None, None)
    return results
