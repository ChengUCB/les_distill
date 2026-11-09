#!/usr/bin/env python3
"""Check archived hashes and every plotted curve without raw data or figure writes."""
from pathlib import Path
import csv
from contextlib import redirect_stdout
import io
import hashlib
import importlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def table_pairs(filename):
    table = np.loadtxt(DATA / filename, ndmin=2)
    return [(table[:, j], table[:, j + 1]) for j in range(0, table.shape[1], 2)]

def expected_fig5():
    with (DATA / "all_caceles_student_water_orientation.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    keys = sorted(set(row["model_key"] for row in rows), key=lambda key: int(next(row["model_order"] for row in rows if row["model_key"] == key)))
    result = []
    for key in keys:
        selected = [row for row in rows if row["model_key"] == key]
        result.append((np.asarray([float(row["distance_A"]) for row in selected]), np.asarray([float(row["mean_cos_theta"]) if row["mean_cos_theta"] else np.nan for row in selected])))
    return result

def main():
    provenance = json.loads((ROOT / "provenance/sources.json").read_text())
    for entry in provenance["files"]:
        path = ROOT / entry["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise AssertionError(f"Source payload changed: {entry['file']}")
    expected = {
        "plot_fig5": expected_fig5(),
        "plot_fig6": table_pairs("foundation_mid_vs_caceles_large_density.dat"),
        "plot_fig7": table_pairs("foundation_mid_vs_caceles_large_cos_theta.dat"),
        "plot_s12": table_pairs("old_new_large_water_density.dat"),
        "plot_s13": table_pairs("caceles_mid_vs_large_density.dat") + table_pairs("caceles_mid_vs_large_cos_theta.dat"),
    }
    original_savefig, original_close = Figure.savefig, plt.close
    Figure.savefig = lambda *args, **kwargs: None
    plt.close = lambda *args, **kwargs: None
    report = {}
    try:
        for name, curves in expected.items():
            before = set(plt.get_fignums())
            module = importlib.import_module(name)
            module.OUTPUT_PDF = ROOT / "reproduced" / f"{name}.pdf"
            module.OUTPUT_PNG = ROOT / "reproduced" / f"{name}.png"
            with redirect_stdout(io.StringIO()):
                if name == "plot_fig5":
                    module.make_figure(module.load_orientation_csv())
                else:
                    module.make_figure()
            figures = set(plt.get_fignums()) - before
            if len(figures) != 1:
                raise AssertionError(f"Expected one figure: {name}")
            figure = plt.figure(figures.pop())
            plotted = [(np.asarray(line.get_xdata()), np.asarray(line.get_ydata())) for ax in figure.axes for line in ax.lines if len(line.get_xdata()) > 2]
            if len(plotted) != len(curves):
                raise AssertionError(f"Curve count differs: {name}")
            for actual, reference in zip(plotted, curves):
                # Equal includes the native grids, NaN masks, and unchanged ordinates.
                np.testing.assert_array_equal(actual[0], reference[0])
                np.testing.assert_array_equal(actual[1], reference[1])
            report[name] = {"curves": len(curves), "all_x_y_arrays_equal": True}
            original_close(figure)
    finally:
        Figure.savefig, plt.close = original_savefig, original_close
    print(json.dumps({"status": "passed", "source_hashes_checked": len(provenance["files"]), "total_curves_checked": sum(x["curves"] for x in report.values()), "figures": report, "numpy": np.__version__, "matplotlib": matplotlib.__version__}, indent=2))

if __name__ == "__main__":
    main()
