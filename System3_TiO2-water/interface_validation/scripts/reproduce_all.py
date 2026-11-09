#!/usr/bin/env python3
"""Run all five final-curve redraws without raw trajectories or caches."""
from pathlib import Path
import subprocess
import sys
from plot_support import output_directory

def main():
    directory = output_directory()
    scripts = Path(__file__).resolve().parent
    for name in ["plot_fig5.py", "plot_fig6.py", "plot_fig7.py", "plot_s12.py", "plot_s13.py"]:
        subprocess.run([sys.executable, str(scripts / name), "--output-dir", str(directory)], check=True)

if __name__ == "__main__":
    main()
