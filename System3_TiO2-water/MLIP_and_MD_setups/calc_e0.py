"""Compute per-element reference energies (E0s) by linear regression over the training set.

The resulting E0 values are printed and saved to a pickle file for use in CACELES-train.py.

Example:
    python calc_e0.py --train-path train-TiO2-water-DFT.xyz
"""

import argparse
import logging
import os
import pickle

import ase.io
import cace

try:
    from cace.tools import compute_average_E0s
except ImportError:
    from cace.tools.utils import compute_average_E0s

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute average atomic reference energies (E0s) from a training set."
    )
    parser.add_argument("--train-path", default="train-TiO2-water-DFT.xyz",
                        help="Path to the training XYZ file.")
    parser.add_argument("--output", default="avge0.pkl",
                        help="Output pickle file for the E0 dictionary.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.train_path):
        logging.error(f"File not found: {args.train_path}")
        return

    logging.info(f"Reading {args.train_path}...")
    atoms_list = ase.io.read(args.train_path, index=':')
    logging.info(f"Read {len(atoms_list)} frames.")

    logging.info("Computing average E0s by linear regression...")
    e0_dict = compute_average_E0s(atoms_list)

    logging.info("-" * 40)
    logging.info(f"E0 results: {e0_dict}")
    logging.info("-" * 40)

    with open(args.output, 'wb') as f:
        pickle.dump(e0_dict, f)
    logging.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
