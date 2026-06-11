"""Compute layer-resolved BEC-derived currents for a TiO2-water trajectory segment.

Example:
    python BEC_part1.py \\
        --model-path best_model_DFT_SCAN.pth \\
        --traj-path md_TiO2-water.traj \\
        --start-frame 0 \\
        --end-frame 50000
"""

import argparse
import os
import pickle
import sys
from itertools import islice
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

import cace
from ase.io.trajectory import Trajectory
from cace.data import AtomicData
from cace.tools import torch_geometric, torch_tools

BATCH_SIZE = 2
DUMP_CHUNK_SIZE = 2000

# Layer boundaries in Angstroms (distance from surface)
LAYER_CUTS = [1.725, 3.725, 6.025, 35.0]
LAYER_NAMES = {0: "1st", 1: "2nd", 2: "3rd", 3: "Bulk"}

# Physical constants for polarization
POLARIZATION_NORM_FACTOR = 1.0 / 9.48933 * 1.333

# Number of lattice oxygen atoms in the TiO2 slab
EXPECTED_LATTICE_O_COUNT = 1080

# z-coordinates of the two TiO2-water interfaces
Z_INTERFACE_BOTTOM = 8.0824
Z_INTERFACE_TOP = 75.2569
Z_CENTER = (Z_INTERFACE_BOTTOM + Z_INTERFACE_TOP) / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute layer-resolved BEC-derived currents for a trajectory segment."
    )
    parser.add_argument("--model-path", required=True, type=str,
                        help="Path to the trained CACE model (.pth).")
    parser.add_argument("--traj-path", default="md_TiO2-water.traj", type=str,
                        help="Path to the ASE trajectory file.")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Start frame index (inclusive).")
    parser.add_argument("--end-frame", type=int, default=50000,
                        help="End frame index (exclusive).")
    parser.add_argument("--output-dir", default="currents_output", type=str,
                        help="Directory for output pickle files.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Torch device (cuda or cpu).")
    return parser.parse_args()


def get_edge_vectors_and_lengths(positions, edge_index, shifts, normalize=False, eps=1e-9):
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths
    return vectors, lengths


def recompute_graph_for_gradients(data):
    if hasattr(data, 'positions') and data.positions is not None:
        positions = data.positions
    else:
        positions = data.pos

    if not positions.requires_grad:
        positions.requires_grad_(True)

    if hasattr(data, 'shifts') and data.shifts is not None:
        shifts = data.shifts
    else:
        shifts = torch.zeros((data.edge_index.shape[1], 3),
                             device=positions.device, dtype=positions.dtype)

    vectors, lengths = get_edge_vectors_and_lengths(
        positions=positions, edge_index=data.edge_index, shifts=shifts
    )

    data.vectors = vectors
    data.edge_attr = vectors
    data.lengths = lengths
    data.pos = positions
    data.positions = positions
    return data


def identify_atomic_roles(atoms):
    """Identify water, lattice-O, and H indices from the first trajectory frame."""
    pos = atoms.positions
    nums = atoms.get_atomic_numbers()

    is_O = (nums == 8)
    o_indices = np.where(is_O)[0]
    dist_from_center = np.abs(pos[o_indices, 2] - Z_CENTER)
    lattice_o_indices = o_indices[np.argsort(dist_from_center)[-EXPECTED_LATTICE_O_COUNT:]]

    is_slab = np.zeros(len(atoms), dtype=bool)
    is_slab[lattice_o_indices] = True
    is_slab[nums == 22] = True  # Ti atoms

    is_nacl = (nums == 11) | (nums == 17)

    water_indices = np.where(~is_slab & ~is_nacl)[0]
    w_nums = nums[water_indices]
    idx_O_in_water = np.where(w_nums == 8)[0]
    idx_H_in_water = np.where(w_nums == 1)[0]

    return water_indices, idx_O_in_water, idx_H_in_water


def compute_layer_currents(atoms, bec_all, water_indices, idx_O_in_water, idx_H_in_water):
    """Compute layer-resolved dipole currents using molecule-based grouping."""
    pos = atoms.positions
    vel = atoms.get_velocities()

    w_pos = pos[water_indices]
    w_vel = vel[water_indices]
    w_bec = bec_all[water_indices]

    j_water_atomic = np.einsum('iab,ib->ia', w_bec, w_vel)
    j_total_water = np.sum(j_water_atomic, axis=0)

    # Assign O atoms to layers based on distance from nearest surface
    o_pos = w_pos[idx_O_in_water]
    o_z = o_pos[:, 2]
    o_dists = np.where(o_z < Z_CENTER, o_z - Z_INTERFACE_BOTTOM, Z_INTERFACE_TOP - o_z)
    o_labels = np.digitize(o_dists, LAYER_CUTS)

    # Assign H atoms to the same layer as their nearest O (keeps molecules intact)
    tree = cKDTree(o_pos)
    dist_h_to_o, nearest_o_idx = tree.query(w_pos[idx_H_in_water], distance_upper_bound=1.3)

    final_w_labels = np.zeros(len(water_indices), dtype=np.int8)
    final_w_labels[idx_O_in_water] = o_labels

    h_labels = np.zeros(len(idx_H_in_water), dtype=np.int8)
    bound_mask = (dist_h_to_o < 1.3)
    h_labels[bound_mask] = o_labels[nearest_o_idx[bound_mask]]

    if np.any(~bound_mask):
        free_h_pos = w_pos[idx_H_in_water[~bound_mask]]
        free_h_z = free_h_pos[:, 2]
        free_h_dists = np.where(free_h_z < Z_CENTER,
                                free_h_z - Z_INTERFACE_BOTTOM,
                                Z_INTERFACE_TOP - free_h_z)
        h_labels[~bound_mask] = np.digitize(free_h_dists, LAYER_CUTS)

    final_w_labels[idx_H_in_water] = h_labels

    results_J = {'Total': j_total_water}
    results_N = {'Total': len(water_indices)}
    for idx, name in LAYER_NAMES.items():
        mask = (final_w_labels == idx)
        if np.any(mask):
            results_J[name] = np.sum(j_water_atomic[mask], axis=0)
            results_N[name] = int(np.sum(mask))
        else:
            results_J[name] = np.zeros(3)
            results_N[name] = 0

    return results_J, results_N, w_pos, w_bec, final_w_labels


def save_chunk(J_buffer, N_buffer, Pos_buffer, BEC_buffer, Label_buffer,
               start_idx, output_dir):
    if not J_buffer['Total']:
        return
    os.makedirs(output_dir, exist_ok=True)

    n_frames = len(J_buffer['Total'])
    end_idx = start_idx + n_frames
    curr_file = os.path.join(output_dir, f"currents_{start_idx:06d}_{end_idx:06d}.pkl")
    with open(curr_file, 'wb') as f:
        pickle.dump({'J': J_buffer, 'N': N_buffer, 'CUTS': LAYER_CUTS}, f)
    print(f"Saved chunk: frames {start_idx}-{end_idx}")


def build_model(model_path, device):
    loaded_obj = torch.load(model_path, map_location=device, weights_only=False)

    if hasattr(loaded_obj, 'models'):
        cace_representation = loaded_obj.models[0].representation
        q_module = None
        for submodel in loaded_obj.models:
            for mod in submodel.output_modules:
                if hasattr(mod, 'per_atom_output_key') and mod.per_atom_output_key == 'q':
                    q_module = mod
                    break
            if q_module:
                break
    else:
        cace_representation = loaded_obj.representation
        q_module = None
        for mod in loaded_obj.output_modules:
            if hasattr(mod, 'per_atom_output_key') and mod.per_atom_output_key == 'q':
                q_module = mod
                break

    if q_module is None:
        print("FATAL ERROR: Could not find charge output module with key 'q'.")
        sys.exit(1)

    q_module.feature_key = 'node_feats'
    q_module.global_charge_state = 0.0
    if hasattr(q_module, 'aggregation_mode'):
        q_module.aggregation_mode = None

    polarization = cace.modules.Polarization(
        pbc=True, normalization_factor=POLARIZATION_NORM_FACTOR, charge_key='q'
    )
    grad = cace.modules.Grad(y_key='polarization', x_key='positions', output_key='bec_complex')
    dephase = cace.modules.Dephase(input_key='bec_complex', phase_key='phase', output_key='CACE_bec')

    model = cace.models.NeuralNetworkPotential(
        input_modules=None,
        representation=cace_representation,
        output_modules=[q_module, polarization, grad, dephase],
    ).to(device).float().eval()

    return model, cace_representation.cutoff


def main():
    args = parse_args()
    device = torch_tools.init_device(args.device)
    model, cutoff = build_model(args.model_path, device)

    traj = Trajectory(args.traj_path, mode='r')
    total_len = len(traj)
    start = max(0, args.start_frame)
    end = min(total_len, args.end_frame)
    n_to_process = end - start

    w_indices, idx_O, idx_H = identify_atomic_roles(traj[0])
    print(f"Processing frames {start} to {end} (batch size: {BATCH_SIZE})")

    J_buffer = {k: [] for k in ['Total'] + list(LAYER_NAMES.values())}
    N_buffer = {k: [] for k in ['Total'] + list(LAYER_NAMES.values())}
    Pos_buffer, BEC_buffer, Label_buffer = [], [], []

    traj_iter = islice(traj, start, end)
    processed_count = 0
    pbar = tqdm(total=n_to_process, desc=f"Frames [{start}-{end}]")

    while True:
        batch = list(islice(traj_iter, BATCH_SIZE))
        if not batch:
            break

        data_list = [AtomicData.from_atoms(at, cutoff=cutoff) for at in batch]
        loader = torch_geometric.dataloader.DataLoader(
            data_list, batch_size=len(batch), shuffle=False
        )
        batch_data = next(iter(loader)).to(device)

        if batch_data.pos is not None:
            batch_data.pos = batch_data.pos.float()
        if hasattr(batch_data, 'cell') and batch_data.cell is not None:
            batch_data.cell = batch_data.cell.float()

        batch_data = recompute_graph_for_gradients(batch_data)
        out = model(batch_data.to_dict())
        bec_all = out['CACE_bec'].detach().cpu().numpy().reshape(-1, 3, 3)
        ptr = batch_data.ptr.cpu().numpy()

        for i, atoms in enumerate(batch):
            bec_frame = bec_all[ptr[i]:ptr[i + 1]]
            j_dict, n_dict, w_pos, w_bec, w_labels = compute_layer_currents(
                atoms, bec_frame, w_indices, idx_O, idx_H
            )
            for k in J_buffer:
                J_buffer[k].append(j_dict[k])
                N_buffer[k].append(n_dict[k])
            Pos_buffer.append(w_pos.astype(np.float32))
            BEC_buffer.append(w_bec.astype(np.float32))
            Label_buffer.append(w_labels.astype(np.int8))
            processed_count += 1

        pbar.update(len(batch))

        if len(J_buffer['Total']) >= DUMP_CHUNK_SIZE:
            chunk_start = start + processed_count - len(J_buffer['Total'])
            save_chunk(J_buffer, N_buffer, Pos_buffer, BEC_buffer, Label_buffer,
                       chunk_start, args.output_dir)
            for k in J_buffer:
                J_buffer[k], N_buffer[k] = [], []
            Pos_buffer, BEC_buffer, Label_buffer = [], [], []

    if J_buffer['Total']:
        chunk_start = start + processed_count - len(J_buffer['Total'])
        save_chunk(J_buffer, N_buffer, Pos_buffer, BEC_buffer, Label_buffer,
                   chunk_start, args.output_dir)
    pbar.close()


if __name__ == "__main__":
    main()
