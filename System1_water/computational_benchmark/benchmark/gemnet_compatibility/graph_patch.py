"""O(N) radius-graph implementation for the GemNet-OC22 MD calculation."""

import logging

import numpy as np
import torch
from ase.neighborlist import primitive_neighbor_list

import fairchem.core.common.utils as _fc_utils
import fairchem.core.models.base as _fc_base

_original_radius_graph_pbc = _fc_utils.radius_graph_pbc

try:
    from matscipy.neighbours import neighbour_list as _matscipy_neighbor_list
except ImportError:
    _matscipy_neighbor_list = None

# GemNet-OC reuses the same positions for several graph types per force call.
_cache = {"key": None, "value": None}


def _build_neighbor_list(pos, cell, pbc, radius):
    """Return a superset of candidate (center, neighbor, offset) pairs."""
    search_radius = float(radius) + 1e-3
    if _matscipy_neighbor_list is not None:
        from ase import Atoms

        atoms = Atoms(
            numbers=np.ones(len(pos), dtype=int),
            positions=pos,
            cell=cell,
            pbc=pbc,
        )
        center, neighbor, offsets = _matscipy_neighbor_list(
            "ijS", atoms, search_radius
        )
    else:
        center, neighbor, offsets = primitive_neighbor_list(
            "ijS", pbc, cell, pos, search_radius, self_interaction=False
        )

    order = np.argsort(center, kind="stable")
    return center[order], neighbor[order], offsets[order]


def radius_graph_pbc_nl(
    data,
    radius,
    max_num_neighbors_threshold,
    enforce_max_neighbors_strictly=False,
    pbc=None,
):
    """Build the fairchem-compatible periodic radius graph using a neighbor list."""
    if pbc is None:
        pbc = [True, True, True]
    if len(data.natoms) != 1:
        return _original_radius_graph_pbc(
            data,
            radius,
            max_num_neighbors_threshold,
            enforce_max_neighbors_strictly,
            pbc,
        )

    device = data.pos.device
    if hasattr(data, "pbc"):
        data.pbc = torch.atleast_2d(data.pbc)
        pbc = [bool(torch.all(data.pbc[:, dim]).item()) for dim in range(3)]

    pos = data.pos.detach().cpu().numpy().astype(np.float64)
    cell = data.cell[0].detach().cpu().numpy().astype(np.float64)
    key = (float(radius), tuple(pbc), pos.tobytes(), cell.tobytes())
    if _cache["key"] == key:
        center, neighbor, shifts = _cache["value"]
    else:
        center, neighbor, shifts = _build_neighbor_list(pos, cell, pbc, radius)
        _cache["key"] = key
        _cache["value"] = (center, neighbor, shifts)

    index1 = torch.from_numpy(np.ascontiguousarray(center)).to(
        device=device, dtype=torch.long
    )
    index2 = torch.from_numpy(np.ascontiguousarray(neighbor)).to(
        device=device, dtype=torch.long
    )
    unit_cell = torch.from_numpy(np.ascontiguousarray(shifts)).to(
        device=device, dtype=data.cell.dtype
    )

    offsets = unit_cell @ data.cell[0]
    diff = data.pos[index1] - (data.pos[index2] + offsets)
    atom_distance_sqr = torch.sum(diff**2, dim=1)
    mask = torch.logical_and(
        torch.le(atom_distance_sqr, radius * radius),
        torch.gt(atom_distance_sqr, 0.0001),
    )
    if not torch.all(mask):
        index1 = index1[mask]
        index2 = index2[mask]
        unit_cell = unit_cell[mask]
        atom_distance_sqr = atom_distance_sqr[mask]

    neighbor_mask, num_neighbors_image = _fc_utils.get_max_neighbors_mask(
        natoms=data.natoms,
        index=index1,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        enforce_max_strictly=enforce_max_neighbors_strictly,
    )
    if not torch.all(neighbor_mask):
        index1 = torch.masked_select(index1, neighbor_mask)
        index2 = torch.masked_select(index2, neighbor_mask)
        unit_cell = torch.masked_select(
            unit_cell.view(-1, 3), neighbor_mask.view(-1, 1).expand(-1, 3)
        ).view(-1, 3)

    edge_index = torch.stack((index2, index1))
    return edge_index, unit_cell, num_neighbors_image


def apply():
    """Install the neighbor-list graph implementation in fairchem."""
    _fc_utils.radius_graph_pbc = radius_graph_pbc_nl
    _fc_base.radius_graph_pbc = radius_graph_pbc_nl
    backend = (
        "matscipy"
        if _matscipy_neighbor_list is not None
        else "ase (install matscipy for a faster C implementation)"
    )
    logging.info("radius_graph_pbc patched with O(N) neighbor list: %s", backend)
    print(f"[graph_patch] radius_graph_pbc -> O(N) neighbor list ({backend})")
