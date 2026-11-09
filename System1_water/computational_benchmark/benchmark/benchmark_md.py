#!/usr/bin/env python3
"""Uniform 100-step ASE MD benchmark for the foundation models in this directory."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import struct
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read
from ase.md.npt import NPT
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)


ROOT = Path(__file__).resolve().parent
# Place external teacher checkpoints in this directory; no weights are bundled.
FOUNDATION_MODELS = ROOT / "foundation_models"
ORB_MODELS = FOUNDATION_MODELS / "orb3"
GEMNET_CHECKPOINT = FOUNDATION_MODELS / "gnoc_oc22_all_s2ef.pt"
UPET_CHECKPOINT = FOUNDATION_MODELS / "pet-omatpes-l-v0.1.0.ckpt"
# Reuse the existing published student checkpoint in System1_water.
MACELES_ORBMOL_CONS_CHECKPOINT = (
    ROOT.parent.parent / "water_MLIPs_RPBE-D3_sampled_configs" / "models" /
    "mlp-train-orbmol_cons" / "H2O-DFT-orbmol_cons_stagetwo.model"
)
# Retained optional original-driver branch; CACELES is not part of Figure S5.
CACELES_SCAN_CHECKPOINT = (
    ROOT.parents[2] / "System3_TiO2-water" / "MLIPs" / "Direct_fit" /
    "best_model_DFT_SCAN.pth"
)
GEMNET_REFERENCE = ROOT / "gemnet_compatibility"

MODEL_NAMES = (
    "uma-s-omol",
    "uma-m-omol",
    "esen-sm-con",
    "esen-md-dir",
    "mace-mp0",
    "mace-omol-xl",
    "mace-off24-medium",
    "mace-mh-1",
    "orb-v3-conservative-omol",
    "orb-v3-direct-omol",
    "gemnet-oc22",
    "upet-pet-omatpes-l-conservative",
    "maceles-orbmol-cons-fp64",
    "maceles-orbmol-cons-fp32",
    "caceles-scan-fp32",
)
PROTOCOL_VERSION = "h200-md-v4"
PRECISION_POLICY = "per-model-default"


def synchronize_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def torch_metadata(require_h200: bool) -> dict:
    import torch

    cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda else None
    if not cuda:
        raise RuntimeError("CUDA is not available; this benchmark is GPU-only")
    if require_h200 and "H200" not in gpu_name.upper():
        raise RuntimeError(f"Expected an H200 GPU, found: {gpu_name}")
    props = torch.cuda.get_device_properties(0)
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_memory_gib": props.total_memory / 1024**3,
    }


def package_versions() -> dict[str, str]:
    versions = {}
    for distribution in ("ase", "torch", "mace-torch", "fairchem-core", "orb-models", "upet"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_atoms(atoms, replicate: tuple[int, int, int]):
    atoms = atoms.copy()
    atoms.pbc = True
    atoms.wrap(eps=1e-12)
    atoms = atoms.repeat(replicate)
    for key in list(atoms.info):
        if key in {"energy", "stress", "free_energy", "virial"} or "energy" in key or "stress" in key:
            del atoms.info[key]
    for key in ("forces", "momenta", "velocities"):
        atoms.arrays.pop(key, None)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    return atoms


def initialize_velocities(atoms, temperature_k: float, seed: int) -> None:
    rng = np.random.RandomState(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k, rng=rng)
    Stationary(atoms, preserve_temperature=True)
    ZeroRotation(atoms, preserve_temperature=True)


def _read_safetensors(path: Path) -> dict:
    """Read the F32/I64 safetensors used by the two local Orb v3 models."""
    import torch

    dtype_map = {
        "F32": (np.dtype("<f4"), torch.float32),
        "I64": (np.dtype("<i8"), torch.int64),
    }
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
        data_start = 8 + header_size
        state = {}
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            if spec["dtype"] not in dtype_map:
                raise ValueError(f"Unsupported safetensors dtype {spec['dtype']} in {path}")
            np_dtype, torch_dtype = dtype_map[spec["dtype"]]
            start, end = spec["data_offsets"]
            handle.seek(data_start + start)
            array = np.frombuffer(handle.read(end - start), dtype=np_dtype).copy()
            array = array.reshape(spec["shape"])
            state[name] = torch.from_numpy(array).to(dtype=torch_dtype)
    return state


def orb_torch_checkpoint(safe_path: Path, cache_dir: Path) -> Path:
    """Convert a local Orb safetensors state dict once for Orb's torch.load API."""
    import fcntl
    import torch

    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{safe_path.stem}.pt"
    lock_path = cache_dir / f"{safe_path.stem}.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists() or target.stat().st_mtime < safe_path.stat().st_mtime:
            temporary = target.with_suffix(f".{os.getpid()}.tmp")
            torch.save(_read_safetensors(safe_path), temporary)
            os.replace(temporary, target)
    return target


def build_calculator(model: str, model_cache: Path):
    if model == "caceles-scan-fp32":
        import torch
        from cace.calculators import CACECalculator

        if not CACELES_SCAN_CHECKPOINT.is_file():
            raise FileNotFoundError(CACELES_SCAN_CHECKPOINT)
        torch.set_default_dtype(torch.float32)
        torch.set_float32_matmul_precision("high")
        cace_nnp = torch.load(
            CACELES_SCAN_CHECKPOINT,
            weights_only=False,
            map_location="cuda",
        )
        cace_nnp.to(dtype=torch.float32).eval()
        calc = CACECalculator(
            model_path=cace_nnp,
            device="cuda",
            compute_stress=False,
            energy_key="CACE_energy",
            forces_key="CACE_forces",
        )
        dtype = str(next(cace_nnp.parameters()).dtype).replace("torch.", "")
        dtype += "-high"
        return calc, str(CACELES_SCAN_CHECKPOINT), dtype

    if model in {"maceles-orbmol-cons-fp64", "maceles-orbmol-cons-fp32"}:
        from mace.calculators import MACECalculator

        if not MACELES_ORBMOL_CONS_CHECKPOINT.is_file():
            raise FileNotFoundError(MACELES_ORBMOL_CONS_CHECKPOINT)
        requested_dtype = "float64" if model.endswith("fp64") else "float32"
        calc = MACECalculator(
            model_paths=str(MACELES_ORBMOL_CONS_CHECKPOINT),
            device="cuda",
            default_dtype=requested_dtype,
        )
        dtype = str(next(calc.models[0].parameters()).dtype).replace("torch.", "")
        return calc, str(MACELES_ORBMOL_CONS_CHECKPOINT), dtype

    if model in {"uma-s-omol", "uma-m-omol", "esen-sm-con", "esen-md-dir"}:
        from fairchem.core import pretrained_mlip
        from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

        identifiers = {
            "uma-s-omol": ("uma-s-1p1", "omol"),
            "uma-m-omol": ("uma-m-1p1", "omol"),
            "esen-sm-con": ("esen-sm-conserving-all-oc25", None),
            "esen-md-dir": ("esen-md-direct-all-oc25", None),
        }
        identifier, task = identifiers[model]
        predictor = pretrained_mlip.get_predict_unit(identifier, device="cuda")
        kwargs = {"task_name": task} if task else {}
        dtype = str(predictor.inference_settings.base_precision_dtype).replace("torch.", "")
        return FAIRChemCalculator(predictor, **kwargs), identifier, dtype

    if model in {"mace-mp0", "mace-omol-xl", "mace-off24-medium", "mace-mh-1"}:
        from mace.calculators import MACECalculator, mace_mp

        paths = {
            "mace-mp0": FOUNDATION_MODELS / "MACE-MP0-128-L2_epoch-199.model",
            "mace-omol-xl": FOUNDATION_MODELS / "MACE-omol-0-extra-large-1024.model",
            "mace-off24-medium": FOUNDATION_MODELS / "MACE-OFF24_medium.model",
            "mace-mh-1": FOUNDATION_MODELS / "mace-mh-1.model",
        }
        path = paths[model]
        if not path.is_file():
            raise FileNotFoundError(path)
        if model == "mace-mh-1":
            calc = mace_mp(model=str(path), device="cuda", head="omol")
        else:
            calc = MACECalculator(model_paths=str(path), device="cuda")
        dtype = str(next(calc.models[0].parameters()).dtype).replace("torch.", "")
        return calc, str(path), dtype

    if model in {"orb-v3-conservative-omol", "orb-v3-direct-omol"}:
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator

        safe_paths = {
            "orb-v3-conservative-omol": ORB_MODELS / "orb-v3-conservative-omol-20250820.safetensors",
            "orb-v3-direct-omol": ORB_MODELS / "orb-v3-direct-omol-20250820.safetensors",
        }
        safe_path = safe_paths[model]
        if not safe_path.is_file():
            raise FileNotFoundError(safe_path)
        checkpoint = orb_torch_checkpoint(safe_path, model_cache)
        loader = (
            pretrained.orb_v3_conservative_omol
            if model == "orb-v3-conservative-omol"
            else pretrained.orb_v3_direct_omol
        )
        orb_model, atoms_adapter = loader(
            weights_path=str(checkpoint),
            device="cuda",
        )
        dtype = str(next(orb_model.parameters()).dtype).replace("torch.", "")
        return (
            ORBCalculator(orb_model, atoms_adapter, device="cuda"),
            str(safe_path),
            dtype,
        )

    if model == "gemnet-oc22":
        sys.path.insert(0, str(GEMNET_REFERENCE))
        import graph_patch
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator

        if not GEMNET_CHECKPOINT.is_file():
            raise FileNotFoundError(GEMNET_CHECKPOINT)
        graph_patch.apply()
        calc = OCPCalculator(checkpoint_path=str(GEMNET_CHECKPOINT), cpu=False)
        return calc, str(GEMNET_CHECKPOINT), "float32"

    if model == "upet-pet-omatpes-l-conservative":
        from upet.calculator import UPETCalculator

        if not UPET_CHECKPOINT.is_file():
            raise FileNotFoundError(UPET_CHECKPOINT)
        calc = UPETCalculator(
            checkpoint_path=str(UPET_CHECKPOINT),
            device="cuda",
            non_conservative=False,
        )
        return calc, str(UPET_CHECKPOINT), "checkpoint default"

    raise ValueError(f"Unknown model: {model}")


def run_md(
    atoms,
    calc,
    steps: int,
    timestep_fs: float,
    temperature_k: float,
    ttime_fs: float,
    thermostat_seed: int,
) -> tuple[float, float]:
    """Prime the initial force, then time exactly one new force evaluation per MD step."""
    atoms.calc = calc
    dyn = NPT(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_k,
        ttime=ttime_fs * units.fs,
        pfactor=None,
        externalstress=0.0,
    )
    synchronize_cuda()
    prime_start = time.perf_counter()
    initial_forces = atoms.get_forces(md=True)
    synchronize_cuda()
    prime_elapsed = time.perf_counter() - prime_start
    if not np.all(np.isfinite(initial_forces)):
        raise FloatingPointError("Non-finite force detected during initial force evaluation")
    start = time.perf_counter()
    dyn.run(steps)
    synchronize_cuda()
    elapsed = time.perf_counter() - start
    final_forces = atoms.get_forces(md=True)
    if not np.all(np.isfinite(final_forces)) or not np.all(np.isfinite(atoms.positions)):
        raise FloatingPointError("Non-finite force or position detected after timed MD")
    return prime_elapsed, elapsed


def result_path(args) -> Path:
    label = "x".join(map(str, args.replicate))
    return args.output_dir / f"{args.model}__r{label}.json"


def benchmark(args) -> dict:
    import ase

    metadata = torch_metadata(args.require_h200)
    base = read(args.input, index=args.frame)
    atoms_template = clean_atoms(base, args.replicate)
    if args.model == "gemnet-oc22":
        # The reference GemNet-OC22 setup marks all atoms as freely movable.
        atoms_template.set_tags(np.full(len(atoms_template), 2, dtype=int))
    metadata.update(
        {
            "status": "running",
            "protocol_version": PROTOCOL_VERSION,
            "precision_policy": (
                "explicit-fp64-fp32-variants"
                if args.model.startswith("maceles-orbmol-cons-")
                else "explicit-fp32-high"
                if args.model == "caceles-scan-fp32"
                else PRECISION_POLICY
            ),
            "model": args.model,
            "input": str(args.input.resolve()),
            "input_sha256": sha256(args.input),
            "benchmark_script_sha256": sha256(Path(__file__).resolve()),
            "frame": args.frame,
            "replicate": list(args.replicate),
            "base_atoms": len(base),
            "atoms": len(atoms_template),
            "cell_lengths_angstrom": atoms_template.cell.lengths().tolist(),
            "temperature_k": args.temperature,
            "ensemble": "NVT-NoseHoover",
            "ttime_fs": args.ttime,
            "timestep_fs": args.timestep,
            "warmup_steps": args.warmup_steps,
            "md_steps": args.steps,
            "timed_force_evaluations": args.steps,
            "seed": args.seed,
            "ase_version": ase.__version__,
            "python_version": platform.python_version(),
            "package_versions": package_versions(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        }
    )

    synchronize_cuda()
    start = time.perf_counter()
    calc, model_source, precision = build_calculator(args.model, args.model_cache)
    synchronize_cuda()
    metadata["model_load_seconds"] = time.perf_counter() - start
    metadata["model_source"] = model_source
    metadata["precision"] = precision

    warm = atoms_template.copy()
    initialize_velocities(warm, args.temperature, args.seed)
    warm_prime, warm_elapsed = run_md(
        warm,
        calc,
        args.warmup_steps,
        args.timestep,
        args.temperature,
        args.ttime,
        args.seed + 1,
    )
    metadata["warmup_initial_force_seconds"] = warm_prime
    metadata["warmup_seconds"] = warm_elapsed

    atoms = atoms_template.copy()
    initialize_velocities(atoms, args.temperature, args.seed)
    initial_force_elapsed, elapsed = run_md(
        atoms,
        calc,
        args.steps,
        args.timestep,
        args.temperature,
        args.ttime,
        args.seed + 1,
    )
    import torch

    metadata.update(
        {
            "status": "ok",
            "initial_force_seconds": initial_force_elapsed,
            "md_seconds": elapsed,
            "milliseconds_per_step": 1000.0 * elapsed / args.steps,
            "steps_per_second": args.steps / elapsed,
            "atom_steps_per_second": len(atoms) * args.steps / elapsed,
            "ns_per_day": args.steps * args.timestep * 1e-6 / elapsed * 86400.0,
            "final_temperature_k": float(atoms.get_temperature()),
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        }
    )
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument(
        "--replicate",
        required=True,
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
    )
    parser.add_argument("--input", type=Path, default=ROOT / "ini.xyz")
    parser.add_argument("--frame", type=int, default=-1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--timestep", type=float, default=1.0)
    parser.add_argument("--ttime", type=float, default=20.0, help="Nose-Hoover coupling time in fs")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark_results")
    parser.add_argument("--model-cache", type=Path, default=ROOT / "benchmark_results/model_cache")
    parser.add_argument("--require-h200", action="store_true")
    args = parser.parse_args()
    args.replicate = tuple(args.replicate)
    if min(args.replicate) < 1 or args.steps < 1 or args.warmup_steps < 1 or args.ttime <= 0:
        parser.error("replicate, steps, and warmup-steps must all be positive")
    return args


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = result_path(args)
    try:
        result = benchmark(args)
        exit_code = 0
    except Exception as error:
        error_text = f"{type(error).__name__}: {error}".lower()
        is_oom = "outofmemory" in error_text or "out of memory" in error_text
        result = {
            "status": "failed",
            "protocol_version": PROTOCOL_VERSION,
            "precision_policy": PRECISION_POLICY,
            "failure_kind": "out_of_memory" if is_oom else "other",
            "model": args.model,
            "replicate": list(args.replicate),
            "md_steps": args.steps,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "hostname": socket.gethostname(),
        }
        # The matrix runner uses this distinct code to prune larger boxes only
        # after a genuine memory-capacity failure.
        exit_code = 42 if is_oom else 1
    temporary = output.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
