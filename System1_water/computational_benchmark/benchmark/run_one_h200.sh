#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:?usage: run_one_h200.sh MODEL NXxNYxNZ [RESULT_DIR]}"
REPLICATE="${2:?usage: run_one_h200.sh MODEL NXxNYxNZ [RESULT_DIR]}"
RESULT_DIR="${3:-${ROOT}/benchmark_results}"
IFS=x read -r NX NY NZ <<< "${REPLICATE}"
if [[ -z "${NX:-}" || -z "${NY:-}" || -z "${NZ:-}" ]]; then
    echo "Replication must have the form NXxNYxNZ, for example 2x3x4" >&2
    exit 2
fi

case "${MODEL}" in
    uma-s-omol|uma-m-omol|esen-sm-con|esen-md-dir)
        BENCHMARK_PYTHON="${BENCHMARK_PYTHON:-python3}"
        ;;
    mace-mp0|mace-omol-xl|mace-off24-medium|mace-mh-1|maceles-orbmol-cons-fp64|maceles-orbmol-cons-fp32)
        BENCHMARK_PYTHON="${BENCHMARK_PYTHON:-python3}"
        ;;
    caceles-scan-fp32)
        BENCHMARK_PYTHON="${BENCHMARK_PYTHON:-python3}"
        ;;
    orb-v3-conservative-omol|orb-v3-direct-omol)
        BENCHMARK_PYTHON="${BENCHMARK_PYTHON:-python3}"
        ;;
    gemnet-oc22)
        BENCHMARK_PYTHON="${BENCHMARK_PYTHON:-python3}"
        ;;
    upet-pet-omatpes-l-conservative)
        BENCHMARK_PYTHON="${BENCHMARK_PYTHON:-python3}"
        ;;
    *)
        echo "Unknown model: ${MODEL}" >&2
        exit 2
        ;;
esac

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-14}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-14}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-14}"
export PYTHONUNBUFFERED=1
export WARP_CACHE_PATH="${RESULT_DIR}/runtime_cache/${MODEL}/warp"
export TORCHINDUCTOR_CACHE_DIR="${RESULT_DIR}/runtime_cache/${MODEL}/torchinductor"
export TRITON_CACHE_DIR="${RESULT_DIR}/runtime_cache/${MODEL}/triton"
export MPLCONFIGDIR="${RESULT_DIR}/runtime_cache/${MODEL}/matplotlib"

mkdir -p \
    "${RESULT_DIR}/logs" \
    "${WARP_CACHE_PATH}" \
    "${TORCHINDUCTOR_CACHE_DIR}" \
    "${TRITON_CACHE_DIR}"
mkdir -p "${MPLCONFIGDIR}"
LOG="${RESULT_DIR}/logs/${MODEL}__r${REPLICATE}.log"

"${BENCHMARK_PYTHON}" "${ROOT}/benchmark_md.py" \
    --model "${MODEL}" \
    --replicate "${NX}" "${NY}" "${NZ}" \
    --steps 100 \
    --warmup-steps 5 \
    --temperature 300 \
    --timestep 1.0 \
    --ttime 20 \
    --seed 20260716 \
    --input "${ROOT}/ini.xyz" \
    --output-dir "${RESULT_DIR}" \
    --model-cache "${RESULT_DIR}/model_cache" \
    --require-h200 2>&1 | tee "${LOG}"
