#!/usr/bin/env bash
set -euo pipefail
# 3DGS-MCMC로 Mip-NeRF 360 지도(.ply)를 학습한다. 경로는 환경 변수로 지정한다.
PYTHON="${GS_PYTHON:-python}"
REPO="${GS_REPO:?Set GS_REPO to the 3DGS-MCMC repository}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the Mip-NeRF 360 dataset (360_v2)}"
OUT_ROOT="${OUT_ROOT:-$REPO/output}"
export PYTHONNOUSERSITE=1
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
declare -A CAP_MAX=( [counter]=1019926 [bonsai]=677236 [kitchen]=1032835 [room]=805992 [stump]=4059012 [bicycle]=3393587 )
cd "$REPO"
mkdir -p "$OUT_ROOT/logs"
scenes=(counter bonsai kitchen room stump bicycle)
if (( $# )); then scenes=("$@"); fi
for scene in "${scenes[@]}"; do
    [[ -v CAP_MAX[$scene] ]] || { echo "Unknown scene: $scene" >&2; exit 1; }
    model="$OUT_ROOT/${scene}_mcmc"
    if [[ -e "$model" ]]; then
        echo "Output already exists; inspect before resuming: $model" >&2
        exit 1
    fi
    "$PYTHON" -u train.py --source_path "$DATA_ROOT/$scene" --model_path "$model" \
        --eval --init_type sfm --cap_max "${CAP_MAX[$scene]}" \
        --scale_reg 0.01 --opacity_reg 0.01 --noise_lr 5e5 \
        --iterations 30000 --port 0 2>&1 | tee "$OUT_ROOT/logs/${scene}_train.log"
    test -s "$model/point_cloud/iteration_30000/point_cloud.ply"
    echo "[$scene] completed"
done
