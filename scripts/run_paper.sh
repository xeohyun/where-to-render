#!/usr/bin/env bash
# 논문 주요 결과(10개 장면 x 3 seeds)를 다시 만든다: 제안 방법(final3)과 무작위 기준선 3종.
# ACE 저장소 루트에서 실행한다(README "설치" 참고). GPU 하나에 워커 하나.
#   GS_PYTHON=<3dgs_mcmc env python> ACE_PYTHON=<ace env python> GS_REPO=<3dgs-mcmc repo> bash scripts/run_paper.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GS_PYTHON:?Set GS_PYTHON (python of the 3DGS-MCMC env)}"
: "${ACE_PYTHON:?Set ACE_PYTHON (python of the ACE env)}"
: "${GS_REPO:?Set GS_REPO (3DGS-MCMC repository, used as the renderer)}"
export GS_PYTHON ACE_PYTHON GS_REPO
R=sampling/mcmc/s2_aim_20261009/run.py
SCENES=counter,bonsai,kitchen,room,stump,bicycle,garden,tnt_truck,tnt_train,livingroom
GPU=${GPU:-0}

# 1) 고정 평가셋(실제 사진 test / LivingRoom 지정 시점 test)
for s in counter bonsai kitchen room stump bicycle garden tnt_truck tnt_train; do
    "$GS_PYTHON" sampling/eval_protocol.py freeze --scene "$s"
done
"$GS_PYTHON" sampling/eval_protocol.py import-poses --scene livingroom --poses sampling/eval_sets/livingroom_test_poses_adjusted.json

# 2) 학습 시점 선택 -> ACE 학습 -> test 평가. 결과는 sampling/mcmc/s2_aim_20261009/results.json에 쌓인다.
for cfg in final3 random_upright random_full random_pillar; do
    python3 -u "$R" --tag "paper_$cfg" --configs "$cfg" --scenes "$SCENES" --seeds 0,1,2 --evals test --gpu "$GPU" --fixed_gpu
done

# 3) Ablation(최종 설정에서 요소 하나만 바꿈)과 학습 영상 수 0.5x-4x. RUN_ABLATION=1 일 때만.
if [ "${RUN_ABLATION:-0}" = 1 ]; then
    python3 -u "$R" --tag paper_f3ab --configs f3ab_noaim,f3ab_nofb,f3ab_vpp0,f3ab_area,f3ab_norelstd \
        --scenes counter,room,livingroom,bicycle,stump --seeds 0,1,2 --evals test --gpu "$GPU" --fixed_gpu
    for k in 0.5 2 4; do
        python3 -u "$R" --tag "paper_ns_x$k" --configs final3,random_upright --scenes counter,room,bicycle \
            --seeds 0,1,2 --evals test --gpu "$GPU" --fixed_gpu --n_scale "$k"
    done
fi
