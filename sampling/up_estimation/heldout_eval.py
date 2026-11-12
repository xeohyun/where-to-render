"""
heldout_eval.py — up 추정 held-out 검증 (2026-10-24)
=====================================================
규칙(PCA → 국소 맨해튼, 부호 = 세 단서 합)을 정할 때 쓰지 않은 3DGS 모델로 같은 함수(view_sampler.estimate_up_from_gaussians,
렌더 공백 단서 포함)를 평가한다. 기준 up = 그 모델의 3DGS 학습 cameras.json 카메라 up 평균(-C2W 두 번째 열, 진단용).
  - playroom (Deep Blending, 새 실내 씬, 일반 3DGS)
  - T&T train·truck (tandt_db 좌표계, 일반 3DGS — 같은 장면이지만 좌표계·Gaussian 모델이 다름)
  - Mip360 6씬 step0a (일반 3DGS, iteration 30000 — 같은 장면이지만 MCMC가 아닌 모델)
env: 3dgs_mcmc, GPU(렌더 공백 단서)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sampling"))
import view_sampler as vs  # noqa: E402

GS = Path(os.environ.get("GS_OUTPUT", "../gaussian-splatting/output"))  # 일반 3DGS 학습 결과 폴더
MODELS = {"playroom(DB)": "ori_playroom", "tnt_train(3DGS)": "ori_train", "tnt_truck(3DGS)": "original_truck"}
MODELS.update({f"{s}(3DGS)": f"{s}_step0a" for s in ["counter", "bonsai", "kitchen", "room", "stump", "bicycle"]})


def main():
    out = {}
    for name, d in MODELS.items():
        ply = GS / d / "point_cloud/iteration_30000/point_cloud.ply"
        if not ply.exists():
            print(f"{name}: ply 없음 {ply}"); continue
        cams = json.loads((GS / d / "cameras.json").read_text())
        ups = np.array([-np.array(c["rotation"])[:, 1] for c in cams])
        ref = ups.mean(0); ref /= np.linalg.norm(ref)
        v = PlyData.read(str(ply))["vertex"]
        g = lambda k: np.asarray(v[k]).astype(np.float64)  # noqa: E731
        xyz = np.stack([g("x"), g("y"), g("z")], 1)
        op = 1 / (1 + np.exp(-g("opacity")))
        sc = np.exp(np.stack([g(f"scale_{i}") for i in range(3)], 1))
        q = np.stack([g(f"rot_{i}") for i in range(4)], 1)
        R = vs.Renderer(str(ply), 3)
        up, info = vs.estimate_up_from_gaussians(xyz, op, sc, q, seed=0, renderer=R)
        del R
        err = float(np.degrees(np.arccos(np.clip(up @ ref, -1, 1))))
        pca = float(np.degrees(np.arccos(np.clip(abs(np.array(info["pca"]) @ ref), 0, 1))))
        out[name] = dict(model=d, up=up.tolist(), ref_up=ref.tolist(), err_deg=err, pca_err_deg=pca, n_cams=len(cams), info=info)
        print(f"{name:16s} 오차 {err:6.1f}° (PCA만 {pca:4.1f}°) 부호점수 {info['sign_score']:+.2f} "
              f"[Sa {info['sign_support']:+.2f} Se {info['sign_rank']:+.2f} Sv {info['sign_void']:+.2f}]", flush=True)
        (Path(__file__).resolve().parent / "results_heldout.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
