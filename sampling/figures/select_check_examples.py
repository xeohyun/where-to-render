"""
select_check_examples.py — 판넬용 렌더 검사 예시 선정 (2026-11-06)
=================================================================
counter 장면의 Random(같은 높이대) 학습셋과 Ours(final3) 학습셋의 각 포즈를 알고리즘과 같은 방식(160px 렌더,
view_sampler.frame_metrics)으로 다시 재서, 기준별로 가장 전형적인 예를 고른다.
  - 빈 화면(void > 0.05)에서 void가 가장 큰 Random 프레임
  - (근접 near > 0.20 프레임은 counter Random 학습셋에 없어 제외)
  - 깊이 퍼짐(relstd > 0.30)에서 relstd가 가장 큰 Random 프레임
  - 통과 예: 세 기준을 모두 통과한 Ours 프레임 중 void+near가 가장 작은 것
결과: select_check_examples.json (경로, 지표 값, 라벨). up은 Ours 실행과 같은 자동 추정 up을 쓴다.
env: 3dgs_mcmc, GPU
    CUDA_VISIBLE_DEVICES=0 python sampling/figures/select_check_examples.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sampling"))
import view_sampler as vs  # noqa: E402

OUT = Path(__file__).resolve().parent
SCENE = "counter"


def metrics_for(R, G, d, w=160):
    rows = []
    rgb = sorted((d / "train/rgb").glob("*.png"))
    from PIL import Image
    W, H = Image.open(rgb[0]).size
    for p in rgb:
        stem = p.name.split(".")[0]
        c2w = np.loadtxt(d / "train/poses" / f"{stem}.pose.txt")
        f = float((d / "train/calibration" / f"{stem}.calibration.txt").read_text())
        Rw = c2w[:3, :3].T; tw = -Rw @ c2w[:3, 3]
        h = max(1, int(round(H * w / W)))
        m = vs.frame_metrics(*R.stats(R.camera(Rw, tw, f * w / W, w, h), Rw, tw), G["near_dist"])
        rows.append(dict(path=str(p.relative_to(ROOT)), **{k: float(v) for k, v in m.items()}))
    return rows


def main():
    cfg = vs.load_scene_cfg(SCENE)
    run = json.loads((ROOT / f"out/{SCENE}_s2_final3_s0/run_args.json").read_text())
    cfg["up"] = np.array(run["up"], float)
    a = vs.geometry_namespace(seed=0)
    a.up = "config"
    R, G = vs.load_scene(cfg, a)
    cache = OUT / "check_metrics_cache.json"
    if cache.exists():
        rnd, ours = json.loads(cache.read_text())
    else:
        rnd = metrics_for(R, G, ROOT / f"out/{SCENE}_avr_random_upright_s0")
        ours = metrics_for(R, G, ROOT / f"out/{SCENE}_s2_final3_s0")
        cache.write_text(json.dumps([rnd, ours]))
    ok = lambda r: r["void_frac"] <= 0.05 and r["near_frac"] <= 0.20 and r["relstd"] <= 0.30  # noqa: E731
    sel = []
    v = max((r for r in rnd if r["void_frac"] > 0.05), key=lambda r: r["void_frac"])
    sel.append(dict(path=v["path"], keep=False, label="빈 화면", value=f"빈 픽셀 {100 * v['void_frac']:.0f}% (기준 ≤ 5%)"))
    # counter Random 학습셋에는 근접 기준(> 20%) 탈락 프레임이 없어 근접 예시는 넣지 않는다(2026-11-06 확인).
    cand = [r for r in rnd if r["relstd"] > 0.30 and r["void_frac"] <= 0.05 and r["near_frac"] <= 0.20]
    s = max(cand or [r for r in rnd if r["relstd"] > 0.30], key=lambda r: r["relstd"])
    sel.append(dict(path=s["path"], keep=False, label="흐림", value=f"깊이 퍼짐 {s['relstd']:.2f} (기준 ≤ 0.30)"))
    # 통과 예: 세 기준을 통과한 Ours 프레임 중 깊이 퍼짐이 가장 낮은 8장을 보고, 장면이 가장 넓게 담긴 frame_000069를 사람이 골랐다(2026-11-06).
    g = next(r for r in ours if r["path"].endswith("frame_000069.color.png"))
    assert ok(g)
    sel.append(dict(path=g["path"], keep=True, label="통과", value=f"빈 {100 * g['void_frac']:.0f}% · 퍼짐 {g['relstd']:.2f}"))
    (OUT / "select_check_examples.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1))
    stats = dict(random_pass=sum(map(ok, rnd)) / len(rnd), ours_pass=sum(map(ok, ours)) / len(ours), n_random=len(rnd), n_ours=len(ours))
    (OUT / "check_pass_rates.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(sel, ensure_ascii=False, indent=1)); print(stats)


if __name__ == "__main__":
    main()
