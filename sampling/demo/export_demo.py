"""
export_demo.py — 최종 발표용 웹 데모 데이터 추출 (2026-10-24)
=============================================================
씬마다 3DGS 지도 점 표본, 정책별(제안 방법 / random 두 종류) 학습 카메라와 썸네일,
실제 test 사진과 정책별 ACE 추정 포즈·오차를 정적 파일로 내보낸다. 웹페이지(index.html)는 이 파일만 읽는다.

좌표: 지도의 up(scenes.json)을 +z로 돌리고, 수평 첫 주성분을 +x로 둔 지역 좌표(중심 = 점 중앙값). 단위는 장면 단위 그대로.
seed 0 학습셋·헤드를 쓴다(씬 요약 수치는 results.json의 seed 3개 평균).
2026-10-28: 제안 방법을 최종 설정(config final3: 무작위 높이 + 구조물을 겨누는 pitch + up 자동 추정)으로 바꿨다. random 기준선은 scenes.json up(GT) 그대로라 비교는 제안 방법에 불리하다.

env: 3dgs_mcmc (plyfile, PIL)
    python sampling/demo/export_demo.py --out docs
"""
import argparse
import json
import statistics as S
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sampling"))
import view_sampler as vs  # noqa: E402

OUTP = ROOT / "output/s2_aim_20261009"
RES = ROOT / "sampling/mcmc/s2_aim_20261009/results.json"

SCENES = {
    "counter": dict(label="counter (Mip360 실내)", kind="mip"),
    "room": dict(label="room (Mip360 실내)", kind="mip"),
    "bicycle": dict(label="bicycle (Mip360 야외)", kind="mip"),
    "stump": dict(label="stump (Mip360 야외)", kind="mip"),
    "garden": dict(label="garden (held-out 야외)", kind="heldout"),
    "livingroom": dict(label="LivingRoom (SuperSplat)", kind="ss"),
}
POLICIES = [("ours", "final3", "제안 방법"), ("rand_up", "random_upright", "Random (같은 높이대·수평)"),
            ("rand_full", "random_full", "Random (완전 무작위)")]


def train_dir(s, cfg):
    kind = SCENES[s]["kind"]
    if cfg in ("aimDrf_rs", "upauto", "final2", "final3") or kind == "heldout":
        return ROOT / f"out/{s}_s2_{cfg}_s0"
    if cfg == "random_upright":
        return ROOT / f"out/{s}_avr_random_upright_s0"
    return ROOT / (f"out/{s}_avr_random_full_s0" if kind == "ss" else f"out/{s}_rk_random_full")


def quat_to_R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def sprite(paths, tile_w, cols, out_path, quality=72):
    ims = [Image.open(p).convert("RGB") for p in paths]
    th = int(round(tile_w * ims[0].height / ims[0].width))
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (tile_w * cols, th * rows), (20, 20, 20))
    for i, im in enumerate(ims):
        sheet.paste(im.resize((tile_w, th), Image.LANCZOS), ((i % cols) * tile_w, (i // cols) * th))
    sheet.save(out_path, quality=quality, optimize=True)
    return dict(file=out_path.name, tile_w=tile_w, tile_h=th, cols=cols, n=len(ims))


def summary(res, s, cfg):
    v = [res.get(f"{s}|{cfg}|s{k}|test") for k in range(3)]
    v = [x for x in v if x]
    return dict(n_seed=len(v), t=S.mean(x["med_trans"] for x in v), r=S.mean(x["med_rot"] for x in v),
                acc=S.mean(x["acc_10cm10deg"] for x in v)) if v else None


def export_scene(s, out, a, res):
    cfg = vs.load_scene_cfg(s)
    from plyfile import PlyData
    v = PlyData.read(cfg["ply"])["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    op = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
    col = np.clip(0.5 + 0.28209479 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1), 0, 1)
    solid = xyz[op > 0.5]  # 장면 크기(extent)는 불투명한 Gaussian만으로 잰다(배경 floater 영향 제외)
    m = op > a.min_opacity
    xyz, col = xyz[m], col[m]
    lo, hi = np.percentile(xyz, 3, axis=0), np.percentile(xyz, 97, axis=0)
    span = hi - lo
    keep = np.all((xyz >= lo - 0.15 * span) & (xyz <= hi + 0.15 * span), 1)
    xyz, col = xyz[keep], col[keep]
    up = cfg["up"]
    c0 = np.median(xyz, 0)
    hz = (xyz - c0) - np.outer((xyz - c0) @ up, up)
    _, _, vt = np.linalg.svd(hz[np.random.default_rng(0).choice(len(hz), min(len(hz), 200000), replace=False)], full_matrices=False)
    ex = vt[0] - (vt[0] @ up) * up; ex /= np.linalg.norm(ex)
    B = np.stack([ex, np.cross(up, ex), up])          # 행: 지역 x, y, z(=up)
    loc = lambda p: (np.asarray(p) - c0) @ B.T        # noqa: E731
    rot = lambda d: np.asarray(d) @ B.T               # noqa: E731
    idx = np.random.default_rng(1).choice(len(xyz), min(len(xyz), a.n_points), replace=False)
    P, C = loc(xyz[idx]), col[idx]
    plo, phi = P.min(0), P.max(0)
    q = np.round((P - plo) / np.maximum(phi - plo, 1e-9) * 65535).astype(np.uint16)
    sd = out / s
    sd.mkdir(parents=True, exist_ok=True)
    raw = q.tobytes() + (C * 255).round().astype(np.uint8).tobytes()  # uint16 위치 x3 + uint8 색 x3, 아티팩트 서빙용 base64 JSON
    (sd / "points.json").write_text(json.dumps(dict(b64=__import__("base64").b64encode(raw).decode())))

    meta = dict(scene=s, label=SCENES[s]["label"], n_points=len(P), bbox=[plo.tolist(), phi.tolist()], policies=[], tests=[])
    ua = json.loads((ROOT / "sampling/up_estimation/results_up_auto_final.json").read_text())["scenes"].get(s)
    if ua:
        meta["up_auto"] = dict(err_deg=ua["err_deg"], up_local=np.round(rot(ua["up"]), 4).tolist())
    for key, cname, label in POLICIES:
        d = train_dir(s, cname)
        frames = sorted((d / "train" / "rgb").glob("*.png"))
        cams = []
        for f in frames:
            T = np.loadtxt(d / "train" / "poses" / f.name.replace(".color.png", ".pose.txt"))
            cams.append([*np.round(loc(T[:3, 3]), 4), *np.round(rot(T[:3, 2]), 4), *np.round(rot(-T[:3, 1]), 4)])
        sp = sprite(frames, a.train_tile, 16, sd / f"train_{key}.jpg")
        meta["policies"].append(dict(key=key, cfg=cname, label=label, cams=cams, sprite=sp, summary=summary(res, s, cname)))

    td = ROOT / f"out/{s}_pd_eval_test/test"
    tframes = sorted((td / "rgb").glob("*"))
    est = {}
    for key, cname, _ in POLICIES:
        rows = (OUTP / f"poses_{s}_pd_eval_test_{s}_s2_{cname}_s0_test.txt").read_text().split("\n")
        for r in rows:
            if not r.strip():
                continue
            x = r.split()
            Rw = quat_to_R(*map(float, x[1:5])); t = np.array(list(map(float, x[5:8])))
            est.setdefault(x[0], {})[key] = dict(c=np.round(loc(-Rw.T @ t), 4).tolist(), f=np.round(rot(Rw.T @ [0, 0, 1.0]), 4).tolist(),
                                                 r=round(float(x[8]), 3), t=round(float(x[9]), 4))
    for f in tframes:
        T = np.loadtxt(td / "poses" / (f.name.split(".")[0] + ".pose.txt"))
        e = est.get(f.name) or est.get(f.name.split(".")[0] + ".color.png")
        meta["tests"].append(dict(name=f.name, c=np.round(loc(T[:3, 3]), 4).tolist(), f=np.round(rot(T[:3, 2]), 4).tolist(),
                                  u=np.round(rot(-T[:3, 1]), 4).tolist(), est=e))
    meta["test_sprite"] = sprite(tframes, a.test_tile, 8, sd / "test.jpg", quality=80)
    # 커버리지: test 카메라마다 시선 각도 차 45도 안에서 가장 가까운 학습 카메라까지의 거리
    tc = np.array([t["c"] for t in meta["tests"]]); tf = np.array([t["f"] for t in meta["tests"]])
    sl = loc(solid)
    ext = float(np.linalg.norm(np.percentile(sl, 95, 0) - np.percentile(sl, 5, 0)))
    for p in meta["policies"]:
        cc = np.array([c[:3] for c in p["cams"]]); cf = np.array([c[3:6] for c in p["cams"]])
        dist = np.linalg.norm(tc[:, None] - cc[None], axis=2)
        dist[(tf @ cf.T) < np.cos(np.deg2rad(45))] = np.inf
        nn = dist.min(1)
        p["coverage"] = dict(median_nn=float(np.median(nn)), frac_within=float(np.mean(nn < 0.1 * ext)), thr=0.1 * ext)
    meta["extent"] = ext
    (sd / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    print(f"{s}: points {len(P)}, tests {len(tframes)}, " +
          ", ".join(f"{p['key']} {len(p['cams'])}장 cov {p['coverage']['frac_within']:.2f}" for p in meta["policies"]), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--n_points", type=int, default=120000)
    ap.add_argument("--min_opacity", type=float, default=0.2)
    ap.add_argument("--train_tile", type=int, default=176)
    ap.add_argument("--test_tile", type=int, default=400)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = json.loads(RES.read_text())
    for s in a.scenes.split(","):
        export_scene(s, out, a, res)
    (out / "scenes.json").write_text(json.dumps([dict(key=s, label=SCENES[s]["label"]) for s in a.scenes.split(",")], ensure_ascii=False))


if __name__ == "__main__":
    main()
