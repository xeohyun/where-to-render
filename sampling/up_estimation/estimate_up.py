"""
estimate_up.py — 3DGS 지도(.ply)만으로 장면의 up(중력 반대 방향)을 추정한다 (2026-10-24)
=========================================================================================
지금까지 Mip360·garden·T&T는 COLMAP 실제 카메라의 평균 up(GT)을 썼다. 이 스크립트는 카메라·사진 없이 Gaussian만 쓴다.

1) 축(방향, 부호 제외): 평평한 Gaussian(가장 짧은 축이 나머지보다 충분히 짧음)의 법선 = 가장 짧은 축.
   법선 분포(반구 위 축 분포)에서 봉우리(바닥·벽·탁자 등 지배 평면 방향)를 찾는다.
   후보 봉우리 중 "그 법선을 가진 Gaussian들이 높이 방향으로 넓게 퍼진 여러 층을 이루지 않고 큰 평면 하나에 몰리는가"보다
   단순한 규칙을 쓴다: 수평면(바닥·지면·탁자)은 사람이 찍은 장면에서 가장 많이 관측되는 평면이므로, 법선 질량이 가장 큰 봉우리를 up 축으로 본다.
   기본(--axis_rule pca_snap): 장면이 가장 얇게 퍼진 방향(위치 PCA 최소 분산 축)으로 대략 수직을 잡고, 그 근처 40° 안의
   법선 봉우리가 있으면 그중 장면이 가장 얇은 봉우리 방향으로 맞춘다(평면이 없는 야외는 PCA 그대로).
   (비교용: mass = 법선 질량 최대, spread = 위치 폭 최소 봉우리, mass_spread = 질량 상위 3개 중 위치 폭 최소)
2) 부호: 그 축 방향으로 지배 평면(질량이 가장 큰 높이)을 찾고, 나머지 Gaussian이 더 많이 놓인 쪽을 위로 본다
   (물체는 바닥 위에 놓이고, 바닥 아래는 관측되지 않아 비어 있다).

진단: scenes.json의 up(GT 또는 렌더로 확인한 값)과의 각도 오차를 같이 출력한다. 알고리즘 입력으로는 쓰지 않는다.
env: 3dgs_mcmc (plyfile), CPU
    python sampling/up_estimation/estimate_up.py --scenes counter,bicycle
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sampling"))
import view_sampler as vs  # noqa: E402

ALL = ["counter", "bonsai", "kitchen", "room", "stump", "bicycle", "garden", "tnt_truck", "tnt_train", "livingroom", "street"]


def load(ply, max_n, rng):
    from plyfile import PlyData
    v = PlyData.read(ply)["vertex"]
    n = len(v.data)
    idx = np.sort(rng.choice(n, max_n, replace=False)) if n > max_n else np.arange(n)
    g = lambda k: np.asarray(v[k])[idx].astype(np.float64)  # noqa: E731
    xyz = np.stack([g("x"), g("y"), g("z")], 1)
    op = 1 / (1 + np.exp(-g("opacity")))
    sc = np.exp(np.stack([g(f"scale_{i}") for i in range(3)], 1))
    q = np.stack([g(f"rot_{i}") for i in range(4)], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return xyz, op, sc, q


def quat_axes(q):
    w, x, y, z = q.T
    R = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                  np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                  np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)
    return R  # (N, 3, 3), 열 i = i번째 축


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    d = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    return d[d[:, 2] >= 0]  # 반구(축은 부호 무관)


def normal_peaks(nrm, w, k=4, sep_deg=30.0, kappa_deg=6.0):
    """축 분포의 봉우리. 반구 격자 점마다 cos 커널 밀도를 재고, 탐욕적으로 서로 sep_deg 이상 떨어진 최대점을 고른 뒤 근방 평균으로 다듬는다."""
    grid = fib_sphere(6000)
    dens = np.zeros(len(grid))
    for i in range(0, len(nrm), 10000):                     # 메모리 절약용 분할
        c = np.abs(nrm[i:i + 10000].astype(np.float32) @ grid.T.astype(np.float32))
        dens += w[i:i + 10000].astype(np.float32) @ np.exp((c - 1) / (1 - np.cos(np.deg2rad(kappa_deg))))
    peaks = []
    order = np.argsort(-dens)
    for j in order:
        if all(abs(grid[j] @ p["dir"]) < np.cos(np.deg2rad(sep_deg)) for p in peaks):
            m = np.abs(nrm @ grid[j]) > np.cos(np.deg2rad(10))
            s = np.sign(nrm[m] @ grid[j])[:, None]
            d = (w[m, None] * s * nrm[m]).sum(0); d /= np.linalg.norm(d)
            peaks.append(dict(dir=d, mass=float(w[m].sum() / w.sum())))
            if len(peaks) == k:
                break
    return peaks


def robust_spread(xyz, d):
    h = xyz @ d
    return float(np.subtract(*np.percentile(h, [95, 5])))


def sign_by_support(xyz, w_plane, axis, bins=200):
    """축 방향 높이에서 수평면 질량이 가장 큰 높이(지배 평면)를 찾고, 전체 Gaussian이 더 많이 놓인 쪽을 위로 본다."""
    h = xyz @ axis
    lo, hi = np.percentile(h, [1, 99])
    hist, e = np.histogram(h, bins=bins, range=(lo, hi), weights=w_plane)
    h0 = 0.5 * (e[np.argmax(hist)] + e[np.argmax(hist) + 1])
    above, below = float(np.mean(h > h0)), float(np.mean(h < h0))
    return (1.0 if above >= below else -1.0), dict(plane_h=float(h0), frac_above=above, frac_below=below)


def estimate(ply, a, rng):
    xyz, op, sc, q = load(ply, a.max_n, rng)
    keep = op > a.min_opacity
    lo, hi = np.percentile(xyz[keep], 2, 0), np.percentile(xyz[keep], 98, 0)
    keep &= np.all((xyz >= lo) & (xyz <= hi), 1)
    xyz, op, sc, q = xyz[keep], op[keep], sc[keep], q[keep]
    R = quat_axes(q)
    o = np.argsort(sc, 1)
    smin, smid = np.take_along_axis(sc, o[:, :1], 1)[:, 0], np.take_along_axis(sc, o[:, 1:2], 1)[:, 0]
    nrm = np.take_along_axis(R, o[:, None, :1], 2)[:, :, 0]
    flat = smid / np.maximum(smin, 1e-12) > a.flat_ratio
    w = op * flat  # 개수 가중(크기 가중은 거대한 배경 Gaussian에 끌림)
    sub = rng.choice(len(nrm), min(len(nrm), a.peak_n), replace=False)
    peaks = normal_peaks(nrm[sub], w[sub], k=a.n_peaks)
    for p in peaks:
        p["spread"] = robust_spread(xyz, p["dir"])
    ev, evec = np.linalg.eigh(np.cov((xyz - xyz.mean(0)).T))
    pca_min = evec[:, 0]
    hz = lambda d: [e for e in np.linalg.svd(np.eye(3) - np.outer(d, d))[0].T[:2]]  # noqa: E731
    thin = lambda d: robust_spread(xyz, d) / np.mean([robust_spread(xyz, e) for e in hz(d)])  # noqa: E731
    for p in peaks:
        p["thin"] = thin(p["dir"])
    def floor_sharpness(d):
        """d를 수직으로 볼 때 법선이 d와 10° 안인 평평한 Gaussian이 한 높이(바닥)에 얼마나 몰리는가:
        가장 많은 높이 구간(폭 = 장면 폭의 1%)에 든 비율. 기울어진 면이면 높이가 퍼져 작아진다."""
        m = (np.abs(nrm @ d) > np.cos(np.deg2rad(10))) & flat
        if m.sum() < 100:
            return 0.0
        h = xyz[m] @ d
        sp = robust_spread(xyz, d)
        hist, _ = np.histogram(h, bins=max(10, int((h.max() - h.min()) / (0.01 * sp))), weights=op[m])
        return float(hist.max() / op[m].sum())
    for p in peaks:
        p["sharp"] = floor_sharpness(p["dir"])
    snapped = None
    if a.axis_rule == "pca_sharp":
        # PCA 축과 그 근처(--snap_deg) 법선 봉우리 중, "평평한 Gaussian이 한 높이에 가장 날카롭게 몰리는" 방향을 고른다
        near = [p for p in peaks if abs(p["dir"] @ pca_min) > np.cos(np.deg2rad(a.snap_deg))]
        cands = near + [dict(dir=pca_min, sharp=floor_sharpness(pca_min), pca=True)]
        best = max(cands, key=lambda p: p["sharp"])
        snapped = None if best.get("pca") else best
        axis = best["dir"]
    elif a.axis_rule == "pca_snap":
        # 장면이 가장 얇게 퍼진 방향(PCA 최소 분산 축)으로 대략적인 수직을 잡고,
        # 그 근처(--snap_deg)에 평평한 Gaussian 법선 봉우리(바닥·지면·탁자)가 있으면 그중 장면이 가장 얇은 봉우리로 맞춘다.
        near = [p for p in peaks if abs(p["dir"] @ pca_min) > np.cos(np.deg2rad(a.snap_deg))]
        snapped = min(near, key=lambda p: p["thin"]) if near else None
        axis = snapped["dir"] if snapped else pca_min
    elif a.axis_rule == "mass":
        axis = peaks[0]["dir"]
    elif a.axis_rule == "spread":
        axis = min(peaks, key=lambda p: p["spread"])["dir"]
    else:
        axis = min(peaks[:3], key=lambda p: p["spread"])["dir"]
    w_plane = w * (np.abs(nrm @ axis) > np.cos(np.deg2rad(10)))
    s, info = sign_by_support(xyz, w_plane, axis)
    pca_info = dict(spread=robust_spread(xyz, pca_min), thin=thin(pca_min), sharp=floor_sharpness(pca_min), snapped=snapped is not None)
    return dict(up=(s * axis).tolist(), axis=axis.tolist(), pca_info=pca_info, peaks=[dict(dir=p["dir"].tolist(), mass=p["mass"], spread=p["spread"], thin=p["thin"], sharp=p["sharp"]) for p in peaks],
                pca_min=pca_min.tolist(), sign_info=info, n=int(len(xyz)), flat_frac=float(flat.mean()))


def evaluate_final(a):
    """알고리즘 파일(view_sampler.estimate_up_from_gaussians, --up auto와 같은 함수)로 11씬을 추정하고 기준 up과 비교."""
    from plyfile import PlyData
    res = {}
    for s in a.scenes.split(","):
        cfg = vs.load_scene_cfg(s)
        v = PlyData.read(cfg["ply"])["vertex"]
        g = lambda k: np.asarray(v[k]).astype(np.float64)  # noqa: E731
        xyz = np.stack([g("x"), g("y"), g("z")], 1)
        op = 1 / (1 + np.exp(-g("opacity")))
        sc = np.exp(np.stack([g(f"scale_{i}") for i in range(3)], 1))
        q = np.stack([g(f"rot_{i}") for i in range(4)], 1)
        kw = dict(cone_deg=a.cone_deg, cap_deg=a.cap_deg, band_deg=a.band_deg)
        Rr = vs.Renderer(cfg["ply"], cfg["sh_degree"]) if a.render_sign else None
        up, info = vs.estimate_up_from_gaussians(xyz, op, sc, q, seed=a.seed, renderer=Rr, **kw)
        del Rr
        ref = cfg["up"]
        res[s] = dict(up=up.tolist(), ref_up=ref.tolist(), ref_source=cfg.get("up_source", ""), err_deg=ang(up, ref),
                      axis_err_deg=min(ang(up, ref), 180 - ang(up, ref)), pca_err_deg=min(ang(info["pca"], ref), 180 - ang(info["pca"], ref)), info=info)
        vt = f" 공백 축/반대 {info['void_axis']:.2f}/{info['void_opposite']:.2f}" if info.get("void_axis") is not None else ""
        print(f"{s:10s} 오차 {res[s]['err_deg']:5.1f}° (축 {res[s]['axis_err_deg']:4.1f}°, PCA만 {res[s]['pca_err_deg']:4.1f}°) 위/아래 {info['frac_above']:.2f}/{info['frac_below']:.2f}{vt} 부호점수 {info['sign_score']:+.2f}", flush=True)
    return res


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default=",".join(ALL))
    ap.add_argument("--max_n", type=int, default=1500000)
    ap.add_argument("--peak_n", type=int, default=150000)
    ap.add_argument("--min_opacity", type=float, default=0.3)
    ap.add_argument("--flat_ratio", type=float, default=3.0)
    ap.add_argument("--n_peaks", type=int, default=8)
    ap.add_argument("--axis_rule", choices=["pca_snap", "pca_sharp", "mass", "spread", "mass_spread"], default="pca_snap")
    ap.add_argument("--snap_deg", type=float, default=40.0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results.json"))
    ap.add_argument("--final", action="store_true", help="알고리즘 파일의 최종 함수(PCA + 국소 맨해튼 보정)로 평가")
    ap.add_argument("--cone_deg", type=float, default=40.0); ap.add_argument("--cap_deg", type=float, default=10.0)
    ap.add_argument("--band_deg", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0, help="추정기 무작위 표본 seed(--final)")
    ap.add_argument("--render_sign", action="store_true", help="부호에 렌더 공백 단서도 쓴다(--up auto와 같음, GPU 필요)")
    a = ap.parse_args()
    if a.final:
        res = evaluate_final(a)
        Path(a.out).write_text(json.dumps(dict(args=vars(a), scenes=res), indent=1, ensure_ascii=False))
        return
    rng = np.random.default_rng(0)
    res = {}
    for s in a.scenes.split(","):
        cfg = vs.load_scene_cfg(s)
        r = estimate(cfg["ply"], a, rng)
        gt = cfg["up"]
        r["ref_up"] = gt.tolist(); r["ref_source"] = cfg.get("up_source", "")
        r["err_deg"] = ang(r["up"], gt); r["axis_err_deg"] = min(ang(r["axis"], gt), ang(-np.array(r["axis"]), gt))
        r["peak_axis_err_deg"] = [min(ang(p["dir"], gt), 180 - ang(p["dir"], gt)) for p in r["peaks"]]
        r["pca_axis_err_deg"] = min(ang(r["pca_min"], gt), 180 - ang(r["pca_min"], gt))
        res[s] = r
        print(f"{s:10s} 축오차 {r['axis_err_deg']:5.1f}°  최종 {r['err_deg']:6.1f}° | 봉우리(질량, 위치폭, GT와 각) " +
              " ".join(f"({p['mass']:.2f},{p['spread']:.1f},{e:.0f}°)" for p, e in zip(r["peaks"], r["peak_axis_err_deg"])) +
              f" | PCA {r['pca_axis_err_deg']:.0f}° | 위/아래 {r['sign_info']['frac_above']:.2f}/{r['sign_info']['frac_below']:.2f}", flush=True)
    Path(a.out).write_text(json.dumps(dict(args=vars(a), scenes=res), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
