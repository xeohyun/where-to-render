"""
make_poster_figs.py — 캡스톤디자인 요약보고서(판넬) 그림 (2026-11-06)
====================================================================
  poster_views.png   : counter 장면에서 학습 시점이 어디에 놓였는가 (위에서 본 지도 + 옆에서 본 높이), Ours vs Random(같은 높이대)
  poster_check.png   : 렌더 검사가 버리는 시점 유형(빈 화면·근접·흐림) vs 남기는 시점 — 실제 학습셋 이미지에서 발췌한 예시
  poster_result.png  : 10개 장면 위치 오차 (Ours vs 두 Random, 로그 축, seed 3개 평균)
  poster_map.png     : counter 3DGS 지도(점군, DC 색)를 비스듬히 본 그림
  poster_thumbs.png  : 제안 방법 학습 이미지 4장(고르게 뽑음)
  poster_ablation.png: 구성 요소별 제거 시 오차 배율(장면별 최댓값, 이전 버전 ablation)
  poster_up.png      : 위쪽 방향 자동 추정 오차 (설계 11장면 + 처음 보는 9모델)
  poster_nscale.png  : 학습 이미지 수 0.5–4배에 따른 위치 오차 (이전 버전 설정, counter·room·bicycle)
값과 이미지는 모두 기존 산출물(out/, results.json)에서 읽는다. 새 실험 없음.
env: ace (matplotlib, plyfile, PIL)
    python sampling/figures/make_poster_figs.py [--only views,check,result]
"""
import argparse
import json
import statistics as S
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RES = json.loads((ROOT / "sampling/mcmc/s2_aim_20261009/results.json").read_text())
for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    font_manager.fontManager.addfont(f)
# 2026-11-07: 판넬 글씨체(본문 Times 계열 serif)에 맞춰 그림도 영어 + Liberation Serif(Times New Roman과 글자 폭 호환)
plt.rcParams.update({"font.family": "Liberation Serif", "mathtext.fontset": "stix", "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False})
NAVY, RED, OURS, RAND, FULL, GRAY = "#002060", "#C00000", "#1f5fbf", "#d08a1a", "#8a6fa8", "#9aa3a8"


def load_poses(d):
    P = [np.loadtxt(p) for p in sorted((d / "train/poses").glob("*.pose.txt"))]
    return np.array([p[:3, 3] for p in P]), np.array([p[:3, 2] for p in P])


def fig_views(scene="counter"):
    from plyfile import PlyData
    cfg = json.loads((ROOT / "sampling/scenes.json").read_text())[scene]
    run = json.loads((ROOT / f"out/{scene}_s2_final3_s0/run_args.json").read_text())
    up = np.array(run["up"], float); up /= np.linalg.norm(up)
    v = PlyData.read(str(ROOT / cfg["ply"]))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1)
    op = 1 / (1 + np.exp(-np.asarray(v["opacity"])))
    xyz = xyz[op > 0.5]
    rng = np.random.default_rng(0)
    xyz = xyz[rng.choice(len(xyz), min(len(xyz), 250000), replace=False)]
    # 좌표계: e1, e2 = 수평, up = 수직. 바닥 p5 = 0, 구조물 p95 = 1 (알고리즘의 s와 같은 정의)
    a = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = a - (a @ up) * up; e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
    h = xyz @ up; floor, top = np.percentile(h, 5), np.percentile(h, 95)
    pts = np.stack([xyz @ e1, xyz @ e2, (h - floor) / (top - floor)], 1)
    keep = (pts[:, 2] > -2.0) & (pts[:, 2] < 3.0)
    pts = pts[keep]
    # 장면 중심부만(카메라 분포 기준 범위)
    # 비교 대상 3개(2026-11-12 사용자 요청): Ours, 같은 높이대·수평 무작위, 위치·회전 완전 무작위(random_full seed 0 = rk_random_full).
    # Ours와 같은 높이대 무작위는 기존 축 범위를 그대로 쓰고, 완전 무작위만 넓은 축으로 그린다.
    sets = {"Ours": (ROOT / f"out/{scene}_s2_final3_s0", OURS),
            "Random (same height)": (ROOT / f"out/{scene}_avr_random_upright_s0", RAND),
            "Random (full)": (ROOT / f"out/{scene}_rk_random_full", FULL)}
    cams = {}
    for k, (d, col) in sets.items():
        c, f = load_poses(d)
        cams[k] = (np.stack([c @ e1, c @ e2, (c @ up - floor) / (top - floor)], 1), np.stack([f @ e1, f @ e2, f @ up], 1), col)

    def box(cs, ylo=None, yhi=None):
        """카메라 분포로 정한 그림 범위: (가로 lo, hi, 세로 ylo, yhi)"""
        allc = np.concatenate(cs)
        lo, hi = np.percentile(allc[:, :2], 1, axis=0), np.percentile(allc[:, :2], 99, axis=0)
        pad = 0.10 * (hi - lo)
        if ylo is None:
            ylo, yhi = np.percentile(allc[:, 2], 1) - 0.15, np.percentile(allc[:, 2], 99) + 0.15
            yhi += 0.16 * (yhi - ylo)                 # 위쪽 글자 자리
        return lo - pad, hi + pad, ylo, yhi
    narrow = box([cams["Ours"][0], cams["Random (same height)"][0]], -0.25, 1.6)
    wide = box([cams["Random (full)"][0]])
    rng_of = {"Ours": narrow, "Random (same height)": narrow, "Random (full)": wide}

    def draw_side(ax, j, c, f, col, R, title=None):
        """옆에서 본 높이 (가로 = 첫 번째 수평축, 세로 = 정규화 높이 s)"""
        lo, hi, ylo, yhi = R
        m = (pts[:, 0] > lo[0]) & (pts[:, 0] < hi[0]) & (pts[:, 1] > lo[1]) & (pts[:, 1] < hi[1])
        ax.scatter(pts[m, 0], pts[m, 2], s=0.5, c="#6f7a80", alpha=0.45, linewidths=0, rasterized=True)
        pitch = np.degrees(np.arcsin(np.clip(f[:, 2], -1, 1)))
        dx = np.sign(f[:, 0] + 1e-9) * np.cos(np.radians(pitch)) * 0.05 * (hi[0] - lo[0])
        dz = np.sin(np.radians(pitch)) * 0.065 * (yhi - ylo)
        ax.quiver(c[:, 0], c[:, 2], dx, dz, color=col, angles="xy", scale_units="xy", scale=1, width=0.005, headwidth=3.5, alpha=0.9)
        ax.scatter(c[:, 0], c[:, 2], s=14, c=col, zorder=3)
        ax.axhline(0, color="#555", lw=1, ls="--"); ax.axhline(1, color="#555", lw=1, ls=":")
        if ylo < -0.3:                               # 알고리즘 높이 범위 밖(바닥 아래, 1.25 위)을 옅게 칠한다
            ax.axhspan(ylo, 0, color="#e9e9e9", zorder=0); ax.axhspan(1.25, yhi, color="#e9e9e9", zorder=0)
            out_ = ((c[:, 2] < 0) | (c[:, 2] > 1.25)).mean()
            ax.text(0.97, 0.03, f"{100 * out_:.0f}% below floor\nor above 1.25", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=21, color=col, fontweight="bold", bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(ylo, yhi); ax.set_xticks([])
        ax.set_yticks([0, 1.0]); ax.set_yticklabels(["floor", "top of\nstructure"], fontsize=21)
        if j == 0:
            ax.set_ylabel("Height (side view)", fontsize=24)
        med = np.median(pitch)
        p5, p95 = np.percentile(pitch, [5, 95])
        lab = (f"median pitch {med:.0f}°" if med < -5 else
               f"pitch {p5:.0f}° to {p95:.0f}°" if p95 - p5 > 20 else "pitch 0° (level)")
        ax.text(0.03, 0.97, lab, transform=ax.transAxes, va="top", fontsize=23, color=col, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
        if title:
            ax.set_title(title, fontsize=27, fontweight="bold", color=col)

    fig, axs = plt.subplots(2, 3, figsize=(19, 11.5), gridspec_kw=dict(height_ratios=[1.45, 1]))
    for j, (k, (c, f, col)) in enumerate(cams.items()):
        lo, hi = rng_of[k][:2]
        ax = axs[0, j]
        m = (pts[:, 0] > lo[0]) & (pts[:, 0] < hi[0]) & (pts[:, 1] > lo[1]) & (pts[:, 1] < hi[1])
        ax.scatter(pts[m, 0], pts[m, 1], s=0.5, c="#6f7a80", alpha=0.45, linewidths=0, rasterized=True)
        hf = f[:, :2] / (np.linalg.norm(f[:, :2], axis=1, keepdims=True) + 1e-9)
        L = 0.06 * np.linalg.norm(hi - lo)
        ax.quiver(c[:, 0], c[:, 1], hf[:, 0] * L, hf[:, 1] * L, color=col, angles="xy", scale_units="xy", scale=1,
                  width=0.004, headwidth=3.5, alpha=0.9)
        ax.scatter(c[:, 0], c[:, 1], s=14, c=col, zorder=3)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(k, fontsize=30, fontweight="bold", color=col)
        if j == 0:
            ax.text(-0.02, 0.5, "Top view", transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=24)
        draw_side(axs[1, j], j, c, f, col, rng_of[k])
    fig.tight_layout()
    fig.savefig(OUT / "poster_views.png", dpi=220)
    plt.close(fig)
    # 옆에서 본 높이만 (판넬 '후보 만들기' 칸용). 세 번째 칸만 축 범위가 넓다(판넬 캡션에 표기).
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 5.4))
    for j, (k, (c, f, col)) in enumerate(cams.items()):
        draw_side(axs[j], j, c, f, col, rng_of[k], title=k)
    fig.tight_layout()
    fig.savefig(OUT / "poster_views_side.png", dpi=220)
    plt.close(fig)


def fig_check():
    """렌더 검사 기준별 예시. 이미지와 수치는 실제 학습셋에서 지표를 계산해 골랐다(select_check_examples.json)."""
    sel = json.loads((OUT / "select_check_examples.json").read_text())
    fig, axs = plt.subplots(1, len(sel), figsize=(15, 4.3))
    met = {r["path"]: r for part in json.loads((OUT / "check_metrics_cache.json").read_text()) for r in part}
    for ax, s in zip(axs, sel):
        r = met[s["path"]]
        if s["keep"]:
            title, value = "Pass", f"void {100 * r['void_frac']:.0f}%, depth spread {r['relstd']:.2f}"
        elif r["void_frac"] > 0.05:
            title, value = "Reject: void", f"void pixels {100 * r['void_frac']:.0f}% (pass if ≤ 5%)"
        else:
            title, value = "Reject: blurry", f"depth spread {r['relstd']:.2f} (pass if ≤ 0.30)"
        im = Image.open(ROOT / s["path"]).convert("RGB")
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        ok = s["keep"]
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor("#2e7d32" if ok else RED); sp.set_linewidth(6)
        ax.set_title(title, fontsize=27, fontweight="bold", color="#2e7d32" if ok else RED, pad=16)   # 그림과 글 사이 여백
        ax.set_xlabel(value, fontsize=22, labelpad=14)
    fig.tight_layout()
    fig.savefig(OUT / "poster_check.png", dpi=200)
    plt.close(fig)


def mean(s, c, key="med_trans"):
    v = [RES.get(f"{s}|{c}|s{k}|test") for k in range(3)]
    return S.mean(x[key] for x in v)


def fig_result():
    sc = ["counter", "bonsai", "kitchen", "room", "stump", "bicycle", "garden", "tnt_truck", "tnt_train", "livingroom"]
    lab = dict(tnt_truck="T&T\ntruck", tnt_train="T&T\ntrain", livingroom="Living\nRoom")
    x = np.arange(len(sc)); w = 0.27
    fig, ax = plt.subplots(figsize=(13, 5.6))
    for i, (c, col, name) in enumerate([("final3", OURS, "Ours"), ("random_upright", RAND, "Random (same height band, level)"), ("random_full", FULL, "Random (full)")]):
        ax.bar(x + (i - 1) * w, [mean(s, c) for s in sc], w, color=col, label=name)
    ax.set_yscale("log"); ax.set_ylabel("Translation error (log)", fontsize=20)
    ax.set_xticks(x); ax.set_xticklabels([lab.get(s, s) for s in sc], fontsize=19); ax.tick_params(axis="y", labelsize=17)
    ax.set_ylim(top=ax.get_ylim()[1] * 6)
    ax.axvspan(5.5, 8.5, color="#eef1f6", zorder=0)
    ax.text(7, ax.get_ylim()[1] * 0.45, "Held-out scenes", ha="center", fontsize=20, color=NAVY, fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=20)
    ax.grid(axis="y", alpha=0.25, which="major")
    fig.tight_layout()
    fig.savefig(OUT / "poster_result.png", dpi=220)
    plt.close(fig)


def fig_map(scene="counter"):
    """3D Gaussian Splatting 지도(점군)를 비스듬히 본 그림. 색은 Gaussian의 DC 색."""
    from plyfile import PlyData
    cfg = json.loads((ROOT / "sampling/scenes.json").read_text())[scene]
    run = json.loads((ROOT / f"out/{scene}_s2_final3_s0/run_args.json").read_text())
    up = np.array(run["up"], float); up /= np.linalg.norm(up)
    v = PlyData.read(str(ROOT / cfg["ply"]))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1)
    op = 1 / (1 + np.exp(-np.asarray(v["opacity"])))
    rgb = np.clip(0.5 + 0.28209479 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1), 0, 1)
    m = op > 0.3
    xyz, rgb = xyz[m], rgb[m]
    rng = np.random.default_rng(1)
    k = rng.choice(len(xyz), min(len(xyz), 900000), replace=False)
    xyz, rgb = xyz[k], rgb[k]
    c = np.median(xyz, 0); xyz = xyz - c
    r = np.percentile(np.linalg.norm(xyz, axis=1), 70); keep = np.linalg.norm(xyz, axis=1) < r
    xyz, rgb = xyz[keep], rgb[keep]
    a = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = a - (a @ up) * up; e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
    th, el = np.radians(35), np.radians(35)                      # 비스듬히 내려다보는 시점
    fwd = np.cos(el) * (np.cos(th) * e1 + np.sin(th) * e2) - np.sin(el) * up
    right = np.cross(fwd, up); right /= np.linalg.norm(right); vup = np.cross(right, fwd)
    depth = xyz @ fwd; o = np.argsort(-depth)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter((xyz @ right)[o], (xyz @ vup)[o], s=2.2, c=rgb[o], linewidths=0, rasterized=True)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(OUT / "poster_map.png", dpi=240)
    plt.close(fig)


def fig_thumbs(scene="counter"):
    """제안 방법 학습 이미지 4장을 2x2로(목표 칸의 흐름도 삽화용). 렌더 검사 통과 프레임 중 깊이 퍼짐 하위 8장에서
    장면이 넓게 담긴 4장을 사람이 골랐다(select_check_examples.py의 후보와 같음)."""
    d = ROOT / f"out/{scene}_s2_final3_s0/train/rgb"
    pick = [d / f"frame_{i:06d}.color.png" for i in (69, 31, 54, 44)]
    ims = [Image.open(p).convert("RGB").resize((600, 400)) for p in pick]
    g = Image.new("RGB", (1210, 810), "white")
    for i, im in enumerate(ims):
        g.paste(im, ((i % 2) * 610, (i // 2) * 410))
    g.save(OUT / "poster_thumbs.png")


def fig_ablation():
    """구성 요소 하나를 뺐을 때 위치 오차 배율(가장 크게 나빠진 장면). 값은 이전 버전(6층 고정, GT up), seed 2–3개."""
    rows = [("w/o height range & aimed pitch", "counter", 54.5), ("w/o pitch fallback", "stump", 21.1),
            ("2 views/position → all passing", "room", 10.8), ("density → area weighting", "bicycle", 7.0),
            ("w/o depth-spread check", "bicycle", 3.4)]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[2] for r in rows], color=RED, height=0.62)
    ax.set_xscale("log"); ax.set_xlim(1, 8000)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=24)
    for yy, r in zip(y, rows):
        ax.text(r[2] * 1.12, yy, f"×{r[2]:g} ({r[1]})", va="center", fontsize=23, fontweight="bold", color=RED)
    ax.axvline(1, color="#333", lw=1)
    ax.set_xticks([1, 10, 100]); ax.set_xticklabels(["×1", "×10", "×100"], fontsize=20)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(OUT / "poster_ablation.png", dpi=220)
    plt.close(fig)


def fig_up():
    """위쪽 방향 자동 추정 오차. 값: sampling/up_estimation (발표 fig_up과 같은 데이터)."""
    a = json.loads((ROOT / "sampling/up_estimation/results_up_auto_final.json").read_text())["scenes"]
    h = json.loads((ROOT / "sampling/up_estimation/results_heldout.json").read_text())
    va, vh = [r["err_deg"] for r in a.values()], [r["err_deg"] for r in h.values()]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(range(len(va)), va, color=OURS, width=0.75, label=f"Development scenes ({len(va)})")
    ax.bar(range(len(va) + 1, len(va) + 1 + len(vh)), vh, color="#8fb3ea", width=0.75, label=f"Held-out maps ({len(vh)})")
    med = S.median(va + vh)
    ax.axhline(med, color=RED, ls="--", lw=2)
    ax.text(len(va) + len(vh) + 0.6, 7.3, f"- - overall median {med:.1f}°", color=RED, fontsize=18, fontweight="bold", ha="right")
    ax.set_xticks([]); ax.set_ylim(0, 11.5); ax.set_ylabel("Up-vector error (°)", fontsize=19)
    ax.tick_params(axis="y", labelsize=17)
    ax.legend(frameon=False, fontsize=17, loc="upper right", ncol=1)
    fig.tight_layout()
    fig.savefig(OUT / "poster_up.png", dpi=220)
    plt.close(fig)


def fig_nscale():
    """학습 이미지 수를 0.5–4배로 바꿨을 때 위치 오차 (이전 버전 aimDrf_v11 설정, seed 3개 평균). 실선 = 제안, 점선 = 무작위(같은 높이대)."""
    xs = [0.5, 1, 2, 4]
    fig, ax = plt.subplots(figsize=(8, 5.4))
    for s, col in zip(["counter", "room", "bicycle"], ["#1f5fbf", "#2e9e5b", "#7a3fb0"]):
        ours = [mean(s, "aimDrf_v11@x0.5"), mean(s, "aimDrf_rs"), mean(s, "aimDrf_v11@x2"), mean(s, "aimDrf_v11@x4")]
        rnd = [mean(s, "random_upright@x0.5"), mean(s, "random_upright"), mean(s, "random_upright@x2"), mean(s, "random_upright@x4")]
        ax.plot(xs, ours, "o-", color=col, lw=3.5, ms=10)
        ax.plot(xs, rnd, "s--", color=col, lw=2, ms=7, alpha=0.45)
        ax.text(4.35, rnd[-1], s, color=col, fontsize=20, fontweight="bold", va="center")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlim(0.42, 7)
    ax.set_xticks(xs); ax.set_xticklabels(["0.5×", "1×", "2×", "4×"], fontsize=20)
    ax.minorticks_off(); ax.tick_params(axis="y", labelsize=17)
    ax.set_ylabel("Translation error (log)", fontsize=20); ax.set_xlabel("Number of training images", fontsize=20)
    ax.plot([], [], "k-", lw=3.5, label="Ours"); ax.plot([], [], "k--", lw=2, alpha=0.45, label="Random (same height band)")
    ax.legend(frameon=False, fontsize=18, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "poster_nscale.png", dpi=220)
    plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="views,check,result,map,thumbs,ablation")
    for k in ap.parse_args().only.split(","):
        globals()[f"fig_{k}"]()
        print("saved", k)
