"""
view_sampler.py — 3DGS 지도(.ply) 하나만으로 ACE 학습 뷰를 고르는 알고리즘 (2026-10-08 정리본)
=============================================================================================
알고리즘 수정은 이 파일 하나에서만 한다(2026-10-02 사용자 요청). 연구·진단 도구(landscape/report)는
view_sampler_research.py, 고정 평가셋은 eval_protocol.py, 씬 설정은 scenes.json.

입력 / 출력
  입력 : 3DGS .ply 하나 + 카메라(해상도, 초점). 실제 사진·실제 카메라 포즈·COLMAP은 쓰지 않는다.
         up 방향은 --up auto(최종 설정 기본)면 지도 Gaussian만으로 추정하고, --up config면 scenes.json 값을 쓴다.
  출력 : ACE 학습셋 <out_dir>/train/{rgb,poses,calibration} — 정확히 --n_train장.
         test는 만들지 않는다. --link_test test 로 고정 평가셋(datasets/eval_fixed/<scene>/test)을 연결한다.

최종 설정 (2026-10-28, `run --preset final`; 근거: EXPERIMENT_LOG 2026-10-09~11, 2026-10-24~28, sampling/mcmc/s2_aim_20261009)
  --s1 pillar --levels 0.1,0.25,0.5,0.75,1.0,1.25 --level_mode uniform --s2 aim --aim_weight density --views_per_point 2 --n_candidates 2000
  --max_relstd 0.30 --s3_render_px 160 --pitch_fallback_step 10 --max_struct 2000000 --relax_if_short 4 --up auto
  = "넓게 제안하고, 지도가 완전하게 보이는 시점만 남긴다": 기둥 위치마다 높이 6개를 s U[0.1, 1.25]에서 무작위로 뽑고,
    방향마다 촘촘히 복원된 구조를 겨누는 pitch를 쓴 뒤, 160px 렌더 완전도 검사(실패 시 10도씩 하향)로 거른다.
  (2026-10-28: 피드백 3에서 6개 고정 층을 같은 범위의 무작위 높이로 바꿔도 같아 교체(--level_mode uniform). pitch까지
   무작위(--s2 random)로 바꾼 final2는 품질 낮은 held-out 지도(T&T train)에서 실패해 aim pitch를 유지.)
  (2026-10-25: --up auto와 --relax_if_short 4(v1.1 안전장치, 목표 장수를 채우면 미작동)를 추가. 10씬 x 3 seed에서
   추정 up 결과가 30/30 두 random 기준선을 이김 — sampling/up_estimation/compare_ace.log)
  = 기둥 축 x 높이 층, 방향마다 촘촘히 복원된 구조물의 고도 중심을 겨누는 pitch, 160px 실제 렌더로 S3 검사,
    탈락 방향은 10도씩 내려 처음 깨끗한 pitch, point당 2장, ray 깊이 퍼짐 필터.
  Mip360 6씬(실사진 test) + SuperSplat 2씬에서 seed 3개 평균 이동 오차가 random(upright/full)보다 모두 작음.
  아래 S0~S6 설명은 기본값(--preset 없음 = 2026-10-08 정리본 동작) 기준이며, 최종 설정은 S1/S2/S3/S5를 위 옵션으로 바꾼다.

알고리즘 `run` — 한 줄 요약
  "물체 근처 빈 공간의 point에서 수평으로 한 바퀴 돌며 찍는다. 검은 화면·근접샷이 나오는 방향은 빼고,
   장면 곳곳을 여러 point가 고르게 보도록 point 순서를 정한 뒤, 렌더해서 불량 사진을 버리며 목표 장수까지 채운다."

  S0 장면 정리   불투명도 > 0.2인 Gaussian만, 떠다니는 점(SOR) 제거. up 방향 높이의 p5를 바닥으로 두고 바닥 Gaussian을 뺀
                 나머지를 '구조물'로 쓴다. 카메라 높이 밴드 = 구조물 높이 p15~p45.
  S1 위치 후보   Gaussian 하나하나를 반지름 2.5*max(scale)인 공으로 보고, 구조물 근처에서 무작위 방향으로 한 걸음 나간 점 중
                 가장 가까운 공 표면까지 거리가 (3*점 간격, 1.5*밴드 폭) 사이이고 높이가 밴드 안인 점 600개.
  S2 높이/pitch  S1 높이 그대로, pitch 0(수평), roll 0.
  S3 방향 검사   point마다 수평 360도를 K=ceil(360/(FOV*(1-0.5)))개 방향(겹침 50%)으로 나누고, 2배 조밀하게 Gaussian 중심을
                 투영해 8x6 칸 검사: 빈 칸 비율 <= 0.15, 칸별 최근접 깊이의 p90 >= 3*밴드 폭(근접샷 아님).
                 합격 방향의 가장 긴 연속 구간(부채꼴)이 4장 이상인 point만 채택하고 그 구간만 찍는다.
  S4 point 순서  sum_g w_g * min(그 Gaussian을 본 point 수, 3)을 최대화하는 lazy greedy(같은 곳을 여러 위치에서 보게).
                 w_g = 불투명도 * (큰 두 scale의 곱).
  S5 사진 검사   실제 렌더 후 프레임 단위로 버림: 미재구성(누적 불투명도 < 0.5) 비율 > 0.05, 근접(깊이 < 0.15*근접 문턱)
                 비율 > 0.20, 선명도(Laplacian 분산) < 0.25 * (처음 48장의 중앙값).
  S6 채우기      S4 순서대로 렌더→검사→저장. 한 point가 전부 걸러지면 다음 순위 point로 넘어가 정확히 --n_train장을 채운다.

기준선 (같은 장면 정리·같은 카메라·같은 장수·같은 렌더러로 비교. 알고리즘 아님)
  random --mode full    : bbox(p1~p99 +10%) 안 균등 위치 중 Gaussian 내부가 아닌 곳 + 완전 무작위 회전(roll 포함).
                          = 2026-10-02 순위 검증 v2의 random_full(구 baseline-random, 결과 동일).
  random --mode upright : 알고리즘과 같은 S1 위치 후보 분포에서 무작위 위치 + 무작위 yaw, pitch 0, roll 0, 검사 없음.
                          "같은 후보에서 고르되 S3~S6 없이 무작위로" — 알고리즘의 선택 단계가 하는 몫을 잰다.
  random --mode pillar  : 최종 알고리즘(final3)과 같은 S1 위치 후보(기둥 축 x 높이 U[0.1, 1.25])에서 무작위 위치 + 완전 무작위
                          회전(roll 포함), 검사 없음. "위치 후보는 알고리즘 것, 방향은 아예 무작위"(2026-10-29 사용자 요청 비교군).
                          --pillar_dir upright: 같은 위치 + 시선 방향은 구면 균등 무작위, roll 0(수평선 유지).
  human                 : 사람이 picker(eval_protocol.py picker --purpose train)로 찍은 포즈 JSON을 같은 렌더러로 렌더.
                          실사진이 없는 SuperSplat 씬에서 Mip360의 '실제 촬영 궤적'에 해당하는 사람 기준선.

알려진 한계 (근거: agent-logs/claude/EXPERIMENT_LOG.md)
  * 높이 밴드(p15~45)가 Mip360 실제 카메라보다 낮다. 실제 사진은 밴드 위에서 약 25~30도 내려다본다(2026-10-01, CONFIRMED).
  * 실제 카메라 조건(FOV 52~58도, 109~272장)에서 Mip360 실사진 test 오차는 dm~수 m(2026-10-02 순위 검증 v2).
    단 같은 조건의 random(full)보다는 6/6씬에서 낫다.
  * ray 깊이 퍼짐(rel_std)은 계산·기록만 하고 필터에는 쓰지 않는다(야외 먼 배경에서 편향, 미해결).
  * 렌더 근거리/원거리 클리핑은 common.build_render_camera 기본값(znear 0.01, zfar 100)을 따른다.

버전 이력
  step0d_ply_only.py v1~v4 → step0f(수평 스윕) → step0g_covered_sweep.py(S3~S6) → view_sampler.py(2026-10-02 단계 함수화)
  → 2026-10-08 정리본: landscape/report 분리, 사용하지 않던 플래그(--s1/--s2/--n_test/--max_relstd) 제거,
    정확히 n_train장에서 멈춤(이전엔 마지막 point의 초과분을 사후 삭제), 배경색 기본값을 scenes.json(bg_color)에서 읽음,
    random --mode upright / human 기준선 추가, --focal_px / --link_test 추가.
    정리 전 파일: sampling/mcmc/vs_cleanup_20261008/view_sampler_before_cleanup.py (같은 인자에서 결과 동일함을 확인).

사용 (env 3dgs_mcmc, GPU)
  python sampling/view_sampler.py run    --scene livingroom --out_dir out/livingroom_vs --n_train 300 --link_test test
  python sampling/view_sampler.py random --scene livingroom --out_dir out/livingroom_rand_up --mode upright --n_train 300 --link_test test
  python sampling/view_sampler.py human  --scene livingroom --out_dir out/livingroom_human --poses livingroom_train_poses.json --link_test test
  실제 카메라와 맞출 때: --width 3115 --height 2076 --focal_px 3192.78
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
GS_REPO = os.environ.get("GS_REPO", str(ROOT.parent / "3dgs-mcmc"))  # 3DGS-MCMC 저장소(렌더러)
EVAL_ROOT = ROOT / "datasets/eval_fixed"
VERSION = "view_sampler 2026-10-28"

UP_PRESETS = {
    "+x": [1.0, 0.0, 0.0], "-x": [-1.0, 0.0, 0.0], "+y": [0.0, 1.0, 0.0],
    "-y": [0.0, -1.0, 0.0], "+z": [0.0, 0.0, 1.0], "-z": [0.0, 0.0, -1.0],
}

# S0/S1 기하 파라미터 기본값(알고리즘·기준선·평가 도구가 모두 같은 값을 쓴다)
GEOMETRY_DEFAULTS = dict(min_opacity=0.2, bbox_percentile=1.0, margin_ratio=0.1, ground_percentile=5.0, ground_margin_ratio=0.05,
                         cam_height_pct_lo=15.0, cam_height_pct_hi=45.0, min_offset_mult=3.0, max_offset_band_mult=1.5,
                         k_sigma=2.5, min_target_dist_mult=3.0, max_target_dist_ratio=0.9, near_dist_ratio=0.15)


# ============================================================================= 공통: 씬, 카메라, 렌더러
def load_scene_cfg(name):
    cfg = json.loads((HERE / "scenes.json").read_text())[name]
    ply = Path(cfg["ply"])
    cfg["ply"] = str(ply if ply.is_absolute() else ROOT / ply)
    s = cfg["up_axis"]
    v = np.array(UP_PRESETS[s] if s in UP_PRESETS else [float(x) for x in s.split(",")], dtype=np.float64)
    cfg["up"] = v / np.linalg.norm(v)
    cfg.setdefault("bg_color", "black")
    return cfg


def geometry_namespace(seed=0, **over):
    """eval_protocol.py 등 CLI 밖에서 S0을 부를 때 쓰는 기본 파라미터 묶음."""
    return argparse.Namespace(**{**GEOMETRY_DEFAULTS, "k_neighbors": 16, "oversample": 3, "seed": seed, **over})


def focal_for(width, fov_deg):
    return (width / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)


def resolve_camera(a):
    """(W, H, focal_px). --focal_px가 있으면 그대로, 없으면 --fov_deg(수평)에서 계산."""
    return a.width, a.height, (a.focal_px if a.focal_px else focal_for(a.width, a.fov_deg))


def background(a, cfg):
    name = a.bg_color or cfg["bg_color"]
    return name, (1.0 if name == "white" else 0.0)


def sweep_forwards(up, n, start_yaw, pitch_rad):
    """up에 수직인 평면에서 n개 방향을 균등 간격으로(+ 상수 pitch)."""
    a = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = a - (a @ up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    yaws = start_yaw + 2 * np.pi * np.arange(n) / n
    f = np.cos(pitch_rad) * (np.cos(yaws)[:, None] * e1 + np.sin(yaws)[:, None] * e2) + np.sin(pitch_rad) * up
    return f / np.linalg.norm(f, axis=1, keepdims=True)


class Renderer:
    """3DGS 렌더 래퍼. color()는 uint8 이미지, stats()는 [z, z^2, 1]을 Gaussian별 색으로 넣은 한 번의 렌더로
    누적 불투명도 alpha, 기대 깊이 E[z], ray 종료 깊이 분산 Var[z]를 픽셀별로 준다(.ply만으로 계산되는 ray 분포)."""

    def __init__(self, ply, sh_degree, device="cuda"):
        sys.path.insert(0, GS_REPO)
        import torch
        from gaussian_renderer import render
        from scene.gaussian_model import GaussianModel
        from arguments import PipelineParams
        self.torch, self.render_fn, self.device = torch, render, device
        self.pipe = PipelineParams(argparse.ArgumentParser())
        self.g = GaussianModel(sh_degree=sh_degree)
        self.g.load_ply(ply)

    def camera(self, R_w2c, t_w2c, focal, W, H):
        from common import build_render_camera
        return build_render_camera(R_w2c, t_w2c, focal, W, H, device=self.device)

    def color(self, cam, bg):
        from common import render_gaussians
        bgt = self.torch.tensor([bg] * 3, dtype=self.torch.float32, device=self.device)
        return np.asarray(render_gaussians(self.g, cam, self.pipe, bgt, self.render_fn))

    def stats(self, cam, R_w2c, t_w2c):
        torch = self.torch
        with torch.no_grad():
            xyz = self.g.get_xyz
            R = torch.as_tensor(R_w2c, dtype=xyz.dtype, device=xyz.device)
            t = torch.as_tensor(t_w2c, dtype=xyz.dtype, device=xyz.device)
            z = (xyz @ R[2] + t[2]).unsqueeze(1)
            o = self.render_fn(cam, self.g, self.pipe, torch.zeros(3, device=xyz.device),
                               override_color=torch.cat([z, z * z, torch.ones_like(z)], 1))["render"]
            a = o[2]
            ez = o[0] / a.clamp(min=1e-6)
            var = (o[1] / a.clamp(min=1e-6) - ez ** 2).clamp(min=0)
        return a.cpu().numpy(), ez.cpu().numpy(), var.cpu().numpy()


def frame_metrics(alpha, ez, var, near_dist):
    """렌더 기준 프레임 지표. void: alpha<0.5 비율. near: 깊이<near_dist 비율. relstd: ray 깊이 퍼짐 sqrt(Var)/E[z]의 중앙값."""
    cov = alpha >= 0.5
    if not cov.any():
        return dict(void_frac=1.0, near_frac=0.0, relstd=1.0, median_depth=0.0)
    rs = np.sqrt(var[cov]) / np.maximum(ez[cov], 1e-6)
    return dict(void_frac=float(1.0 - cov.mean()), near_frac=float((cov & (ez < near_dist)).mean()),
                relstd=float(np.median(rs)), median_depth=float(np.median(ez[cov])))


# ============================================================================= S0 장면 정리
def s0_scene_geometry(gaussians_xyz, opacity, scales, up, a):
    """opacity/SOR 필터, 바닥 제외, 높이 기준. 결과는 dict(numpy). a: GEOMETRY_DEFAULTS 키를 가진 namespace."""
    from scipy.spatial import cKDTree
    from common import remove_floater_points, gaussian_effective_radius, local_point_scale
    m = opacity > a.min_opacity
    clean, keep = remove_floater_points(gaussians_xyz[m], k=20, std_ratio=2.0)
    cop, csc = opacity[m][keep], scales[m][keep]
    h = clean @ up
    vr = np.subtract(*np.percentile(h, [99.0, 1.0]))
    floor = float(np.percentile(h, a.ground_percentile))
    sm = h > floor + a.ground_margin_ratio * vr
    sx, ssc, sop, sh = clean[sm], csc[sm], cop[sm], h[sm]
    ms = getattr(a, "max_struct", None)
    if ms and len(sx) > ms:  # 초대형 씬(Street 1,300만 개) 속도용 고정 표본. 200만 개 이하 씬에는 영향 없음
        k = np.sort(np.random.default_rng(12345).choice(len(sx), int(ms), replace=False))
        sx, ssc, sop, sh = sx[k], ssc[k], sop[k], sh[k]
    top2 = np.sort(ssc, axis=1)[:, 1:]
    lo = np.percentile(clean, a.bbox_percentile, axis=0)
    hi = np.percentile(clean, 100 - a.bbox_percentile, axis=0)
    span = hi - lo
    lo, hi = lo - span * a.margin_ratio, hi + span * a.margin_ratio
    blo, bhi = np.percentile(sh, [a.cam_height_pct_lo, a.cam_height_pct_hi])
    rng = np.random.default_rng(a.seed)
    G = dict(clean=clean, struct=sx, struct_radii=gaussian_effective_radius(ssc, k_sigma=a.k_sigma),
             struct_weight=sop * top2[:, 0] * top2[:, 1], tree=cKDTree(sx), up=up,
             floor=floor, top=float(np.percentile(h, 95.0)), lo=lo, hi=hi,
             band=(float(blo), float(bhi)), band_range=float(bhi - blo), scene_extent=float(np.linalg.norm(hi - lo)),
             point_scale=local_point_scale(sx, k=8, rng=rng))
    G["min_offset"] = G["point_scale"] * a.min_offset_mult
    G["max_offset"] = G["band_range"] * a.max_offset_band_mult
    G["min_target_dist"] = a.min_target_dist_mult * G["band_range"]
    G["max_target_dist"] = a.max_target_dist_ratio * G["scene_extent"]
    G["near_dist"] = a.near_dist_ratio * G["min_target_dist"]
    return G


def estimate_up_from_gaussians(xyz, opacity, scales, quats, seed=0, max_n=1500000, min_opacity=0.3, flat_ratio=3.0,
                               cone_deg=40.0, cap_deg=10.0, band_deg=5.0, grid_n=8000, sub_n=150000, renderer=None, n_void=48):
    """S(-1) up 자동 추정(2026-10-24): 카메라·사진 없이 Gaussian만으로 장면의 up(중력 반대 방향)을 찾는다.
    1) 대략적인 수직: 불투명 Gaussian 위치의 최소 분산 축(PCA). 사람이 찍은 장면은 수평으로 넓고 수직으로 얇다.
    2) 정밀 보정(맨해튼 점수): 평평한 Gaussian(가장 짧은 축이 중간 축의 1/flat_ratio 이하)의 법선 = 가장 짧은 축.
       후보 방향 u마다 "법선이 u와 평행(바닥·지면·탁자, cap_deg 안)한 질량" x "법선이 u와 직각(벽·기둥·줄기, band_deg 안)한 질량"을
       균일 분포 대비 비율로 재고, PCA 축에서 cone_deg 안의 최대점을 축으로 쓴다.
    3) 부호: 세 단서 값의 합(각각 [-1, 1] 범위, +면 축 방향이 위). 단서마다 틀리는 장면이 달라 서로 보완한다
       (11씬 x 파라미터 6설정에서 다수결은 1건, 단일 단서는 1~30건 틀렸고 합은 0건 — 같은 씬으로 고른 규칙이라 held-out 검증은 없음).
       - 지배 평면 지지(Sa): 평행 법선 Gaussian이 가장 많이 모인 높이(지배 평면)보다 위에 있는 Gaussian 비율 - 아래 비율
         (물체는 바닥 위에 놓이고 바닥 아래는 관측되지 않는다). 가장 큰 평면이 탁자 윗면이면 틀릴 수 있다.
       - 수평면 순위(Se): 평행 법선 Gaussian들이 전체 높이 분포에서 평균적으로 아래쪽에 있는 정도(1 - 2 x 높이 백분위의 평균).
         탁자 위 장면(counter)처럼 수평면이 높은 곳에 몰리면 틀릴 수 있다.
       - 공백(void, renderer가 있을 때): 장면 중간 높이에서 축 방향과 반대 방향을 64px로 렌더한 빈 화면 비율의 차
         (지도는 촬영된 곳만 복원하고, 사람은 천장·하늘을 거의 찍지 않는다). 실내에서 강하고, 천장까지 복원된 씬·나무가 많은 야외에서 약하다.
       renderer가 없으면 공백 단서는 0이다.
    quats: (w, x, y, z) 정규화 쿼터니언, scales: 실제 크기(exp 적용 후), opacity: sigmoid 적용 후."""
    rng = np.random.default_rng(seed)
    n = len(xyz)
    if n > max_n:
        k = np.sort(rng.choice(n, max_n, replace=False))
        xyz, opacity, scales, quats = xyz[k], opacity[k], scales[k], quats[k]
    xyz = np.asarray(xyz, np.float64)
    keep = opacity > min_opacity
    lo, hi = np.percentile(xyz[keep], 2, 0), np.percentile(xyz[keep], 98, 0)
    keep &= np.all((xyz >= lo) & (xyz <= hi), 1)
    xyz, opacity, scales, q = xyz[keep], opacity[keep], scales[keep], quats[keep] / np.linalg.norm(quats[keep], axis=1, keepdims=True)
    _, evec = np.linalg.eigh(np.cov((xyz - xyz.mean(0)).T))
    pca = evec[:, 0]
    w_, x_, y_, z_ = q.T
    R = np.stack([np.stack([1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - w_ * z_), 2 * (x_ * z_ + w_ * y_)], -1),
                  np.stack([2 * (x_ * y_ + w_ * z_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - w_ * x_)], -1),
                  np.stack([2 * (x_ * z_ - w_ * y_), 2 * (y_ * z_ + w_ * x_), 1 - 2 * (x_ * x_ + y_ * y_)], -1)], 1)
    o = np.argsort(scales, 1)
    smin = np.take_along_axis(scales, o[:, :1], 1)[:, 0]; smid = np.take_along_axis(scales, o[:, 1:2], 1)[:, 0]
    nrm = np.take_along_axis(R, o[:, None, :1], 2)[:, :, 0]
    flat = smid / np.maximum(smin, 1e-12) > flat_ratio
    w = opacity * flat
    sub = rng.choice(len(nrm), min(len(nrm), sub_n), replace=False)
    ns, ws = nrm[sub].astype(np.float32), w[sub].astype(np.float32)
    i = np.arange(grid_n) + 0.5
    phi, th = np.arccos(1 - 2 * i / grid_n), np.pi * (1 + 5 ** 0.5) * i
    grid = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    grid = grid[grid[:, 2] >= 0].astype(np.float32)
    cap, band = np.zeros(len(grid)), np.zeros(len(grid))
    for j in range(0, len(ns), 10000):
        c = np.abs(ns[j:j + 10000] @ grid.T)
        cap += ws[j:j + 10000] @ (c > np.cos(np.deg2rad(cap_deg)))
        band += ws[j:j + 10000] @ (c < np.sin(np.deg2rad(band_deg)))
    tot = max(float(ws.sum()), 1e-9)
    score = (cap / (tot * (1 - np.cos(np.deg2rad(cap_deg))))) * (band / (tot * np.sin(np.deg2rad(band_deg))))
    score[np.abs(grid @ pca) < np.cos(np.deg2rad(cone_deg))] = -1
    axis = grid[int(np.argmax(score))].astype(np.float64)
    axis /= np.linalg.norm(axis)
    h = xyz @ axis
    pm = flat & (np.abs(nrm @ axis) > np.cos(np.deg2rad(cap_deg)))
    hist, e = np.histogram(h[pm], bins=200, range=tuple(np.percentile(h, [1, 99])), weights=opacity[pm])
    h0 = 0.5 * (e[np.argmax(hist)] + e[np.argmax(hist) + 1])
    above, below = float(np.mean(h > h0)), float(np.mean(h < h0))
    void_up = void_down = None
    if renderer is not None:
        from common import look_at_rotation
        hz = xyz - np.outer(h, axis)
        lo_h, hi_h = np.percentile(h, [40, 60])
        ref = np.cross(axis, [1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.cross(axis, [0.0, 1.0, 0.0])
        Wv = 64; fv = focal_for(Wv, 90.0); vu, vd = [], []
        for _ in range(n_void):
            p = hz[rng.integers(len(hz))] + rng.uniform(lo_h, hi_h) * axis
            for d, acc in ((axis, vu), (-axis, vd)):
                Rw = look_at_rotation(d + 1e-3 * ref, ref); tw = -Rw @ p
                al, _, _ = renderer.stats(renderer.camera(Rw, tw, fv, Wv, Wv), Rw, tw)
                acc.append(float((al < 0.5).mean()))
        void_up, void_down = float(np.mean(vu)), float(np.mean(vd))
    hs = np.sort(h)
    se = float(np.average(1 - 2 * np.searchsorted(hs, h[pm]) / len(hs), weights=opacity[pm])) if pm.any() else 0.0
    sa = above - below
    sv = (void_up - void_down) if void_up is not None else 0.0
    sign_score = float(sa + se + sv)
    up = axis if sign_score >= 0 else -axis
    return up, dict(pca=pca.tolist(), axis=axis.tolist(), plane_h=float(h0), frac_above=above, frac_below=below,
                    void_axis=void_up, void_opposite=void_down, sign_support=float(sa), sign_rank=se, sign_void=float(sv),
                    sign_score=float(sign_score),
                    score=float(score.max()), n=int(len(xyz)), flat_frac=float(flat.mean()))


def load_scene(cfg, a):
    """렌더러와 S0 기하를 함께 준비. --up auto면 지도에서 up을 추정해 cfg["up"]을 바꾼다(scenes.json 값은 진단용 비교에만)."""
    R = Renderer(cfg["ply"], cfg["sh_degree"])
    g = R.g
    if getattr(a, "up", "config") == "auto":
        ref = cfg["up"]
        up, info = estimate_up_from_gaussians(g.get_xyz.detach().cpu().numpy(), g.get_opacity.detach().cpu().numpy().reshape(-1),
                                              g.get_scaling.detach().cpu().numpy(), g.get_rotation.detach().cpu().numpy(), seed=0,
                                              renderer=R)
        cfg["up"], cfg["up_source"] = up, "auto(지도 Gaussian으로 추정)"
        cfg["up_auto_info"] = dict(info, up=up.tolist(), ref_up=ref.tolist(),
                                   ref_angle_deg=float(np.degrees(np.arccos(np.clip(up @ ref, -1, 1)))))
        print(f"[S-1] up 자동 추정 {np.round(up, 4).tolist()} (scenes.json 값과 {cfg['up_auto_info']['ref_angle_deg']:.1f}°, 진단용)", flush=True)
    G = s0_scene_geometry(g.get_xyz.detach().cpu().numpy(), g.get_opacity.detach().cpu().numpy().reshape(-1),
                          g.get_scaling.detach().cpu().numpy(), cfg["up"], a)
    return R, G


# ============================================================================= S1 위치 후보
def s1_band_candidates(G, n, rng, a):
    from common import band_filtered_shell_candidates
    return band_filtered_shell_candidates(G["struct"], G["struct_radii"], G["lo"], G["hi"], G["min_offset"], G["max_offset"],
                                          G["up"], G["band"][0], G["band"][1], rng, n, k_neighbors=a.k_neighbors,
                                          oversample=a.oversample)


def s1_pillar_candidates(G, n, rng, a):
    """S1(pillar): 수평 위치(기둥 축) x 높이 층(--levels, 바닥 p5=0, 구조물 p95=1). 높이 밴드에 묶이지 않고, 각 층 위치가
    Gaussian 밖·bbox 안인지만 검사한다(n = 검사 전 위치 수). 사용자 '기둥' 아이디어(2026-10-02)."""
    from common import surface_shell_candidates
    levels = np.array([float(x) for x in a.levels.split(",")])
    up = G["up"]
    c = surface_shell_candidates(G["struct"], G["struct_radii"], G["lo"], G["hi"], G["min_offset"], G["max_offset"],
                                 rng, int(np.ceil(n / len(levels))), k_neighbors=a.k_neighbors)
    ax = c - np.outer(c @ up, up)
    if getattr(a, "level_mode", "fixed") == "uniform":
        # 발표 피드백 검증(2026-10-24): 고정 층 대신 기둥마다 같은 개수의 높이를 [min(levels), max(levels)]에서 균등 추출
        lv = rng.uniform(levels.min(), levels.max(), size=(len(ax), len(levels)))
    else:
        lv = np.broadcast_to(levels, (len(ax), len(levels)))
    pos = (ax[:, None, :] + (G["floor"] + lv * (G["top"] - G["floor"]))[:, :, None] * up[None, None, :]).reshape(-1, 3)
    return pos[level_position_ok(G, pos, a)]


def level_position_ok(G, pos, a):
    """카메라 위치가 Gaussian 내부가 아니고(표면거리 > min_offset) bbox 안인지."""
    from common import gaussian_surface_distance
    inside = np.all((pos >= G["lo"]) & (pos <= G["hi"]), axis=1)
    d = gaussian_surface_distance(pos, G["struct"], G["struct_radii"], k_neighbors=a.k_neighbors, tree=G["tree"])
    return inside & (d > G["min_offset"])


# ============================================================================= S3 방향 검사 + 부채꼴
def s3_scan(G, position, probe_f, focal, W, H, a, rng):
    """방향별 빈 칸 비율과 칸별 최근접 깊이 p90, 시야 안 구조물 Gaussian 인덱스(렌더 없이 중심점 투영)."""
    from common import look_at_rotation
    struct, tree, up = G["struct"], G["tree"], G["up"]
    near, far = G["min_offset"], G["max_target_dist"]
    local = np.asarray(tree.query_ball_point(position, r=far))
    n = len(probe_f)
    if len(local) == 0:
        return dict(black_frac=np.ones(n), pctl_depth=np.zeros(n), idx_in=[np.empty(0, int)] * n, local_idx=local)
    if len(local) > a.local_cap:
        local = rng.choice(local, a.local_cap, replace=False)
    rel = struct[local] - position
    gw, gh = a.grid_w, a.grid_h
    bf, pd, ii = [], [], []
    for f in probe_f:
        cam = rel @ look_at_rotation(f, up).T
        z = cam[:, 2]
        inr = (z > near) & (z < far)
        if inr.any():
            zz = z[inr]
            u = focal * cam[inr, 0] / zz + W / 2.0
            v = focal * cam[inr, 1] / zz + H / 2.0
            ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        else:
            ok = np.zeros(0, bool)
        if not ok.any():
            bf.append(1.0); pd.append(0.0); ii.append(np.empty(0, int))
            continue
        ii.append(np.where(inr)[0][ok])
        ci = (v[ok] * gh / H).astype(int).clip(0, gh - 1) * gw + (u[ok] * gw / W).astype(int).clip(0, gw - 1)
        cm = np.full(gw * gh, np.inf)
        np.minimum.at(cm, ci, zz[ok])
        occ = np.isfinite(cm)
        bf.append(1.0 - occ.mean())
        pd.append(float(np.percentile(cm[occ], 90)))
    return dict(black_frac=np.array(bf), pctl_depth=np.array(pd), idx_in=ii, local_idx=local)


def longest_circular_true_run(mask):
    n = len(mask)
    if mask.all():
        return 0, n
    if not mask.any():
        return None
    cut = int(np.argmax(~mask))
    rolled = np.roll(mask, -cut)
    best_s, best_l, cs, cl = 0, 0, None, 0
    for i, v in enumerate(rolled):
        if v:
            if cl == 0:
                cs = i
            cl += 1
            if cl > best_l:
                best_l, best_s = cl, cs
        else:
            cl = 0
    return ((best_s + cut) % n, best_l) if best_l else None


def arc_indices(start, length, n, stride):
    return [(start + j * stride) % n for j in range(0, (length + stride - 1) // stride) if j * stride < length]


def s2_aim_pitch(G, pos, horiz_dirs, half_fov, a):
    """S2(aim): 방향마다 pitch를 '그 방향 수평 쐐기(±half_fov) 안 구조물의 높이 중심'을 겨누게 정한다.
    구조물 Gaussian의 고도각 atan2(높이차, 수평거리)를 겉보기 크기(w/d^2)로 가중한 중앙값, [pitch_min, pitch_max]로 자름.
    높은 곳에서는 내려다보고 낮은 곳에서는 올려다본다(기둥 아이디어). 쐐기에 점이 없으면 0."""
    up = G["up"]
    local = np.asarray(G["tree"].query_ball_point(pos, r=G["max_target_dist"]))
    out = np.zeros(len(horiz_dirs))
    if len(local) == 0:
        return out
    rel = G["struct"][local] - pos
    dh = rel @ up
    hv = rel - np.outer(dh, up)
    hd = np.linalg.norm(hv, axis=1)
    keep = hd > G["min_offset"]
    rel, dh, hv, hd, wt = rel[keep], dh[keep], hv[keep], hd[keep], G["struct_weight"][local][keep]
    elev = np.arctan2(dh, hd)
    if a.aim_weight == "density":
        wt = np.ones_like(wt)  # Gaussian 개수(=3DGS가 촘촘히 복원한 곳, 원래 촬영이 집중된 곳)를 겨눔
    wt = wt / np.maximum(hd ** 2 + dh ** 2, 1e-9)
    cosv = (hv / hd[:, None]) @ horiz_dirs.T
    lo, hi = np.deg2rad(a.pitch_min), np.deg2rad(a.pitch_max)
    for k in range(len(horiz_dirs)):
        m = cosv[:, k] > np.cos(half_fov)
        if m.sum() < 10:
            continue
        o = np.argsort(elev[m]); cw = np.cumsum(wt[m][o])
        out[k] = np.clip(elev[m][o][np.searchsorted(cw, cw[-1] / 2)], lo, hi)
    return out


def s3_probe(G, pos, n_probe, focal, W, H, a, rng):
    """S2 방향 후보(pitch 0 또는 aim) + S3 기하 스캔."""
    pf = sweep_forwards(G["up"], n_probe, rng.uniform(0, 2 * np.pi), 0.0)
    if a.s2 in ("aim", "random"):
        if a.s2 == "aim":
            p = s2_aim_pitch(G, pos, pf, np.arctan(W / 2.0 / focal), a)
        else:  # 발표 피드백 검증(2026-10-24): 방향마다 pitch를 [pitch_min, pitch_max]에서 균등 추출(aim 대신)
            p = rng.uniform(np.deg2rad(a.pitch_min), np.deg2rad(a.pitch_max), size=len(pf))
        pf = np.cos(p)[:, None] * pf + np.sin(p)[:, None] * G["up"][None, :]
    return pf, s3_scan(G, pos, pf, focal, W, H, a, rng)


def s3_closeup_quantile(G, cands, n_probe, focal, W, H, a, rng):
    """--closeup_q: 근접샷 문턱을 '이 씬 후보 방향들(빈 칸 기준 통과)의 칸별 최근접 깊이 p90' 분포의 q 분위로 정한다.
    기본(3*높이 밴드 폭)은 야외처럼 키 큰 구조물이 있는 씬에서 실제 촬영 깊이보다 몇 배 길어 먼 미재구성 영역만 보게 됐다.
    RNG 상태를 저장·복원해 이후 본 스캔이 같은 방향·같은 표본을 쓰게 한다."""
    st = rng.bit_generator.state
    pool = []
    for c in cands:
        _, r = s3_probe(G, c, n_probe, focal, W, H, a, rng)
        pool.append(r["pctl_depth"][r["black_frac"] <= a.black_max_frac])
    rng.bit_generator.state = st
    pool = np.concatenate(pool)
    return float(np.quantile(pool, a.closeup_q)) if len(pool) else G["min_target_dist"]


def s3_point_views(G, pos, n_probe, focal, W, H, a, rng, R=None):
    """한 point에서 S2(--s2 level: pitch 0 / aim: 방향별 자동 pitch) 스윕 방향을 검사해 찍을 방향(부채꼴)을 정한다.
    --s3_render_px가 있으면 기하 검사를 통과한 방향을 저해상도로 실제 렌더해 void(및 --max_relstd) 기준도 S3에서 적용한다
    (중심점 투영은 크고 반투명한 야외 Gaussian이 칸을 덮는 것으로 보아 빈 영역을 과소평가 — stump에서 S3 통과 99%, 렌더 후 void 탈락 거의 전부).
    반환: (채택 시 hub dict 아니면 None, 로그 행)."""
    pf, r = s3_probe(G, pos, n_probe, focal, W, H, a, rng)
    ok_dir = (r["black_frac"] <= a.black_max_frac) & (r["pctl_depth"] >= G["min_target_dist"])
    if a.s3_render_px and R is not None and (ok_dir.any() or a.pitch_fallback_step):
        from common import look_at_rotation
        w = a.s3_render_px; h = max(1, int(round(H * w / W))); fl = focal * w / W

        def clean(f):
            Rw = look_at_rotation(f, G["up"]); tw = -Rw @ pos
            m = frame_metrics(*R.stats(R.camera(Rw, tw, fl, w, h), Rw, tw), G["near_dist"])
            return (m["void_frac"] <= a.max_void_frac and m["near_frac"] <= a.max_near_frac
                    and (a.max_relstd is None or m["relstd"] <= a.max_relstd))
        for j in range(n_probe):
            if ok_dir[j]:
                ok_dir[j] = clean(pf[j])
            if ok_dir[j] or not a.pitch_fallback_step:
                continue
            # --pitch_fallback_step: 통과하지 못한 방향은 pitch를 step도씩 내려 처음 통과하는 각도를 쓴다
            # (야외에서 위/수평 시점은 미재구성 하늘이 보여 탈락 — 실제 사람 촬영처럼 약간 내려다보면 깨끗함)
            up = G["up"]; pj = float(np.arcsin(np.clip(pf[j] @ up, -1, 1)))
            hz = pf[j] - (pf[j] @ up) * up; hz /= np.linalg.norm(hz)
            q = pj - np.deg2rad(a.pitch_fallback_step)
            while q >= np.deg2rad(a.pitch_min) - 1e-9:
                f = np.cos(q) * hz + np.sin(q) * up
                if clean(f):
                    pf[j] = f; ok_dir[j] = True
                    break
                q -= np.deg2rad(a.pitch_fallback_step)
    arc = longest_circular_true_run(ok_dir)
    ridx = arc_indices(arc[0], arc[1], n_probe, a.probe_oversample) if arc else []
    ok = len(ridx) >= a.min_arc_views
    arc_all = list(ridx)
    if ok and a.views_per_point and len(ridx) > a.views_per_point:
        # 위치 다양성: 부채꼴 전체 대신 고르게 벌린 views_per_point장만 → 같은 장수에 더 많은 point
        m = a.views_per_point
        ridx = [ridx[int((i + 0.5) * len(ridx) / m)] for i in range(m)]
    row = dict(ok=ok, n_pass_dirs=int(ok_dir.sum()), arc_views=len(ridx),
               is_full=bool(arc and arc[1] == n_probe), height=float(pos @ G["up"]))
    if not ok:
        return None, row
    si = np.unique(np.concatenate([r["idx_in"][j] for j in ridx]))
    return dict(pos=pos, seen_idx=r["local_idx"][si] if len(si) else si, forwards=pf[ridx],
                spare_forwards=pf[[j for j in arc_all if j not in ridx]], is_full=bool(arc[1] == n_probe)), row


# ============================================================================= S4 point 순서
def s4_greedy_order(accepted, weight, k_cap):
    """sum_g w_g*min(hub_count_g,k_cap) lazy greedy(부분모듈러 → 일반 그리디와 같은 이득열, 동점 순서만 다를 수 있음)."""
    import heapq
    M = np.zeros(len(weight), dtype=np.int32)

    def gain(j):
        s = accepted[j]["seen_idx"]
        return float((weight[s] * (M[s] < k_cap)).sum()) if len(s) else 0.0

    heap = [(-gain(j), j) for j in range(len(accepted))]
    heapq.heapify(heap)
    order = []
    while heap:
        _, j = heapq.heappop(heap)
        g = gain(j)
        if heap and g < -heap[0][0]:
            heapq.heappush(heap, (-g, j))
            continue
        order.append(j)
        s = accepted[j]["seen_idx"]
        M[s] = np.minimum(M[s] + 1, k_cap)
    return order


# ============================================================================= S5 사진 검사
class PhotoFilter:
    """렌더된 프레임의 불량 사유 목록(빈 목록 = 합격). 선명도 하한은 처음 calib_frames장의 중앙값에 비례(씬마다 스케일이 달라서)."""

    def __init__(self, a):
        self.a, self.thr, self.med = a, a.min_sharpness, None

    def calibrate(self, sharpness_values):
        self.med = float(np.median(sharpness_values))
        self.thr = self.a.min_sharpness_rel * self.med

    def reasons(self, r):
        a, w = self.a, []
        if r["void_frac"] > a.max_void_frac: w.append(f"void{r['void_frac']:.2f}")
        if r["near_frac"] > a.max_near_frac: w.append(f"near{r['near_frac']:.2f}")
        if r["sharpness"] < self.thr: w.append(f"sharp{r['sharpness']:.1f}")
        if a.max_relstd is not None and r["relstd"] > a.max_relstd: w.append(f"relstd{r['relstd']:.2f}")
        return w


# ============================================================================= 출력 공통
def prepare_out(a, cfg):
    out = Path(a.out_dir)
    if out.exists():
        if not getattr(a, "clean", False):
            raise SystemExit(f"{out} 가 이미 있습니다(--clean 으로 지우거나 새 이름 사용)")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    return out


def finish_out(out, a, cfg, extra, preview):
    """run_args.json, 미리보기, (선택) 고정 test 링크."""
    from common import save_top_k_grid
    (out / "run_args.json").write_text(json.dumps(dict(vars(a), version=VERSION, up=cfg["up"].tolist(), up_auto_info=cfg.get("up_auto_info"), ply=cfg["ply"], **extra),
                                                  indent=1, ensure_ascii=False))
    if preview:
        save_top_k_grid(out / "train_preview.jpg", preview[:48], cols=6)
    n = len(list((out / "train" / "rgb").glob("*.png")))
    if a.link_test != "none":
        src = (EVAL_ROOT / a.scene / a.link_test).resolve()
        if not (src / "rgb").is_dir():
            raise SystemExit(f"고정 평가셋 {src} 이 없습니다")
        (out / "test").symlink_to(src)
    print(f"완료 {out}: train {n}장" + (f", test -> {a.scene}/{a.link_test}" if a.link_test != "none" else ""))
    return n


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


# ============================================================================= run (S0~S6)
def cmd_run(a):
    from common import look_at_rotation, write_ace_frame, hub_views_per_overlap, focal_to_fov, laplacian_sharpness
    cfg = load_scene_cfg(a.scene)
    rng = np.random.default_rng(a.seed)
    out = prepare_out(a, cfg)
    W, H, focal = resolve_camera(a)
    bg_name, bg = background(a, cfg)
    K = hub_views_per_overlap(focal_to_fov(focal, W), a.overlap_ratio)
    n_probe = K * a.probe_oversample
    R, G = load_scene(cfg, a)
    print(f"[S0] {a.scene}: 구조물 {len(G['struct'])}, 밴드 {G['band']}, 근접샷 문턱 {G['min_target_dist']:.3f}, K={K}")

    cands = (s1_band_candidates if a.s1 == "band" else s1_pillar_candidates)(G, a.n_candidates, rng, a)
    print(f"[S1:{a.s1}] 위치 후보 {len(cands)}, S2 {a.s2}")
    if a.closeup_q is not None:
        old = G["min_target_dist"]
        G["min_target_dist"] = s3_closeup_quantile(G, cands, n_probe, focal, W, H, a, rng)
        G["near_dist"] = a.near_dist_ratio * G["min_target_dist"]
        print(f"[S3] 근접샷 문턱 {old:.3f} -> {G['min_target_dist']:.3f} (후보 방향 p90 깊이의 {a.closeup_q} 분위)")

    t0 = time.time()
    accepted, scan_log = [], []
    for i, c in enumerate(cands):
        hub, row = s3_point_views(G, c, n_probe, focal, W, H, a, rng, R)
        scan_log.append(dict(cand=i, **row))
        if hub is not None:
            accepted.append(hub)
    print(f"[S3] 채택 point {len(accepted)}/{len(cands)} ({time.time() - t0:.0f}s)")
    if not accepted:
        if getattr(a, "relax_if_short", 0) > 0:
            # 2026-10-28: 지도 품질이 낮아 S3에서 위치가 하나도 남지 않을 때도(T&T train, final2 seed 0·1) v1.1 완화를 적용한다.
            # 이전에는 S6에서 장수가 모자랄 때만 완화해, 이 경우는 완화 없이 종료됐다. 채택 위치가 있는 씬에서는 동작이 같다.
            del R
            return relaxed_rerun(a, 0, "S3")
        raise SystemExit("채택 point 없음")
    order = s4_greedy_order(accepted, G["struct_weight"], a.k_cap)
    if getattr(a, "s4_order", "greedy") == "random":  # E4f ablation: 커버리지 순서 대신 무작위 순서(채택 point 집합은 같음)
        order = list(np.random.default_rng(a.seed + 777).permutation(len(accepted)))

    flt = PhotoFilter(a)
    n_written, n_hub = 0, 0
    rows, log, preview = [], [], []
    relaxable = []  # 빈 화면/선명도만으로 걸러진 프레임(마지막 완화 채우기 후보)

    def render_point(hub, key="forwards"):
        """깊이 통계를 먼저 렌더하고, 선명도 보정이 끝난 뒤에는 void/near/relstd로 이미 탈락한 프레임의 컬러 렌더·선명도
        계산을 건너뛴다(결정은 같고 속도만 빠름 — 야외처럼 거의 다 걸러지는 씬에서 S6 시간이 수 시간이던 문제)."""
        fr = []
        for vi, f in enumerate(hub[key]):
            Rw = look_at_rotation(f, G["up"]); tw = -Rw @ hub["pos"]
            cam = R.camera(Rw, tw, focal, W, H)
            m = frame_metrics(*R.stats(cam, Rw, tw), G["near_dist"])
            dead = flt.thr is not None and any(x.startswith(("void", "near", "relstd")) for x in flt.reasons(dict(m, sharpness=np.inf)))
            img = None if dead else R.color(cam, bg)
            fr.append(dict(img=img, R=Rw, t=tw, view_idx=vi, sharpness=np.nan if dead else float(laplacian_sharpness(img)), **m))
        return fr

    def assign(rank, j, frames, key="forwards"):
        nonlocal n_written, n_hub
        if n_written >= a.n_train:
            return
        ws = [flt.reasons(r) for r in frames]
        relaxable.extend((r["void_frac"], -np.nan_to_num(r["sharpness"]), rank, j, key, r["view_idx"]) for r, w in zip(frames, ws)
                         if w and all(x.startswith(("void", "sharp")) for x in w))
        kept = [r for r, w in zip(frames, ws) if not w][:a.n_train - n_written]
        name = f"train{n_hub}"
        log.append(dict(rank=rank, accepted_idx=j, hub=name if kept else "", n_views=len(frames), n_kept=len(kept),
                        rejected=";".join(f"v{r['view_idx']}:" + "+".join(w) for r, w in zip(frames, ws) if w),
                        is_full=accepted[j]["is_full"]))
        if not kept:
            return
        for r in kept:
            stem = write_ace_frame(out, "train", n_written, r["img"], r["R"], r["t"], focal)
            rows.append(dict(frame=stem, hub_id=name, view_idx=r["view_idx"], greedy_rank=rank,
                             void_frac=round(r["void_frac"], 4), near_frac=round(r["near_frac"], 4),
                             relstd=round(r["relstd"], 4), median_depth=round(r["median_depth"], 3),
                             sharpness=round(r["sharpness"], 2)))
            if r is kept[0]:
                preview.append((out / "train" / "rgb" / f"{stem}.color.png", f"{name} v{r['view_idx']}"))
            n_written += 1
        n_hub += 1
        print(f"  rank {rank} -> {name}: {len(kept)}/{len(frames)}  [train {n_written}/{a.n_train}]")

    # S6: 순서대로 렌더→검사→저장. 선명도 하한이 정해지기 전(처음 calib_frames장)에는 렌더만 모아 두었다가 한꺼번에 검사.
    pending = []
    for rank, j in enumerate(order):
        if n_written >= a.n_train:
            break
        frames = render_point(accepted[j])
        if flt.thr is None:
            pending.append((rank, j, frames))
            if sum(len(f) for _, _, f in pending) >= a.calib_frames:
                flt.calibrate([r["sharpness"] for _, _, f in pending for r in f])
                for it in pending: assign(*it)
                pending = []
            continue
        assign(rank, j, frames)
    if pending:
        flt.calibrate([r["sharpness"] for _, _, f in pending for r in f])
        for it in pending: assign(*it)
    # --views_per_point로 point를 다 써도 모자라면, 같은 순서로 각 point의 남은 부채꼴 방향을 2차로 채운다
    for rank, j in enumerate(order):
        if n_written >= a.n_train:
            break
        if len(accepted[j].get("spare_forwards", [])):
            assign(rank, j, render_point(accepted[j], "spare_forwards"), "spare_forwards")
    # 그래도 모자라면(야외처럼 미재구성 하늘이 많아 S5 void가 거의 다 거르는 씬) 빈 화면/선명도로만 걸린 프레임을
    # 빈 화면 비율이 작은 순으로 다시 렌더해 채운다. 흐림(relstd)·근접(near)으로 걸린 프레임은 쓰지 않는다.
    n_relaxed = 0
    if n_written < a.n_train and relaxable:
        print(f"[S6] 완화 채우기: {a.n_train - n_written}장 부족, 후보 {len(relaxable)}")
        for void, _, rank, j, key, vi in sorted(relaxable):
            if n_written >= a.n_train:
                break
            hub = accepted[j]
            Rw = look_at_rotation(hub[key][vi], G["up"]); tw = -Rw @ hub["pos"]
            cam = R.camera(Rw, tw, focal, W, H)
            img = R.color(cam, bg)
            m = frame_metrics(*R.stats(cam, Rw, tw), G["near_dist"])
            stem = write_ace_frame(out, "train", n_written, img, Rw, tw, focal)
            rows.append(dict(frame=stem, hub_id=f"relaxed_r{rank}", view_idx=vi, greedy_rank=rank,
                             void_frac=round(m["void_frac"], 4), near_frac=round(m["near_frac"], 4), relstd=round(m["relstd"], 4),
                             median_depth=round(m["median_depth"], 3), sharpness=round(float(laplacian_sharpness(img)), 2)))
            n_written += 1; n_relaxed += 1

    if n_written < a.n_train and getattr(a, "relax_if_short", 0) > 0:
        del R
        return relaxed_rerun(a, n_written, "S6")
    write_csv(out / "hub_index.csv", rows)
    write_csv(out / "render_backfill_log.csv", log)
    write_csv(out / "candidate_scan_log.csv", scan_log)
    if n_written < a.n_train:
        print(f"[경고] train {n_written} < {a.n_train} (후보 소진, --n_candidates 증가 필요)")
    print(f"[S6] {n_hub} point, 선명도 하한 {flt.thr:.3f}")
    finish_out(out, a, cfg, dict(camera=dict(width=W, height=H, focal_px=focal), bg_color=bg_name, K=K,
                                 min_target_dist=G["min_target_dist"], near_dist=G["near_dist"],
                                 sharpness_median=flt.med, sharpness_thr=flt.thr, n_points=n_hub, n_relaxed=n_relaxed,
                                 relax_round=getattr(a, "relax_round", 0)), preview)


# ============================================================================= 기준선
def relaxed_rerun(a, n_written, stage):
    """v1.1(2026-10-12): 지도 품질이 낮아(예: T&T train, 실제 시점 렌더도 relstd 중앙값 0.19) 목표 장수를 못 채우거나
    S3에서 위치가 하나도 남지 않으면 relstd/void 기준을 완화해 처음부터 다시 고른다. 목표를 채운 씬에서는 실행되지 않는다."""
    import copy
    b = copy.copy(a)
    b.relax_if_short = a.relax_if_short - 1
    b.relax_round = getattr(a, "relax_round", 0) + 1
    b.max_relstd = None if a.max_relstd is None else a.max_relstd * 1.25
    b.max_void_frac = a.max_void_frac * 1.5
    b.clean = True
    print(f"[{stage}] {n_written}/{a.n_train}장 -> 완화 {b.relax_round}회차 재실행 (max_relstd {b.max_relstd}, max_void {b.max_void_frac:.3f})", flush=True)
    return cmd_run(b)


def cmd_random(a):
    """무작위 기준선. full: bbox 균등 위치(Gaussian 내부 제외) + Haar 회전. upright: S1 후보 분포 위치 + 무작위 yaw, pitch 0, roll 0.
    pillar: 최종 알고리즘의 S1 위치 후보(기둥 축 x 높이 U[0.1, 1.25]) + Haar 회전."""
    from scipy.spatial.transform import Rotation
    from common import look_at_rotation, write_ace_frame
    cfg = load_scene_cfg(a.scene)
    rng = np.random.default_rng(a.seed)
    out = prepare_out(a, cfg)
    W, H, focal = resolve_camera(a)
    bg_name, bg = background(a, cfg)
    R, G = load_scene(cfg, a)
    preview, k = [], 0
    if a.mode == "full":
        while k < a.n_train:
            p = G["lo"] + rng.random(3) * (G["hi"] - G["lo"])
            if not level_position_ok(G, p[None], a)[0]:
                continue
            Rw = Rotation.random(random_state=rng.integers(1 << 31)).as_matrix(); tw = -Rw @ p
            stem = write_ace_frame(out, "train", k, R.color(R.camera(Rw, tw, focal, W, H), bg), Rw, tw, focal)
            preview.append((out / "train" / "rgb" / f"{stem}.color.png", f"{k}"))
            k += 1
    elif a.mode == "pillar":
        # 2026-10-29 사용자 요청 비교군: 위치는 최종 알고리즘(final3)의 S1 후보 분포, 방향은 완전 무작위(Haar), S2~S6 없음.
        a.levels, a.level_mode = "0.1,0.25,0.5,0.75,1.0,1.25", "uniform"
        pool = np.empty((0, 3))
        while len(pool) < a.n_train:
            pool = np.concatenate([pool, s1_pillar_candidates(G, max(2000, 4 * a.n_train), rng, a)])
        pos = pool[rng.choice(len(pool), a.n_train, replace=False)]
        print(f"[pillar] 위치 풀 {len(pool)}에서 {a.n_train}개 무작위 선택")
        for k, p in enumerate(pos):
            if getattr(a, "pillar_dir", "full") == "upright":  # 시선은 구면 균등(pitch = arcsin U[-1,1], ±89도로 자름), roll 0
                q = np.clip(np.arcsin(rng.uniform(-1, 1)), -np.deg2rad(89), np.deg2rad(89))
                f = sweep_forwards(G["up"], 1, rng.uniform(0, 2 * np.pi), q)[0]
                Rw = look_at_rotation(f, G["up"])
            else:
                Rw = Rotation.random(random_state=rng.integers(1 << 31)).as_matrix()
            tw = -Rw @ p
            stem = write_ace_frame(out, "train", k, R.color(R.camera(Rw, tw, focal, W, H), bg), Rw, tw, focal)
            preview.append((out / "train" / "rgb" / f"{stem}.color.png", f"{k}"))
    elif a.mode == "filtered":
        # 완전도 필터 무작위(2026-10-12, 지도 신뢰도 가설 검증용 E4): 장면 어디서든 위치·yaw·pitch(U[-60,40])를 무작위로 뽑고
        # 160px 렌더가 완전한(void<=0.05, near<=0.20, relstd<=0.30) 시점만 받아들인다. 높이 층·자동 pitch·커버리지 순서 없음.
        w = 160; h = max(1, int(round(H * w / W))); fl = focal * w / W
        thr = dict(void=0.05, near=0.20, relstd=0.30); tries = 0
        pool = None
        if getattr(a, "pos_source", "bbox") == "pillar":  # E4c: 알고리즘과 같은 구조물 근처 위치(기둥 축 x 높이 층)에서만
            a.levels = "0.1,0.25,0.5,0.75,1.0,1.25"
            pool = s1_pillar_candidates(G, 20000, rng, a)
            print(f"[filtered] pillar 위치 풀 {len(pool)}")
        while k < a.n_train:
            tries += 1
            if tries % (100 * a.n_train) == 0:  # 지도 품질이 낮아 통과가 드물면 v1.1처럼 완화
                thr = dict(void=thr["void"] * 1.5, near=thr["near"], relstd=thr["relstd"] * 1.25)
                print(f"[filtered] {tries}회 시도에 {k}장 -> 기준 완화 {thr}")
            p = pool[rng.integers(len(pool))] if pool is not None else G["lo"] + rng.random(3) * (G["hi"] - G["lo"])
            if getattr(a, "s_min", None) is not None or getattr(a, "s_max", None) is not None:  # E4b: 지도에서 추정한 바닥(p5) 기준 높이 제한
                sh = (p @ G["up"] - G["floor"]) / (G["top"] - G["floor"])
                if (a.s_min is not None and sh < a.s_min) or (a.s_max is not None and sh > a.s_max):
                    continue
            if not level_position_ok(G, p[None], a)[0]:
                continue
            if getattr(a, "pano_min", 0) > 0:  # E4g: 둘레 완전도 — 12개 yaw 중 깨끗한(pitch 0/-15/-30 중 하나) 방향 수가 pano_min 이상인 위치만
                clean_dirs = []
                for y in rng.uniform(0, 2 * np.pi) + 2 * np.pi * np.arange(12) / 12:
                    for q in (0.0, -15.0, -30.0):
                        fq = sweep_forwards(G["up"], 1, y, np.deg2rad(q))[0]; Rq = look_at_rotation(fq, G["up"]); tq = -Rq @ p
                        mq = frame_metrics(*R.stats(R.camera(Rq, tq, fl, w, h), Rq, tq), G["near_dist"])
                        if mq["void_frac"] <= thr["void"] and mq["near_frac"] <= thr["near"] and mq["relstd"] <= thr["relstd"]:
                            clean_dirs.append(fq); break
                if len(clean_dirs) < a.pano_min:
                    continue
                f = clean_dirs[rng.integers(len(clean_dirs))]
                Rw = look_at_rotation(f, G["up"]); tw = -Rw @ p
                stem = write_ace_frame(out, "train", k, R.color(R.camera(Rw, tw, focal, W, H), bg), Rw, tw, focal)
                preview.append((out / "train" / "rgb" / f"{stem}.color.png", f"{k}"))
                k += 1
                continue
            if getattr(a, "pitch_source", "random") == "aim":  # E4d: 무작위 yaw + 그 방향의 aim pitch(density 가중)
                hz = sweep_forwards(G["up"], 1, rng.uniform(0, 2 * np.pi), 0.0)
                a.aim_weight = "density"; a.pitch_min = -45.0; a.pitch_max = 30.0
                q = s2_aim_pitch(G, p, hz, np.arctan(W / 2.0 / focal), a)[0]
                f = np.cos(q) * hz[0] + np.sin(q) * G["up"]
            else:
                f = sweep_forwards(G["up"], 1, rng.uniform(0, 2 * np.pi), np.deg2rad(rng.uniform(-60, 40)))[0]
            Rw = look_at_rotation(f, G["up"]); tw = -Rw @ p
            if getattr(a, "far_check", False):  # E4h: 알고리즘 S3의 근접샷 기준(칸별 최근접 깊이 p90 >= 3*밴드 폭)과 빈 칸 기준만 추가
                a.local_cap = 30000; a.grid_w = 8; a.grid_h = 6
                sc = s3_scan(G, p, f[None], focal, W, H, a, rng)
                if not (sc["black_frac"][0] <= 0.15 and sc["pctl_depth"][0] >= G["min_target_dist"]):
                    continue
            m = frame_metrics(*R.stats(R.camera(Rw, tw, fl, w, h), Rw, tw), G["near_dist"])
            if m["void_frac"] > thr["void"] or m["near_frac"] > thr["near"] or m["relstd"] > thr["relstd"]:
                continue
            stem = write_ace_frame(out, "train", k, R.color(R.camera(Rw, tw, focal, W, H), bg), Rw, tw, focal)
            preview.append((out / "train" / "rgb" / f"{stem}.color.png", f"{k}"))
            k += 1
            vpp = getattr(a, "views_per_pos", 1)
            if vpp > 1:  # E4e: 같은 위치에서 50% 겹치는 이웃 yaw(같은 pitch)를 번갈아 시도해 최대 vpp장(겹치는 다중 시점)
                fu = f @ G["up"]; hz0 = f - fu * G["up"]; hz0 /= np.linalg.norm(hz0); e2 = np.cross(G["up"], hz0)
                dy = 2 * np.arctan(W / 2.0 / focal) * 0.5; got = 1
                for j in [1, -1, 2, -2, 3, -3]:
                    if got >= vpp or k >= a.n_train:
                        break
                    hzj = np.cos(j * dy) * hz0 + np.sin(j * dy) * e2
                    fj = np.sqrt(max(0.0, 1 - fu ** 2)) * hzj + fu * G["up"]
                    Rj = look_at_rotation(fj, G["up"]); tj = -Rj @ p
                    mj = frame_metrics(*R.stats(R.camera(Rj, tj, fl, w, h), Rj, tj), G["near_dist"])
                    if mj["void_frac"] > thr["void"] or mj["near_frac"] > thr["near"] or mj["relstd"] > thr["relstd"]:
                        continue
                    write_ace_frame(out, "train", k, R.color(R.camera(Rj, tj, focal, W, H), bg), Rj, tj, focal)
                    k += 1; got += 1
        print(f"[filtered] {tries}회 시도, 통과율 {a.n_train / tries:.3f}")
    else:
        pos = np.empty((0, 3))
        while len(pos) < a.n_train:
            pos = np.concatenate([pos, s1_band_candidates(G, a.n_train - len(pos), rng, a)])
        for k, p in enumerate(pos):
            f = sweep_forwards(G["up"], 1, rng.uniform(0, 2 * np.pi), 0.0)[0]
            Rw = look_at_rotation(f, G["up"]); tw = -Rw @ p
            stem = write_ace_frame(out, "train", k, R.color(R.camera(Rw, tw, focal, W, H), bg), Rw, tw, focal)
            preview.append((out / "train" / "rgb" / f"{stem}.color.png", f"{k}"))
    finish_out(out, a, cfg, dict(camera=dict(width=W, height=H, focal_px=focal), bg_color=bg_name), preview)


def cmd_human(a):
    """사람 기준선: picker(--purpose train)에서 내려받은 포즈 JSON을 같은 렌더러로 렌더. 카메라는 JSON 값(picker 화면 구도)."""
    import hashlib
    from common import look_at_rotation, write_ace_frame
    cfg = load_scene_cfg(a.scene)
    J = json.loads(Path(a.poses).read_text())
    assert J["scene"] == a.scene, f"포즈 파일 씬 {J['scene']} != {a.scene}"
    up = np.array(J["up"], dtype=np.float64)
    assert np.allclose(up / np.linalg.norm(up), cfg["up"], atol=1e-6), "포즈 파일 up이 scenes.json과 다름"
    caps = J["captures"]
    if not caps:
        raise SystemExit("포즈가 없습니다")
    out = prepare_out(a, cfg)
    W, H = int(J["width"]), int(J["height"])
    focal = focal_for(W, float(J["hfov_deg"]))
    bg_name, bg = background(a, cfg)
    R, G = load_scene(cfg, a)
    rows, preview = [], []
    for k, c in enumerate(caps):
        p = np.array(c["position"], dtype=np.float64)
        Rw = look_at_rotation(np.array(c["forward"], dtype=np.float64), cfg["up"]); tw = -Rw @ p
        cam = R.camera(Rw, tw, focal, W, H)
        stem = write_ace_frame(out, "train", k, R.color(cam, bg), Rw, tw, focal)
        m = frame_metrics(*R.stats(cam, Rw, tw), G["near_dist"])
        rows.append(dict(frame=stem, s=round(c.get("s", float("nan")), 4), pitch_deg=round(c["pitch_deg"], 2),
                         auto=int(bool(c.get("auto"))), **{kk: round(v, 4) for kk, v in m.items()}))
        preview.append((out / "train" / "rgb" / f"{stem}.color.png", f"{k}"))
    write_csv(out / "human_index.csv", rows)
    finish_out(out, a, cfg, dict(camera=dict(width=W, height=H, focal_px=focal), bg_color=bg_name,
                                 poses_file=str(Path(a.poses).resolve()),
                                 poses_sha256=hashlib.sha256(Path(a.poses).read_bytes()).hexdigest()), preview)


# ============================================================================= CLI
FINAL_PRESET = dict(s1="pillar", levels="0.1,0.25,0.5,0.75,1.0,1.25", level_mode="uniform", s2="aim", aim_weight="density", views_per_point=2,
                    n_candidates=2000, max_relstd=0.30, s3_render_px=160, pitch_fallback_step=10.0, max_struct=2000000,
                    relax_if_short=4, up="auto")


def add_scene_args(p):
    p.add_argument("--scene", required=True, help="scenes.json 키")
    p.add_argument("--seed", type=int, default=0)
    for k, v in GEOMETRY_DEFAULTS.items():
        p.add_argument(f"--{k}", type=float, default=v)
    p.add_argument("--k_neighbors", type=int, default=16)
    p.add_argument("--oversample", type=int, default=3)
    p.add_argument("--max_struct", type=int, default=None, help="S0 구조물 Gaussian 수 상한(초과 시 고정 표본, 속도용)")
    p.add_argument("--up", choices=["config", "auto"], default="config",
                   help="config: scenes.json up_axis / auto: 지도 Gaussian만으로 추정(estimate_up_from_gaussians, GT 불필요)")


def add_output_args(p, camera=True):
    p.add_argument("--out_dir", required=True); p.add_argument("--clean", action="store_true")
    p.add_argument("--link_test", choices=["none", "test", "dev"], default="none",
                   help="<out_dir>/test를 datasets/eval_fixed/<scene>/<split> 심볼릭 링크로")
    p.add_argument("--bg_color", choices=["black", "white"], default=None, help="기본: scenes.json의 bg_color")
    if camera:
        p.add_argument("--width", type=int, default=1600); p.add_argument("--height", type=int, default=1064)
        p.add_argument("--fov_deg", type=float, default=65.0, help="수평 FOV(--focal_px가 없을 때)")
        p.add_argument("--focal_px", type=float, default=None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="알고리즘(S0~S6)"); add_scene_args(r); add_output_args(r)
    r.add_argument("--preset", choices=["none", "final"], default="none",
                   help="final = 2026-10-28 최종 설정(FINAL_PRESET: 무작위 높이 + 구조물 겨누는 pitch + 렌더 완전도 검사 + up 자동 추정; 개별 플래그보다 우선)")
    r.add_argument("--n_train", type=int, default=300)
    r.add_argument("--n_candidates", type=int, default=600, help="S1 위치 후보 수(pillar: 높이 층 검사 전 위치 수)")
    r.add_argument("--s1", choices=["band", "pillar"], default="band", help="band: 높이 밴드 p15~45 / pillar: 기둥 축 x 높이 층")
    r.add_argument("--levels", default="0.25,0.5,0.75,1.0,1.25", help="--s1 pillar 높이 층(바닥 p5=0, 구조물 p95=1)")
    r.add_argument("--s2", choices=["level", "aim", "random"], default="level",
                   help="level: pitch 0 / aim: 방향별로 구조물 높이 중심을 겨눔 / random: 방향별 pitch U[pitch_min, pitch_max](검증용)")
    r.add_argument("--level_mode", choices=["fixed", "uniform"], default="fixed",
                   help="--s1 pillar 높이: fixed = --levels 고정 층 / uniform = 기둥마다 같은 개수를 [min, max] 균등 추출(검증용)")
    r.add_argument("--pitch_min", type=float, default=-45.0); r.add_argument("--pitch_max", type=float, default=30.0)
    r.add_argument("--aim_weight", choices=["area", "density"], default="area",
                   help="S2 aim 가중: area = 겉넓이(불투명도x큰 두 scale)/거리^2, density = 개수/거리^2(큰 배경 Gaussian에 덜 끌림)")
    r.add_argument("--overlap_ratio", type=float, default=0.5, help="S3 인접 방향 겹침")
    r.add_argument("--probe_oversample", type=int, default=2, help="S3 검사 방향 조밀도(K의 배수)")
    r.add_argument("--grid_w", type=int, default=8); r.add_argument("--grid_h", type=int, default=6)
    r.add_argument("--black_max_frac", type=float, default=0.15, help="S3 빈 칸 비율 상한")
    r.add_argument("--s3_render_px", type=int, default=0,
                   help="S3에서 기하 검사 통과 방향을 이 폭으로 저해상도 렌더해 void/near/relstd를 미리 적용(0 = 끔)")
    r.add_argument("--pitch_fallback_step", type=float, default=0.0,
                   help="--s3_render_px와 함께: 검사에 탈락한 방향을 이 각도(도)씩 내려 처음 통과하는 pitch로 대체(0 = 끔)")
    r.add_argument("--relax_if_short", type=int, default=0,
                   help="목표 장수를 못 채우면 relstd x1.25, void x1.5로 완화해 최대 이 횟수만큼 다시 실행(v1.1, 0 = 끔)")
    r.add_argument("--closeup_q", type=float, default=None,
                   help="S3 근접샷 문턱을 후보 방향 깊이 분포의 분위로(예 0.25). 기본: 3*높이 밴드 폭")
    r.add_argument("--min_arc_views", type=int, default=4, help="S3 부채꼴 최소 장수(point 채택 기준)")
    r.add_argument("--views_per_point", type=int, default=0,
                   help="S3 point당 찍는 최대 장수(부채꼴 안에서 고르게). 0 = 부채꼴 전체(2026-10-08 정리본 동작)")
    r.add_argument("--local_cap", type=int, default=30000, help="S3 검사에 쓰는 주변 Gaussian 수 상한")
    r.add_argument("--k_cap", type=int, default=3, help="S4 같은 Gaussian을 몇 point까지 보면 충분한지")
    r.add_argument("--s4_order", choices=["greedy", "random"], default="greedy", help="S4 순서(ablation용 random)")
    r.add_argument("--max_void_frac", type=float, default=0.05); r.add_argument("--max_near_frac", type=float, default=0.20)
    r.add_argument("--min_sharpness_rel", type=float, default=0.25); r.add_argument("--min_sharpness", type=float, default=None)
    r.add_argument("--calib_frames", type=int, default=48)
    r.add_argument("--max_relstd", type=float, default=None,
                   help="S5 ray 깊이 퍼짐(중앙값 sqrt(Var z)/E z) 상한. 반투명 근접 흐림 제거. 기본 꺼짐(야외 먼 배경 편향 미해결)")
    for name, mode in [("random", None), ("baseline-random", "full")]:
        b = sub.add_parser(name, help="무작위 기준선" if mode is None else "= random --mode full (이전 이름)")
        add_scene_args(b); add_output_args(b)
        b.add_argument("--n_train", type=int, default=300)
        if mode is None:
            b.add_argument("--mode", choices=["full", "upright", "pillar", "filtered"], required=True)
            b.add_argument("--s_min", type=float, default=None, help="filtered: 정규화 높이 하한(바닥 p5=0)")
            b.add_argument("--pillar_dir", choices=["full", "upright"], default="full",
                           help="pillar: 방향 full(Haar 회전, roll 포함) / upright(구면 균등 시선, roll 0)")
            b.add_argument("--pos_source", choices=["bbox", "pillar"], default="bbox", help="filtered: 위치 후보(bbox 균등 / 알고리즘의 구조물 근처 기둥 위치)")
            b.add_argument("--pitch_source", choices=["random", "aim"], default="random", help="filtered: pitch(U[-60,40] / 알고리즘 aim)")
            b.add_argument("--views_per_pos", type=int, default=1, help="filtered: 통과한 위치에서 50%% 겹치는 이웃 방향으로 최대 몇 장")
            b.add_argument("--pano_min", type=int, default=0, help="filtered: 둘레 완전도 — 12방향 중 깨끗한 방향 최소 수(0 = 끔)")
            b.add_argument("--far_check", action="store_true", help="filtered: 알고리즘 S3의 근접샷(p90 깊이)·빈 칸 기하 기준 추가")
            b.add_argument("--s_max", type=float, default=None, help="filtered: 정규화 높이 상한(구조물 p95=1)")
        else:
            b.set_defaults(mode=mode)
    h = sub.add_parser("human", help="사람 기준선(picker 포즈 JSON 렌더)"); add_scene_args(h); add_output_args(h, camera=False)
    h.add_argument("--poses", required=True)
    a = ap.parse_args()
    if a.cmd == "run" and a.preset == "final":
        for k, v in FINAL_PRESET.items():
            setattr(a, k, v)
    dict(run=cmd_run, random=cmd_random, human=cmd_human, **{"baseline-random": cmd_random})[a.cmd](a)


if __name__ == "__main__":
    main()
