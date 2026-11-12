"""
common.py
---------
Step 0-a / 0-b 실험에서 공유하는 유틸리티 모음.

의존성 (설치 필요):
    pip install pycolmap numpy pillow scikit-image torch torchvision --break-system-packages
    (렌더링용) 원본 3D Gaussian Splatting repo를 clone 후 sys.path에 추가:
        git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
        cd gaussian-splatting && pip install submodules/diff-gaussian-rasterization submodules/simple-knn

주의:
    getWorld2View2 / getProjectionMatrix / render() 호출부는 graphdeco-inria/gaussian-splatting
    repo의 공개 API(GaussianModel, gaussian_renderer.render)를 그대로 사용하되, 카메라 객체는
    이 repo의 내부 Camera 클래스(nn.Module, GT 이미지 필요) 대신 렌더링에만 필요한 최소 필드를
    담은 SimpleNamespace로 직접 구성합니다. 이는 GT 이미지 없이 임의 pose를 렌더링해야 하는
    우리 상황(novel pose synthesis)에 맞춘 선택입니다.
    3DGS repo는 버전에 따라 render() 시그니처가 조금씩 바뀌었으므로(예: separate_sh,
    use_trained_exp 인자 추가), 여러분 로컬 repo의 gaussian_renderer/__init__.py를 한 번
    확인하고 필요시 render_gaussians() 안의 호출부만 맞춰주세요.
    **반드시 Step 0-a(실제 학습 pose 재렌더링)를 먼저 돌려서 렌더링이 real image와
    시각적으로 맞는지 확인한 뒤 나머지 파이프라인을 신뢰하세요.**
"""
import math
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# 1. COLMAP pose loading
# ---------------------------------------------------------------------------

def load_colmap_poses(sparse_dir):
    """COLMAP sparse reconstruction(cameras.bin/images.bin)을 읽어
    이미지별 world-to-camera pose + intrinsics를 반환.

    Returns: list of dict, 각 원소:
        {
          "name": str,               # 원본 이미지 파일명 (e.g. "DSC_0001.JPG")
          "R_w2c": (3,3) ndarray,    # world -> camera rotation
          "t_w2c": (3,)  ndarray,    # world -> camera translation
          "cam_center": (3,) ndarray,# world 좌표계에서의 카메라 중심
          "focal_px": float,         # fx ~= fy 가정 (ACE 요구사항과 동일)
          "width": int,
          "height": int,
        }
    이름순으로 정렬해서 반환 (재현성 + ACE의 "alphabetical ordering" 대응 관행에 맞춤).
    """
    import pycolmap

    sparse_dir = Path(sparse_dir)
    recon = pycolmap.Reconstruction(str(sparse_dir))

    poses = []
    for image_id, image in recon.images.items():
        cam = recon.cameras[image.camera_id]

        # pycolmap API가 버전에 따라 달라서 여러 형태를 다 시도.
        #   - 일부 버전: image.cam_from_world 가 이미 계산된 Rigid3d 속성(property)
        #   - 일부 버전: image.cam_from_world() 를 호출해야 Rigid3d를 반환하는 메서드
        #   - 구버전: image.qvec / image.tvec
        try:
            cfw = image.cam_from_world
            if callable(cfw):
                cfw = cfw()  # 메서드인 경우 호출
            R_w2c = cfw.rotation.matrix()
            t_w2c = np.asarray(cfw.translation, dtype=np.float64)
        except AttributeError:
            try:
                R_w2c = qvec2rotmat(np.asarray(image.qvec, dtype=np.float64))
                t_w2c = np.asarray(image.tvec, dtype=np.float64)
            except AttributeError as e:
                raise AttributeError(
                    "이 pycolmap 버전의 Image 객체 API를 자동으로 못 찾았습니다. "
                    "다음을 실행해서 나온 결과를 알려주세요:\n"
                    "  import pycolmap; im = pycolmap.Reconstruction('<sparse/0 경로>').images\n"
                    "  im = list(im.values())[0]; print([a for a in dir(im) if not a.startswith('_')])"
                ) from e

        params = np.asarray(cam.params, dtype=np.float64)
        model = cam.model_name if hasattr(cam, "model_name") else str(cam.model)

        # 카메라 모델별로 focal length 파라미터 위치가 다름.
        # PINHOLE: fx, fy, cx, cy / SIMPLE_PINHOLE, SIMPLE_RADIAL: f, cx, cy, (k) ...
        if "SIMPLE" in model:
            fx = fy = params[0]
        else:
            fx, fy = params[0], params[1]

        if abs(fx - fy) > 1e-3 * max(fx, fy):
            # ACE(=DSAC*)는 fx==fy(단일 focal length)를 가정합니다.
            # 이미 노트에 정리하신 것처럼 정사각형 픽셀로 취급해 평균을 씁니다.
            print(f"[warn] {image.name}: fx({fx:.2f}) != fy({fy:.2f}), averaging.")
        focal_px = float((fx + fy) / 2.0)

        cam_center = -R_w2c.T @ t_w2c

        poses.append(dict(
            name=image.name,
            R_w2c=R_w2c,
            t_w2c=t_w2c,
            cam_center=cam_center,
            focal_px=focal_px,
            width=int(cam.width),
            height=int(cam.height),
        ))

    poses.sort(key=lambda p: p["name"])
    return poses


def load_gs_cameras_json(cameras_json_path):
    """3DGS 학습 결과 폴더에 자동 생성되는 cameras.json을 직접 읽는다.
    이 파일은 학습에 '실제로 쓰인' pose를 담고 있어 COLMAP 재파싱보다 신뢰도가 높다
    (예: --eval 모드로 학습했다면 train camera만 여기 들어있음).

    JSON 각 원소 스키마 (graphdeco-inria repo의 camera_to_JSON 출력):
        {"img_name": ..., "width":, "height":, "position": [x,y,z] (camera center, world 좌표),
         "rotation": 3x3 (camera-to-world 회전행렬), "fx":, "fy":}
    """
    entries = json.loads(Path(cameras_json_path).read_text())
    poses = []
    for e in entries:
        R_c2w = np.array(e["rotation"], dtype=np.float64)
        C = np.array(e["position"], dtype=np.float64)  # world 좌표계에서의 카메라 중심
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ C
        poses.append(dict(
            name=e["img_name"],
            R_w2c=R_w2c,
            t_w2c=t_w2c,
            cam_center=C,
            focal_px=float((e["fx"] + e["fy"]) / 2.0),
            width=int(e["width"]),
            height=int(e["height"]),
        ))
    poses.sort(key=lambda p: p["name"])
    return poses


def qvec2rotmat(qvec):
    """COLMAP 쿼터니언(w,x,y,z) -> 3x3 회전행렬. (구버전 pycolmap 대응용)"""
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


# ---------------------------------------------------------------------------
# 2. Minimal camera object + 3DGS render wrapper
# ---------------------------------------------------------------------------

def get_world2view2(R_w2c, t_w2c):
    """3DGS 컨벤션의 world-to-view 4x4 행렬.

    주의: 여기서 받는 R_w2c는 '진짜' world-to-camera 회전입니다(추가 transpose 불필요).
    graphdeco-inria repo의 원본 getWorld2View2(R, T)는 인자 R이 camera-to-world라고
    가정하고 내부에서 한 번 transpose하는데, 우리는 처음부터 world-to-camera를 갖고
    있으므로 여기서 또 transpose하면 이중으로 뒤집혀 회전이 거꾸로 들어갑니다.
    (이 버그로 실제로 렌더링 각도가 어긋난 적이 있었음 — 원인 확인 완료.)
    """
    Rt = np.zeros((4, 4), dtype=np.float32)
    Rt[:3, :3] = R_w2c
    Rt[:3, 3] = t_w2c
    Rt[3, 3] = 1.0
    return Rt  # world -> camera, 3DGS convention (caller가 .transpose(0,1) 적용)


def get_projection_matrix(znear, zfar, fov_x, fov_y):
    """표준 perspective projection (3DGS repo와 동일 공식). znear/zfar는
    scene scale에 맞게 조정 필요 (기본값은 대략적인 실내/실외 절충값)."""
    tan_half_fovy = math.tan(fov_y / 2)
    tan_half_fovx = math.tan(fov_x / 2)

    top = tan_half_fovy * znear
    bottom = -top
    right = tan_half_fovx * znear
    left = -right

    P = np.zeros((4, 4), dtype=np.float32)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def focal_to_fov(focal_px, pixels):
    return 2 * math.atan(pixels / (2 * focal_px))


def build_render_camera(R_w2c, t_w2c, focal_px, width, height, znear=0.01, zfar=100.0,
                         device="cuda"):
    """render()에 넘길 최소 카메라 객체 (GT 이미지 불필요)."""
    import torch

    fov_x = focal_to_fov(focal_px, width)
    fov_y = focal_to_fov(focal_px, height)

    world_view_transform = torch.tensor(get_world2view2(R_w2c, t_w2c)).transpose(0, 1).to(device)
    projection_matrix = torch.tensor(
        get_projection_matrix(znear, zfar, fov_x, fov_y)
    ).transpose(0, 1).to(device)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(
        projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    return SimpleNamespace(
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
        camera_center=camera_center,
        FoVx=fov_x,
        FoVy=fov_y,
        image_width=width,
        image_height=height,
        znear=znear,
        zfar=zfar,
    )


def render_gaussians(gaussians, cam, pipe, bg_color, gs_render_fn):
    """gs_render_fn: gaussian_renderer.render 함수를 그대로 전달.
    반환: (H,W,3) uint8 numpy array."""
    import torch

    with torch.no_grad():
        pkg = gs_render_fn(cam, gaussians, pipe, bg_color)
        img = pkg["render"]  # (3,H,W), float in [0,1] (모델/버전에 따라 clamp 필요할 수 있음)
        img = img.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
        return (img * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3. ACE(DSAC*) 포맷 I/O
#    scene/{split}/rgb, poses, calibration  (파일명 alphabetical하게 대응)
# ---------------------------------------------------------------------------

def w2c_to_ace_pose(R_w2c, t_w2c):
    """ACE/DSAC*는 pose 파일에 camera-to-world 4x4(h)를 저장하고
    y(scene coord) = h @ e(camera coord) 로 정의합니다."""
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_w2c.T
    c2w[:3, 3] = -R_w2c.T @ t_w2c
    return c2w


def write_ace_frame(root, split, frame_idx, image_uint8, R_w2c, t_w2c, focal_px):
    """단일 프레임을 ACE 포맷으로 저장."""
    root = Path(root)
    rgb_dir = root / split / "rgb"
    pose_dir = root / split / "poses"
    calib_dir = root / split / "calibration"
    for d in (rgb_dir, pose_dir, calib_dir):
        d.mkdir(parents=True, exist_ok=True)

    stem = f"frame_{frame_idx:06d}"
    Image.fromarray(image_uint8).save(rgb_dir / f"{stem}.color.png")

    c2w = w2c_to_ace_pose(R_w2c, t_w2c)
    np.savetxt(pose_dir / f"{stem}.pose.txt", c2w, fmt="%.8f")

    with open(calib_dir / f"{stem}.calibration.txt", "w") as f:
        f.write(f"{focal_px:.6f}\n")

    return stem


# ---------------------------------------------------------------------------
# 4. Quality metrics / logging
# ---------------------------------------------------------------------------

def compute_psnr_ssim(img_a_uint8, img_b_uint8):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    a, b = img_a_uint8, img_b_uint8
    if a.shape != b.shape:
        # 렌더 해상도와 real 이미지 해상도가 다르면 real을 렌더 크기로 리사이즈
        b = np.array(Image.fromarray(b).resize((a.shape[1], a.shape[0]), Image.LANCZOS))
    psnr = peak_signal_noise_ratio(a, b, data_range=255)
    ssim = structural_similarity(a, b, channel_axis=2, data_range=255)
    return float(psnr), float(ssim)


def laplacian_sharpness(img_uint8):
    """실제 대응 이미지가 없는 0-b 뷰의 렌더 품질 proxy. 값이 낮을수록 blur/floater 의심."""
    import cv2
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pose_distance_to_nearest(cam_center, real_cam_centers):
    """가장 가까운 real training camera와의 거리 (trust region 밖으로 나갔는지 확인용)."""
    d = np.linalg.norm(np.asarray(real_cam_centers) - cam_center[None, :], axis=1)
    return float(d.min())


def save_side_by_side(debug_dir, stem, rendered_uint8, real_uint8):
    """디버깅용: real | rendered 를 가로로 이어붙여 저장. 눈으로 바로 비교하기 위함."""
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    r = rendered_uint8
    real = real_uint8
    if real.shape[:2] != r.shape[:2]:
        real = np.array(Image.fromarray(real).resize((r.shape[1], r.shape[0]), Image.LANCZOS))
    combo = np.concatenate([real, np.full((r.shape[0], 4, 3), 255, dtype=np.uint8), r], axis=1)
    Image.fromarray(combo).save(debug_dir / f"{stem}_real_vs_rendered.png")


def sharpness_reference_from_dir(rgb_dir, max_images=None):
    """이미 검증된(예: 0-a) 렌더링 폴더에서 sharpness 분포를 계산.
    이 scene 고유의 텍스처 정도를 반영한 '건강한' 기준선으로 쓴다
    (scene마다 절대 sharpness 값 자체가 크게 달라서, 하나의 절대 threshold를
    모든 scene에 쓰면 안 된다는 게 이미 확인됐음 -> scene별 기준선 필수)."""
    paths = sorted(Path(rgb_dir).glob("*.png"))
    if max_images:
        paths = paths[:max_images]
    vals = []
    for p in paths:
        img = np.array(Image.open(p).convert("RGB"))
        vals.append(laplacian_sharpness(img))
    return np.array(vals)


# ---------------------------------------------------------------------------
# 5. Normal-guided viewpoint orientation (FC-Planner의 skeleton->surface ray
#    아이디어를, skeleton 없이 surface normal만으로 단순화해서 적용)
# ---------------------------------------------------------------------------

def estimate_normals(points, k=20):
    """포인트클라우드 각 점에서 로컬 PCA(최소 고유값의 고유벡터)로 normal을 추정.
    Open3D 없이 scipy KD-tree + numpy만 사용 (배치 eigh로 벡터화해서 빠름).

    반환되는 normal의 부호(표면의 안쪽/바깥쪽 중 어디를 향하는지)는 아직
    정해지지 않음 - 실제로 쓸 때(카메라 위치 기준)로 재정렬해야 함
    (orient_normal_toward 참고).

    Returns: (normals (N,3) ndarray, cKDTree) - tree는 이후 최근접 조회에 재사용
    """
    tree = cKDTree(points)
    k = min(k, len(points))
    _, idx = tree.query(points, k=k)
    neighbors = points[idx]  # (N, k, 3)
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', centered, centered) / k  # (N, 3, 3)
    # np.linalg.eigh는 마지막 두 축을 행렬로 보고 나머지 축(N)에 대해 배치로 동작
    # (오름차순 eigenvalue이므로 [...,0]이 가장 작은 고유값 = normal 방향)
    _, eigvecs = np.linalg.eigh(cov)
    normals = eigvecs[:, :, 0]
    return normals, tree


def orient_normal_toward(normal, from_point, to_point):
    """normal의 부호를, from_point -> to_point와 같은 쪽을 향하도록 재정렬.
    (예: from_point=표면점, to_point=카메라 위치 -> normal이 카메라 쪽을 향하게 됨)"""
    d = to_point - from_point
    return normal if np.dot(normal, d) > 0 else -normal


def look_at_rotation(forward_world, up_ref):
    """forward_world(카메라가 바라볼 world 방향, 단위벡터 아니어도 됨)와
    up_ref(world 기준 '위' 방향)로 OpenCV/COLMAP 컨벤션의 R_w2c(3x3)를 만든다.
    (카메라 로컬 축: x=right, y=down, z=forward — common.py 전체에서 쓰는
    world-to-camera 컨벤션과 일치)"""
    f = forward_world / (np.linalg.norm(forward_world) + 1e-8)
    up = up_ref / (np.linalg.norm(up_ref) + 1e-8)
    if abs(np.dot(f, up)) > 0.999:  # forward와 up이 거의 평행하면(짐벌락) 다른 기준축으로 대체
        up = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    right = np.cross(f, up)
    right /= (np.linalg.norm(right) + 1e-8)
    true_up = np.cross(right, f)
    true_up /= (np.linalg.norm(true_up) + 1e-8)
    R_c2w = np.stack([right, -true_up, f], axis=1)  # 열: camera x, y(=아래), z(=forward)를 world 좌표로
    return R_c2w.T  # world-to-camera


def remove_floater_points(points, k=20, std_ratio=2.0):
    """Statistical Outlier Removal: 각 점의 k-최근접 이웃까지 평균 거리를 구해서,
    그 값이 전체 평균보다 std_ratio*표준편차 이상 큰 점(=주변에 이웃이 성기게
    흩어진 점, floater로 의심)을 제거한다. Open3D의 remove_statistical_outlier와
    같은 개념이며, 여기서는 scipy KD-tree로 구현.

    Returns: (필터링된 points, keep_mask(원본 인덱스 기준 bool))
    """
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=min(k + 1, len(points)))  # +1: 자기 자신(거리 0) 포함되므로
    mean_knn_dist = dists[:, 1:].mean(axis=1)  # 자기 자신 제외하고 평균
    threshold = mean_knn_dist.mean() + std_ratio * mean_knn_dist.std()
    keep_mask = mean_knn_dist < threshold
    return points[keep_mask], keep_mask


def local_point_scale(points, k=8, sample_size=50000, rng=None):
    """이 점 구름이 '얼마나 촘촘한지'를 나타내는 로컬 스케일(전형적인 최근접 이웃 간격)을
    계산한다. scene_extent(bbox 대각선) 같은 전역 스칼라 대신 이 값을 min/max_offset의
    기준으로 쓰면, Street처럼 scene 전체는 거대하지만 실제 구조물은 촘촘한 경우에도
    "가깝다/멀다"가 항상 물리적으로 의미 있게 유지된다(전역 스케일에 비례해 커지지 않음).

    전체 점을 다 query하면 느릴 수 있어 sample_size개만 무작위로 뽑아 질의한다(트리 자체는
    전체 점으로 만든다 — 이웃 후보가 줄어들면 안 되므로).

    Returns: 전형적 이웃 거리의 median (스칼라).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    tree = cKDTree(points)
    if len(points) > sample_size:
        query_points = points[rng.choice(len(points), sample_size, replace=False)]
    else:
        query_points = points
    k = min(k + 1, len(points))
    dists, _ = tree.query(query_points, k=k)
    mean_knn_dist = dists[:, 1:].mean(axis=1)
    return float(np.median(mean_knn_dist))


# ---------------------------------------------------------------------------
# 6. Clean-guided viewpoint sampling: bounded voxelization + shell + greedy 선택
# ---------------------------------------------------------------------------

def voxel_occupancy_and_distance(clean_points, lo, hi, voxel_size, density_threshold):
    """Bounded voxelization의 공통 부분: occupied voxel 그리드와, 모든 voxel에서
    '가장 가까운 occupied voxel까지의 거리' 그리드를 계산한다.

    - occupied voxel: density_threshold개 이상의 clean_points를 포함한 voxel
    - scipy.ndimage.distance_transform_edt로 전체 거리 그리드를 한 번에 계산
      (voxel 개수만큼 KD-tree 쿼리를 반복하는 것보다 훨씬 빠름).

    voxel_shell_candidates()와 진단/시각화 스크립트(inspect_voxel_occupancy.py)가
    이 함수를 공유해서, 시각화가 실제 알고리즘과 어긋나지 않게 한다.

    Returns: (occ, dist, grid_shape, lo) — occ/dist는 grid_shape 모양의 배열.
    """
    from scipy.ndimage import distance_transform_edt

    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    grid_shape = np.maximum(np.ceil((hi - lo) / voxel_size).astype(int), 1)

    idx = np.floor((clean_points - lo) / voxel_size).astype(int)
    valid = np.all((idx >= 0) & (idx < grid_shape), axis=1)
    idx = idx[valid]

    occ = np.zeros(grid_shape, dtype=bool)
    if len(idx) > 0:
        flat = np.ravel_multi_index(idx.T, grid_shape)
        counts = np.bincount(flat, minlength=int(np.prod(grid_shape))).reshape(grid_shape)
        occ = counts >= density_threshold

    if not occ.any():
        raise ValueError("occupied voxel이 하나도 없습니다. density_threshold나 "
                          "voxel_size(grid_resolution)를 조정하세요.")

    dist = distance_transform_edt(~occ, sampling=[voxel_size] * 3)
    return occ, dist, grid_shape, lo


def voxel_shell_candidates(clean_points, lo, hi, voxel_size, density_threshold,
                            min_offset, max_offset, rng, n_candidates):
    """Bounded voxelization으로 '표면을 감싸는 빈 공간(shell)'을 찾고, 그 안에서
    n_candidates개의 연속 좌표(X,Y,Z) 후보를 랜덤 샘플링한다.

    shell: occupied가 아니면서, 가장 가까운 occupied voxel까지의 거리가
    (min_offset, max_offset) 사이인 voxel (너무 가까우면 표면에 박힘, 너무 멀면
    재구성 안 된 영역). occupied/거리 계산은 voxel_occupancy_and_distance() 참고.
    """
    occ, dist, grid_shape, lo = voxel_occupancy_and_distance(
        clean_points, lo, hi, voxel_size, density_threshold)

    shell_mask = (~occ) & (dist > min_offset) & (dist < max_offset)
    shell_indices = np.argwhere(shell_mask)  # (M, 3)

    if len(shell_indices) == 0:
        raise ValueError("유효한 shell voxel이 없습니다. min_offset/max_offset/"
                          "voxel_size를 조정하세요.")

    chosen = shell_indices[rng.integers(0, len(shell_indices), size=n_candidates)]
    jitter = rng.uniform(0, voxel_size, size=(n_candidates, 3))
    positions = lo + chosen * voxel_size + jitter
    return positions


# ---------------------------------------------------------------------------
# 6b. Grid-free, Gaussian-volume-aware free space (2026-09-18 재설계)
#
# voxel_shell_candidates()의 문제 두 가지를 해결하기 위한 대안:
#   (a) voxel_size가 스칼라 하나라서 scene의 수평/수직 스케일이 크게 다르면(Street처럼)
#       한쪽 축의 해상도가 완전히 붕괴할 수 있다.
#   (b) "occupied"를 Gaussian 중심점 개수로만 판단해서, 크고 납작한 Gaussian(벽 등)
#       사이의 빈 중심점 간격을 실제로는 표면 안쪽인데도 "비어있다"고 착각할 수 있다.
#
# 해결: voxel 격자 자체를 없애고, 각 Gaussian을 반지름 k_sigma*max(scale)인 구(sphere)로
# 근사해서(회전은 무시 — 실제 타원체보다 크게 잡으므로 안전한 쪽으로 보수적), 어느
# 연속 좌표에서든 "가장 가까운 Gaussian 표면까지 거리"를 KD-tree로 직접 계산한다.
# 이러면 voxel_size라는 파라미터 자체가 없어져서 scene 스케일에 관계없이 항상 올바르고,
# Gaussian의 실제 부피도 반영된다.
# ---------------------------------------------------------------------------

def gaussian_effective_radius(scales, k_sigma=2.5):
    """scale(N,3, 이미 exp() 적용된 실제 물리 스케일)에서 각 Gaussian을 감싸는 보수적인
    구(sphere) 반지름을 계산한다. 회전을 무시하고 세 축 중 가장 큰 값을 쓰므로, 납작하거나
    길쭉한 Gaussian도 절대 과소평가(=실제보다 작게 잡아 안을 뚫고 지나가는 일)하지 않는다."""
    return k_sigma * scales.max(axis=1)


def gaussian_surface_distance(query_points, centers, radii, k_neighbors=16, tree=None):
    """query_points 각각에서 '가장 가까운 Gaussian 표면까지의 부호 거리'를 근사 계산.

    각 query point에서 중심점 기준 최근접 k_neighbors개를 찾고, 그중
    (중심까지 거리 - 그 Gaussian의 반지름)의 최솟값을 표면 거리로 쓴다. 정확한 볼록껍질
    거리는 아니지만(주변 Gaussian이 몰려있으면 진짜 최근접 표면을 놓칠 수 있음), k를
    적당히 크게 주면(기본 16) 실용적으로 충분하다.

    Returns: (N,) 배열. 0 이하 = 어떤 Gaussian의 반지름 안(occupied). 클수록 표면에서 멂.
    """
    if tree is None:
        tree = cKDTree(centers)
    k = min(k_neighbors, len(centers))
    center_dists, idx = tree.query(query_points, k=k)
    if k == 1:
        center_dists = center_dists[:, None]
        idx = idx[:, None]
    surf_dists = center_dists - radii[idx]
    return surf_dists.min(axis=1)


def surface_shell_candidates(struct_points, radii, lo, hi, min_offset, max_offset,
                              rng, n_candidates, oversample=4, k_neighbors=16):
    """voxel_shell_candidates()의 grid-free 대체. 실제 구조물(struct_points) 근처에서
    바깥쪽으로 무작위 방향 x 무작위 거리만큼 떨어진 지점을 제안하고, 그 지점의 진짜
    표면 거리(gaussian_surface_distance)가 (min_offset, max_offset) 사이인 것만 채택한다.

    scene 전체를 uniform하게 rejection-sampling하는 대신 '실제 구조물 근처에서 시작해
    바깥으로 한 걸음 나간다'는 전략이라, Street처럼 bbox 대부분이 텅 빈 야외 scene에서도
    효율적이다(무작위 앵커가 실제 구조물이므로 성공률이 항상 준수함).
    """
    tree = cKDTree(struct_points)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)

    collected = []
    budget = n_candidates
    attempts = 0
    max_attempts = 50
    while budget > 0 and attempts < max_attempts:
        m = budget * oversample
        anchor_idx = rng.integers(0, len(struct_points), size=m)
        anchors = struct_points[anchor_idx]

        directions = rng.normal(size=(m, 3))
        directions /= (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)
        offsets = rng.uniform(min_offset, max_offset, size=m)
        proposals = anchors + directions * offsets[:, None]

        in_bbox = np.all((proposals >= lo) & (proposals <= hi), axis=1)
        proposals = proposals[in_bbox]
        if len(proposals) == 0:
            attempts += 1
            continue

        surf_dist = gaussian_surface_distance(proposals, struct_points, radii,
                                               k_neighbors=k_neighbors, tree=tree)
        valid = (surf_dist > min_offset) & (surf_dist < max_offset)
        collected.append(proposals[valid])
        budget = n_candidates - sum(len(c) for c in collected)
        attempts += 1

    if not collected or sum(len(c) for c in collected) == 0:
        raise ValueError("유효한 shell 후보가 없습니다. min_offset_ratio/max_offset_ratio나 "
                          "k_sigma를 조정하세요.")
    positions = np.concatenate(collected, axis=0)
    if len(positions) > n_candidates:
        positions = positions[rng.choice(len(positions), n_candidates, replace=False)]
    return positions


def farthest_point_positions(points, n, rng, weight=None):
    """Greedy 최원점 선택(farthest-point sampling). 이미 뽑힌 점들까지의 최소 거리가 가장 큰
    점을 반복해서 뽑아, 학습 데이터의 공간적 분포를 균등하게 편다. `greedy_select`(점수 기반,
    반경 안 점수를 깎는 방식)와 달리 순수 기하학적 커버리지만 본다 — 점수를 안 쓰는 step0d의
    "무작위 균등 선택" 철학과 맞으면서, random.choice보다 공간 분포가 실제로 고르다
    (random.choice는 큰 후보 풀에서도 우연히 한 구역에 몰릴 수 있음).

    weight(길이 len(points), 선택): 주어지면 (최소거리 * weight)를 기준으로 뽑아, 구조가
    조밀한 곳을 살짝 더 선호하면서도 공간적으로 퍼지게 할 수 있다. 기본 None = 순수 균등.

    Returns: 선택된 인덱스 배열 (points 기준), 길이 min(n, len(points)).
    """
    n = min(n, len(points))
    m = len(points)
    selected = np.empty(n, dtype=np.int64)
    first = int(rng.integers(m))
    selected[0] = first
    min_dist = np.linalg.norm(points - points[first], axis=1)
    for i in range(1, n):
        crit = min_dist * weight if weight is not None else min_dist
        nxt = int(np.argmax(crit))
        selected[i] = nxt
        d = np.linalg.norm(points - points[nxt], axis=1)
        min_dist = np.minimum(min_dist, d)
    return selected


def multi_object_anchors(struct_points, n_anchors, rng, sample_size=50000):
    """구조물 포인트를 대표하는 n_anchors개의 '물체/영역 대표점'을 뽑고, 전체 구조물 포인트를
    가장 가까운 대표점에 배정해 대략적인 클러스터 id로 쓴다. k-means 대신 FPS + 최근접 배정을
    쓰는 이유: 반복 없이 한 번에 계산되면서도(가벼움) 공간을 고르게 나눈다(밀도가 다른
    영역에서도 한쪽에 대표점이 쏠리지 않음). 이 클러스터 id는 "한 뷰에 서로 다른 물체가 몇 개
    보이는지"를 세는 용도로만 쓰이며, 실제 물체 분할이 아니라 거친 공간적 구획일 뿐이다.

    Returns: anchors (n_anchors, 3), point_cluster_id (len(struct_points),) — struct_points와
    같은 순서.
    """
    pts = struct_points
    if len(pts) > sample_size:
        sample = pts[rng.choice(len(pts), sample_size, replace=False)]
    else:
        sample = pts
    anchor_idx = farthest_point_positions(sample, n_anchors, rng)
    anchors = sample[anchor_idx]
    _, cluster_id = cKDTree(anchors).query(struct_points, k=1)
    return anchors, cluster_id


def frustum_visible_mask(points, R_w2c, cam_center, focal_px, width, height, min_dist, max_dist):
    """world 좌표 points가 카메라(회전 R_w2c, 위치 cam_center, 핀홀 focal_px/width/height) 시야
    안에 들어오는지, 그리고 깊이가 (min_dist, max_dist) 사이인지의 boolean mask.
    OpenCV/이 저장소 컨벤션(z-forward, w2c 회전)과 동일: cam_coords = R_w2c @ (world - center).
    """
    cam = (points - cam_center) @ R_w2c.T
    z = cam[:, 2]
    zz = np.maximum(z, 1e-6)
    u = focal_px * cam[:, 0] / zz + width / 2.0
    v = focal_px * cam[:, 1] / zz + height / 2.0
    return (z > min_dist) & (z < max_dist) & (u >= 0) & (u < width) & (v >= 0) & (v < height)


def band_filtered_shell_candidates(struct_points, radii, lo, hi, min_offset, max_offset,
                                    up_ref, band_lo_h, band_hi_h, rng, n_candidates,
                                    k_neighbors=16, oversample=3, max_rounds=8):
    """surface_shell_candidates()가 만든 후보 중 up_ref 방향 투영(높이)이 [band_lo_h, band_hi_h]
    (카메라 높이 밴드) 안에 드는 것만 남긴다.

    up_ref가 좌표축과 정렬돼 있으면(예: (0,1,0)) lo/hi의 해당 좌표를 직접 밴드로 잘라 넣는 게
    더 효율적이지만, 실제 3DGS(-MCMC) 체크포인트의 up은 보통 좌표축과 정렬돼 있지 않다(예:
    Mip-NeRF 360 원본 COLMAP 프레임을 그대로 쓰는 mcmc 모델들은 up이 대각선 벡터 — 실측:
    counter (0.06,-0.86,-0.51), bonsai (0.08,-0.77,-0.64) 등). 그래서 lo/hi는 그대로 (수평)
    percentile bbox로만 쓰고, 높이 제한은 여기서 `struct_points @ up_ref`(임의의 벡터에 대해서도
    성립하는 일반 투영)로 사후 필터링한다. 밴드가 전체 높이의 일부일수록 통과율이 낮아지므로
    부족하면 요청량을 두 배씩 늘려 재시도한다(최대 max_rounds회).
    """
    collected = []
    ask = n_candidates * oversample
    for _ in range(max_rounds):
        batch = surface_shell_candidates(struct_points, radii, lo, hi, min_offset, max_offset,
                                          rng, ask, k_neighbors=k_neighbors)
        h = batch @ up_ref
        collected.append(batch[(h >= band_lo_h) & (h <= band_hi_h)])
        if sum(len(c) for c in collected) >= n_candidates:
            break
        ask *= 2
    total = sum(len(c) for c in collected)
    if total == 0:
        raise ValueError("카메라 높이 밴드 안의 후보가 없습니다. --cam_height_pct_lo/hi나 "
                          "up_axis 값을 확인하세요.")
    positions = np.concatenate(collected, axis=0)
    if len(positions) > n_candidates:
        positions = positions[rng.choice(len(positions), n_candidates, replace=False)]
    return positions


def local_geometry_stats(clean_points, clean_opacities, tree, query_points, k=20):
    """query_points 각각의 주변 local geometry에서 렌더링 없이 계산 가능한
    informativeness proxy 3종을 계산.

    - density: k-최근접 이웃까지 평균 거리 (작을수록 촘촘 = 신뢰도 높음)
    - scatter_score: 로컬 PCA 세 고유값의 균등도, 0(완전 평평)~1(사방으로 복잡/코너)
      로 정규화 (지난번 논의한 keypoint-entropy 대체 지표)
    - opacity_conf: 주변 점들의 평균 opacity (낮으면 불확실한 영역)
    """
    k = min(k, len(clean_points))
    dists, idxs = tree.query(query_points, k=k)
    density = dists.mean(axis=1)

    neighbors = clean_points[idxs]  # (Q, k, 3)
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    cov = np.einsum('qki,qkj->qij', centered, centered) / k
    eigvals = np.linalg.eigvalsh(cov)  # (Q, 3), 오름차순, 전부 >= 0
    eigvals = np.clip(eigvals, 1e-12, None)
    normalized = eigvals / eigvals.sum(axis=1, keepdims=True)  # 세 값의 합이 1

    # normalized가 [1/3,1/3,1/3]에 가까울수록(분산 0) '사방으로 복잡'하다는 뜻,
    # [~0,~0,~1]에 가까울수록(분산 최대 2/9) '평평하다'는 뜻.
    # 분산을 0~1로 뒤집어서 스케일 (1=복잡, 0=평평).
    var = normalized.var(axis=1)
    scatter_score = 1.0 - var / (2.0 / 9.0)
    scatter_score = np.clip(scatter_score, 0.0, 1.0)

    opacity_conf = clean_opacities[idxs].mean(axis=1)

    return density, scatter_score, opacity_conf


def greedy_select(positions, scores, n, diversity_radius, penalty_weight=2.0):
    """Gravitation 스타일 greedy 선택: 매 반복마다 점수가 제일 높은 후보를 뽑고,
    그 주변(diversity_radius 이내) 후보들의 점수를 거리에 비례해서 깎는다
    (이미 뽑힌 곳 근처가 계속 또 뽑히는 걸 방지 -> 공간적으로 퍼지게 됨).

    Returns: 선택된 후보의 인덱스 배열 (positions/scores 기준 인덱스)
    """
    n = min(n, len(positions))
    remaining_mask = np.ones(len(positions), dtype=bool)
    adjusted = scores.astype(np.float64).copy()
    selected = []

    for _ in range(n):
        idx_pool = np.where(remaining_mask)[0]
        best = idx_pool[np.argmax(adjusted[idx_pool])]
        selected.append(int(best))
        remaining_mask[best] = False

        d = np.linalg.norm(positions - positions[best], axis=1)
        decay = np.clip(1.0 - d / diversity_radius, 0.0, None)
        adjusted -= penalty_weight * decay

    return np.array(selected)


def check_near_camera_floaters(candidate_positions, forwards, floater_tree, near_dist,
                                half_angle_deg=35.0):
    """각 후보 카메라의 시야(forward 방향, half_angle_deg 반각의 원뿔) 안,
    near_dist 이내에 floater(낮은 opacity 또는 SOR로 제거된 고립점)가 있는지 체크.
    카메라 바로 앞을 floater가 가리는 후보를 걸러내기 위함 (ray-casting의 단순화 버전:
    실제 ray marching 대신, '시야 원뿔 안에 floater 점이 있는가'로 근사).

    주의: forwards는 world-to-camera 회전(R_w2c)의 3번째 '행'(row), 즉 forward =
    R_w2c[2,:] = R_w2c.T @ [0,0,1] 로 계산해서 넘겨야 합니다 (R_w2c의 3번째 '열'이
    아님 - 이 부분에서 이전에 혼동이 있었습니다, 확인 감사합니다).

    Returns: bool 배열 (True면 근접 floater가 감지된 위험한 후보)
    """
    cos_half_angle = np.cos(np.deg2rad(half_angle_deg))
    flags = np.zeros(len(candidate_positions), dtype=bool)
    if floater_tree is None or floater_tree.n == 0:
        return flags

    floater_pts = floater_tree.data
    nearby_lists = floater_tree.query_ball_point(candidate_positions, r=near_dist)
    for i, idxs in enumerate(nearby_lists):
        if not idxs:
            continue
        vecs = floater_pts[idxs] - candidate_positions[i]
        dists = np.linalg.norm(vecs, axis=1)
        valid = dists > 1e-8
        if not valid.any():
            flags[i] = True  # floater가 카메라 위치에 거의 겹침
            continue
        cos_angles = (vecs[valid] @ forwards[i]) / dists[valid]
        if np.any(cos_angles > cos_half_angle):
            flags[i] = True
    return flags


def save_top_k_grid(out_path, image_paths_and_labels, cols=5, thumb_w=320):
    """여러 이미지를 라벨과 함께 grid로 합쳐서 저장 (top-K 렌더링 결과 육안 검수용).
    image_paths_and_labels: [(image_path, label_str), ...] 리스트, 순서대로 배치됨."""
    from PIL import Image, ImageDraw

    imgs = []
    for path, label in image_paths_and_labels:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        thumb_h = int(h * thumb_w / w)
        imgs.append((img.resize((thumb_w, thumb_h)), label))

    thumb_h = imgs[0][0].height
    label_h = 24
    n = len(imgs)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols

    grid = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), (20, 20, 20))
    draw = ImageDraw.Draw(grid)
    for i, (img, label) in enumerate(imgs):
        r, c = divmod(i, cols)
        x, y = c * thumb_w, r * (thumb_h + label_h)
        grid.paste(img, (x, y))
        draw.rectangle([x, y + thumb_h, x + thumb_w, y + thumb_h + label_h], fill=(20, 20, 20))
        draw.text((x + 4, y + thumb_h + 4), label, fill=(255, 255, 255))
    grid.save(out_path)


def yaw_sweep_forwards(base_forward, up_ref, n_views):
    """base_forward를 up_ref 축을 중심으로 360도 균등하게 회전시킨 forward 벡터들을
    생성한다 (수평 파노라마 스윕). 인접한 두 뷰는 각도 차이가 (360/n_views)도라서,
    이 값이 카메라 FOV보다 작으면 자동으로 시야가 겹친다 (overlap) - 이게 descriptor
    학습에 필요한 multi-view 겹침을 후보 생성 단계에서부터 구조적으로 보장하는 방법.

    Returns: (n_views, 3) forward 벡터 배열
    """
    base_forward = base_forward / (np.linalg.norm(base_forward) + 1e-8)
    up_ref = up_ref / (np.linalg.norm(up_ref) + 1e-8)
    forwards = []
    for i in range(n_views):
        angle = 2 * np.pi * i / n_views
        rot = Rotation.from_rotvec(up_ref * angle)
        forwards.append(rot.apply(base_forward))
    return np.array(forwards)


def hub_views_per_overlap(fov_h_rad, overlap_ratio):
    """수평 FOV와 원하는 오버랩 비율로부터, 360도를 다 덮는 데 필요한 hub당
    뷰 개수를 계산한다. delta_theta = FOV_h * (1 - overlap_ratio)."""
    delta_theta = fov_h_rad * (1.0 - overlap_ratio)
    delta_theta = max(delta_theta, np.deg2rad(1.0))  # 0 나눗셈 방지
    return int(np.ceil(2 * np.pi / delta_theta))


def visualize_top_novel_views_3d(out_image_path, clean_xyz, real_centers, candidate_positions, selected_idx, R_list, top_frame_indices, scene_extent):
    """
    선별된 Top Novel Views 전용 3D Visualizer:
    - 전체 Point Cloud (Z축 Rainbow)
    - Real Cameras (Grey)
    - Top Novel View Cameras (Cyan Diamond) + 시선 화살표 (Lime Green)
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[Warning] plotly가 설치되어 있지 않습니다.")
        return

    # [1] 아웃라이어 필터링
    p_low, p_high = np.percentile(clean_xyz, [1, 99], axis=0)
    inside_mask = np.all((clean_xyz >= p_low) & (clean_xyz <= p_high), axis=1)
    filtered_xyz = clean_xyz[inside_mask]

    if len(filtered_xyz) > 30000:
        idx = np.random.choice(len(filtered_xyz), 30000, replace=False)
        vis_xyz = filtered_xyz[idx]
    else:
        vis_xyz = filtered_xyz

    fig = go.Figure()

    # [2] Terrain Point Cloud (Rainbow)
    fig.add_trace(go.Scatter3d(
        x=vis_xyz[:, 0], y=vis_xyz[:, 1], z=vis_xyz[:, 2],
        mode='markers',
        marker=dict(size=1.5, color=vis_xyz[:, 2], colorscale='Jet', opacity=0.3),
        name='3DGS Map'
    ))

    # [3] Real Cameras (Light Gray)
    fig.add_trace(go.Scatter3d(
        x=real_centers[:, 0], y=real_centers[:, 1], z=real_centers[:, 2],
        mode='markers',
        marker=dict(size=3, color='lightgray'),
        name='Real Cameras'
    ))

    # [4] Top Novel View Cameras
    top_pos = candidate_positions[top_frame_indices]
    fig.add_trace(go.Scatter3d(
        x=top_pos[:, 0], y=top_pos[:, 1], z=top_pos[:, 2],
        mode='markers',
        marker=dict(size=7, color='cyan', symbol='diamond'),
        name='Top Novel Views'
    ))

    # [5] Camera Look-At Direction
    line_x, line_y, line_z = [], [], []
    vec_len = scene_extent * 0.15
    for idx in top_frame_indices:
        pos = candidate_positions[idx]
        R_w2c = R_list[idx]
        forward = R_w2c.T @ np.array([0, 0, 1])
        target = pos + forward * vec_len

        line_x.extend([pos[0], target[0], None])
        line_y.extend([pos[1], target[1], None])
        line_z.extend([pos[2], target[2], None])

    fig.add_trace(go.Scatter3d(
        x=line_x, y=line_y, z=line_z,
        mode='lines',
        line=dict(color='lime', width=4),
        name='Top Novel Look-At'
    ))

    fig.update_layout(
        title="Top Novel Views 3D Inspection Visualizer",
        scene=dict(
            xaxis=dict(range=[p_low[0], p_high[0]]),
            yaxis=dict(range=[p_low[1], p_high[1]]),
            zaxis=dict(range=[p_low[2], p_high[2]]),
            aspectmode='data'
        )
    )

    html_path = out_image_path.with_suffix('.html')
    fig.write_html(str(html_path))
    print(f"  [Visual] Top Novel Views 3D Inspection saved to: {html_path}")


# ---------------------------------------------------------------------------
# 7. No-reference IQA (BRISQUE/NIQE via pyiqa) - real 대응 이미지 없이도 쓸 수
#    있는 품질 지표. sharpness(Laplacian)보다 "사람이 느끼는 부자연스러움"에
#    더 가까움 (floater로 인한 색/질감 이상은 sharpness로는 못 잡는 경우가 있었음).
# ---------------------------------------------------------------------------

def load_iqa_metrics(names, device="cuda"):
    """pyiqa 기반 no-reference IQA metric들을 미리 로드 (프레임마다 새로 만들면 느림).
    names 예: ["brisque", "niqe"]. pip install pyiqa 필요 (torch 기반).
    두 metric 다 '낮을수록 좋은 품질'입니다."""
    import pyiqa
    metrics = {}
    for name in names:
        metrics[name] = pyiqa.create_metric(name, device=device)
    return metrics


def compute_iqa_scores(metrics, img_uint8, device="cuda"):
    """img_uint8: (H,W,3) RGB uint8. 각 metric에 (1,3,H,W) 0~1 텐서로 넣어 점수 계산."""
    import torch
    t = torch.from_numpy(img_uint8.astype(np.float32) / 255.0)
    t = t.permute(2, 0, 1).unsqueeze(0).to(device)
    scores = {}
    with torch.no_grad():
        for name, metric in metrics.items():
            scores[name] = float(metric(t).item())
    return scores