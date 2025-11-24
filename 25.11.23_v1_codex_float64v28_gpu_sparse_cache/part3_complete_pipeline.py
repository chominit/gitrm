# -*- coding: utf-8 -*-
"""
part3_complete_pipeline.py
완전한 Forward + Backward 파이프라인 (모든 사이트 처리)
Scanline 방식으로 메모리 효율적 처리
"""

from pathlib import Path
import numpy as np
from typing import List, Tuple, Dict
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import constants as C
from camera_io import load_camera_db
from camera_calibration import get_camera_matrix, get_image_size, compute_camera_footprint_bbox
from img_mask import img_mask_diff
from gsd_parser import get_tolerance
from geometry import r_from_opk
from tqdm import tqdm
import laspy

# ==========================================
# 1. 공통 함수
# ==========================================

def load_point_cloud(las_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """포인트 클라우드 로드"""
    coords_path = las_dir / "coords_float64.npy"
    colors_path = las_dir / "colors_uint8.npy"

    if coords_path.exists() and colors_path.exists():
        coords = np.load(coords_path)
        colors = np.load(colors_path)
        print(f"   포인트: {len(coords):,}")
        return coords, colors
    else:
        raise FileNotFoundError(f"NPY 캐시 없음: {las_dir}")


def compute_row_ground_band(row_idx: int, img_height: int, img_width: int,
                            C_cam: np.ndarray, R_c2w: np.ndarray, K: np.ndarray,
                            ground_Z: float, debug: bool = False) -> Tuple[float, float, float, float]:
    """Row의 지면 투영 영역 계산"""
    K_inv = np.linalg.inv(K)

    # Row의 샘플 픽셀들
    sample_cols = np.linspace(0, img_width-1, 5, dtype=int)

    ground_points = []

    for col in sample_cols:
        pixel = np.array([col, row_idx, 1.0])
        ray_cam = K_inv @ pixel
        ray_world = R_c2w @ ray_cam
        # Ray 방향: z>0이면(위쪽) 뒤집어서 항상 지면(음의 z) 쪽을 보도록 보정
        if ray_world[2] > 0:
            ray_world = -ray_world
        ray_world = ray_world / np.linalg.norm(ray_world)

        # UAV는 아래를 촬영: ray_world[2]가 음수일 때 지면과 교차
        if abs(ray_world[2]) > 0.001:
            t = (ground_Z - C_cam[2]) / ray_world[2]
            if t != 0 and abs(t) > 0.1:
                ground_point = C_cam + t * ray_world
                # Ground Z 근처인지 확인
                if abs(ground_point[2] - ground_Z) < 50.0:
                    ground_points.append(ground_point[:2])
                elif debug:
                    print(f"         Ground point Z={ground_point[2]:.1f} too far from {ground_Z:.1f}")
            elif debug:
                print(f"         Ray t={t:.2f} too close, ray_z={ray_world[2]:.3f}")
        elif debug:
            print(f"         Ray nearly horizontal: ray_z={ray_world[2]:.6f}")

    if len(ground_points) == 0:
        if debug:
            print(f"         No valid ground intersections for row {row_idx}")
        return None

    ground_points = np.array(ground_points)

    # 여유 마진
    margin = 1.0
    x_min = ground_points[:, 0].min() - margin
    x_max = ground_points[:, 0].max() + margin
    y_min = ground_points[:, 1].min() - margin
    y_max = ground_points[:, 1].max() + margin

    return (x_min, x_max, y_min, y_max)


# ==========================================
# 2. FORWARD PASS (Scanline)
# ==========================================

def process_rays_gpu(rays: 'cp.ndarray', origin: 'cp.ndarray',
                     coords: 'cp.ndarray', h_tol: float, v_tol: float) -> 'cp.ndarray':
    """GPU ray casting - 이중 배치 (ray batch × point batch)"""
    import cupy as cp

    N = len(rays)
    M = len(coords)

    if M == 0:
        return cp.full(N, -1, dtype=cp.int32)

    # 배치 크기 설정 (기본값, GPU 활용 극대화)
    MAX_RAY_BATCH = 50000      # Ray 배치: 5만개씩
    MAX_POINT_BATCH = 120000    # Point 배치: 12만개씩

    # GPU 메모리 상황에 따라 동적으로 배치 크기 조정
    try:
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        max_ratio = getattr(C, "MAX_GPU_MEMORY_USAGE_RATIO", 0.9)
        max_allowed_bytes = int(total_bytes * max_ratio)
        current_used = total_bytes - free_bytes
        available_for_us = max_allowed_bytes - current_used

        if available_for_us > 0:
            # float64 기준 대략적인 per (ray, point) 메모리 (여유를 위해 64 bytes로 가정)
            BYTES_PER_PAIR = 64
            max_pairs = max(1, available_for_us // BYTES_PER_PAIR)
            max_ray_batch_by_mem = max_pairs // max(1, MAX_POINT_BATCH)
            if max_ray_batch_by_mem <= 0:
                MAX_POINT_BATCH = max(1000, max_pairs // max(1, MAX_RAY_BATCH))
                max_ray_batch_by_mem = max(1, max_pairs // MAX_POINT_BATCH)
            MAX_RAY_BATCH = max(1000, min(MAX_RAY_BATCH, max_ray_batch_by_mem))
    except Exception:
        # memGetInfo 사용 불가하거나 CPU 폴백인 경우 기본값 사용
        pass

    best_indices = cp.full(N, -1, dtype=cp.int32)
    best_depths = cp.full(N, cp.inf, dtype=cp.float64)

    # Ray 배치 루프
    for ray_start in range(0, N, MAX_RAY_BATCH):
        ray_end = min(ray_start + MAX_RAY_BATCH, N)
        ray_batch = rays[ray_start:ray_end]
        N_batch = len(ray_batch)

        batch_indices = cp.full(N_batch, -1, dtype=cp.int32)
        batch_depths = cp.full(N_batch, cp.inf, dtype=cp.float64)

        # Point 배치 루프
        for pt_start in range(0, M, MAX_POINT_BATCH):
            pt_end = min(pt_start + MAX_POINT_BATCH, M)
            pt_batch = coords[pt_start:pt_end]

            # Distance 계산 (N_batch × M_batch)
            P = pt_batch - origin
            proj_dist = cp.dot(ray_batch, P.T)

            # 카메라 앞만
            valid = proj_dist > 0.1

            # 투영점
            proj_pts = origin + ray_batch[:, None, :] * proj_dist[:, :, None]

            # Residual
            res = pt_batch[None, :, :] - proj_pts
            h_dist = cp.linalg.norm(res[:, :, :2], axis=2)
            v_dist = cp.abs(res[:, :, 2])

            # GSD 체크
            hits = valid & (h_dist <= h_tol) & (v_dist <= v_tol)

            # Best hit 업데이트 (ray batch 내에서)
            for i in range(N_batch):
                hit_idx = cp.where(hits[i])[0]
                if len(hit_idx) > 0:
                    depths = proj_dist[i, hit_idx]
                    min_idx = cp.argmin(depths)
                    if depths[min_idx] < batch_depths[i]:
                        batch_depths[i] = depths[min_idx]
                        batch_indices[i] = pt_start + hit_idx[min_idx]

        # 전체 결과에 합치기
        best_depths[ray_start:ray_end] = batch_depths
        best_indices[ray_start:ray_end] = batch_indices

    return best_indices


def forward_scanline(site_name: str, coords: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Forward pass - Scanline 방식"""
    import cupy as cp

    print(f"\n[FORWARD] {site_name}")

    # 설정
    det_dir = C.DETECTION_DIRS.get(site_name)
    output_dir = C.PART3_DIR / site_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not det_dir or not det_dir.exists():
        print(f"   [SKIP] Detection 디렉토리 없음")
        return np.zeros(len(coords), dtype=np.int32)

    cam_db = load_camera_db(site_name)

    # 지면 필터링 비활성화: 모든 포인트 사용
    fg_coords = coords
    fg_colors = colors
    print(f"   Ground filtering: DISABLED, using all points ({len(fg_coords):,})", flush=True)

    # Ground Z
    Z_values = [cam['C'][2] for cam in cam_db['images'].values()]
    Z_avg = float(np.mean(Z_values))
    H_site = C.FLIGHT_ALT_BY_SITE.get(site_name, 80.0)
    ground_Z = Z_avg - H_site

    print(f"   Z_avg: {Z_avg:.2f}m, H_site: {H_site:.1f}m, ground_Z: {ground_Z:.2f}m")

    # 파라미터
    try:
        print(f"   Getting camera matrix...", flush=True)
        K = get_camera_matrix(site_name)
        print(f"   Got K matrix: {K.shape}", flush=True)
        K_inv = np.linalg.inv(K)
        print(f"   Got K_inv", flush=True)
        print(f"   Getting GSD tolerance...", flush=True)
        h_tol, v_tol = get_tolerance(site_name)
        print(f"   GSD tolerance: h={h_tol:.3f}m, v={v_tol:.3f}m", flush=True)

        # 이미지 크기 가져오기 (site별로 다름)
        IMG_WIDTH, IMG_HEIGHT = get_image_size(site_name)
        print(f"   Image size (from site info): {IMG_WIDTH} × {IMG_HEIGHT}", flush=True)

        # 가로 분할 크기 (절반)
        COL_CHUNK = IMG_WIDTH // 2
        print(f"   Column chunk size: {COL_CHUNK}", flush=True)

    except Exception as e:
        print(f"   [ERROR] Failed to get camera params: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return np.zeros(len(coords), dtype=np.int32)

    # Vote 초기화
    votes = np.zeros(len(fg_coords), dtype=np.int32)

    # 이미지별 포인트 캐시 디렉터리
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    # Footprint 전체 범위 (모든 이미지 고려)
    print(f"   Computing footprint bounds...", flush=True)
    all_cams = np.array([cam['C'] for cam in cam_db['images'].values()])
    margin = float(os.environ.get("FP_MARGIN_M", "0"))
    fp_x_min = all_cams[:, 0].min() - margin
    fp_x_max = all_cams[:, 0].max() + margin
    fp_y_min = all_cams[:, 1].min() - margin
    fp_y_max = all_cams[:, 1].max() + margin

    # 전체 Footprint 내 points
    print(f"   Filtering footprint points...", flush=True)
    mask_fp = (
        (fg_coords[:, 0] >= fp_x_min) & (fg_coords[:, 0] <= fp_x_max) &
        (fg_coords[:, 1] >= fp_y_min) & (fg_coords[:, 1] <= fp_y_max)
    )
    fp_indices = np.where(mask_fp)[0]
    fp_coords = fg_coords[fp_indices]

    print(f"   Footprint points: {len(fp_coords):,}", flush=True)
    print(f"   Point Z range: [{fp_coords[:, 2].min():.2f}, {fp_coords[:, 2].max():.2f}]", flush=True)

    # GPU 준비 (필요 시 이미지별 subset만 올리므로 전체 GPU 업로드는 생략)
    fp_coords_gpu = None  # 지연 로딩

    # 이미지별 처리
    images = cam_db['images']
    total_rays = 0

    # 중간 저장 설정
    CHECKPOINT_INTERVAL = 50  # 50장마다 저장
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # 이미지 병렬 워커 결정: 기본 10, 환경변수 IMAGE_WORKERS로 덮어쓰기
    _worker_env = os.environ.get("IMAGE_WORKERS", "").strip()
    _cpu = os.cpu_count() or 1
    default_workers = C.PARALLEL_CONFIG.get("num_workers", 10)
    try:
        _worker_val = int(_worker_env) if _worker_env else default_workers
    except ValueError:
        _worker_val = default_workers
    num_workers = max(1, min(_worker_val, _cpu))

    print(f"   Images: {len(images)} (workers={num_workers})")
    votes_lock = threading.Lock()
    # GPU 자원 보호용 세마포어 (동시 GPU 커널/업로드 제한)
    gpu_slots = int(os.environ.get("IMAGE_GPU_SLOTS", "1") or "1")
    gpu_sem = threading.Semaphore(max(1, gpu_slots))

    def _process_one(idx: int, img_name: str) -> int:
        nonlocal total_rays
        cam = images[img_name]

        # 마스크 로드
        img_path = det_dir / img_name
        if not img_path.exists():
            return 0

        mask = img_mask_diff(str(img_path))
        if mask is None or not mask.any():
            return 0

        # 마스크 크기 확인 (site 정보와 일치해야 함)
        mask_h, mask_w = mask.shape
        local_IMG_WIDTH, local_IMG_HEIGHT = IMG_WIDTH, IMG_HEIGHT
        local_COL_CHUNK = COL_CHUNK
        if mask_w != IMG_WIDTH or mask_h != IMG_HEIGHT:
            print(f"   [WARNING] {img_name}: mask size {mask_w}×{mask_h} != expected {IMG_WIDTH}×{IMG_HEIGHT}")
            local_IMG_WIDTH, local_IMG_HEIGHT = mask_w, mask_h
            local_COL_CHUNK = local_IMG_WIDTH // 2

        # 카메라 파라미터
        C_cam = np.array(cam['C'], dtype=np.float64)
        R_wc = r_from_opk(cam['omega'], cam['phi'], cam['kappa'])
        R_c2w = R_wc.T

        # GPU 업로드는 세마포어로 보호해 동시점유 과다 방지
        with gpu_sem:
            C_cam_gpu = cp.asarray(C_cam, dtype=cp.float64)
            R_c2w_gpu = cp.asarray(R_c2w, dtype=cp.float64)
            K_inv_gpu = cp.asarray(K_inv, dtype=cp.float64)

        debug_count = 0
        band_none_count = 0
        no_points_count = 0
        local_total_rays = 0

        # Debug: 첫 이미지 정보 (thread 0만)
        if idx == 0:
            print(f"   [DEBUG] First image: {img_name}")
            print(f"   [DEBUG] Using image size: {local_IMG_WIDTH} × {local_IMG_HEIGHT}")
            print(f"   [DEBUG] Col chunk size: {local_COL_CHUNK}")
            print(f"   [DEBUG] Camera: X={C_cam[0]:.1f}, Y={C_cam[1]:.1f}, Z={C_cam[2]:.1f}")

        # 이미지별 포인트 subset 캐시 로드/생성
        cache_file = cache_dir / f"{img_name}_idx.npy"
        if cache_file.exists():
            img_fp_idx = np.load(cache_file)
        else:
            # 카메라 footprint bbox 기반 XY 필터
            x_min_fp, x_max_fp, y_min_fp, y_max_fp = compute_camera_footprint_bbox(C_cam, site_name, margin=C.FOOTPRINT_MARGIN)
            mask_img_xy = (
                (fp_coords[:, 0] >= x_min_fp) & (fp_coords[:, 0] <= x_max_fp) &
                (fp_coords[:, 1] >= y_min_fp) & (fp_coords[:, 1] <= y_max_fp)
            )
            # Z 필터는 완화 (카메라 아래쪽)
            Z_min = ground_Z - 20.0
            Z_max = C_cam[2] + 20.0
            mask_img = mask_img_xy & (fp_coords[:, 2] >= Z_min) & (fp_coords[:, 2] <= Z_max)
            img_fp_idx = np.where(mask_img)[0].astype(np.int32)
            np.save(cache_file, img_fp_idx)

        if len(img_fp_idx) == 0:
            return 0

        # 선택된 포인트만 GPU 전송
        with gpu_sem:
            pts_gpu = cp.asarray(fp_coords[img_fp_idx], dtype=cp.float64)

        # 마스크 밀도에 따라 sparse/row 모드 선택
        nonblack = np.count_nonzero(mask)
        density = nonblack / (mask.size + 1e-9)
        sparse_thresh = float(os.environ.get("SPARSE_THRESH", "0.10"))  # 기본 10% 이하이면 sparse

        if density <= sparse_thresh:
            # Sparse mode: 마스크 좌표만 가져와 한번에 레이 생성
            y_idx, x_idx = np.nonzero(mask)
            if len(x_idx) == 0:
                return 0

            pixels = np.stack([x_idx, y_idx, np.ones(len(x_idx))], axis=1).astype(np.float64)
            rays_cam = (K_inv @ pixels.T).T
            rays_world = (R_c2w @ rays_cam.T).T
            rays_world = rays_world / np.linalg.norm(rays_world, axis=1, keepdims=True)
            up_mask = rays_world[:, 2] > 0
            if np.any(up_mask):
                rays_world[up_mask] *= -1

            max_ray_batch = 50000
            for rs in range(0, len(rays_world), max_ray_batch):
                re = min(rs + max_ray_batch, len(rays_world))
                with gpu_sem:
                    rays_gpu = cp.asarray(rays_world[rs:re], dtype=cp.float64)
                    hits = process_rays_gpu(rays_gpu, C_cam_gpu, pts_gpu, h_tol, v_tol)
                hits_cpu = cp.asnumpy(hits)
                valid = hits_cpu >= 0
                if np.any(valid):
                    global_idx = fp_indices[img_fp_idx[hits_cpu[valid]]]
                    with votes_lock:
                        np.add.at(votes, global_idx, 1)
                local_total_rays += (re - rs)
        else:
            # Row별 처리 (stride 적용) - 진행률 표시 옵션
            row_stride = max(1, int(os.environ.get("ROW_STRIDE", "4")))  # 기본 4
            show_row_progress = os.environ.get("TQDM_ROWS", "0") != "0"  # 기본 OFF
            row_iter = range(0, local_IMG_HEIGHT, row_stride)
            if show_row_progress:
                row_iter = tqdm(row_iter, desc=f"      Rows {img_name}", leave=False)

            for row in row_iter:
                row_mask = mask[row, :]
                if not row_mask.any():
                    continue

                for col_start in range(0, local_IMG_WIDTH, local_COL_CHUNK):
                    col_end = min(col_start + local_COL_CHUNK, local_IMG_WIDTH)
                    chunk_mask = row_mask[col_start:col_end]
                    if not chunk_mask.any():
                        continue

                    x_idx = np.where(chunk_mask)[0] + col_start
                    y_idx = np.full(len(x_idx), row, dtype=np.int32)
                    if len(x_idx) == 0:
                        continue

                    mid_col = (col_start + col_end) // 2
                    do_debug = (debug_count < 3)
                    sample_cols = [col_start, mid_col, col_end-1]

                    ground_points = []
                    for col in sample_cols:
                        pixel = np.array([col, row, 1.0])
                        ray_cam = K_inv @ pixel
                        ray_world = R_c2w @ ray_cam
                        ray_world = ray_world / np.linalg.norm(ray_world)
                        if abs(ray_world[2]) > 0.001:
                            t = (ground_Z - C_cam[2]) / ray_world[2]
                            if t != 0 and abs(t) > 0.1:
                                ground_point = C_cam + t * ray_world
                                if abs(ground_point[2] - ground_Z) < 50.0:
                                    ground_points.append(ground_point[:2])

                    if len(ground_points) == 0:
                        band_none_count += 1
                        if do_debug:
                            print(f"      [DEBUG] Band None: row {row}, cols [{col_start}:{col_end}]")
                            debug_count += 1
                        continue

                    ground_points = np.array(ground_points)
                    margin = 1.0
                    x_min = ground_points[:, 0].min() - margin
                    x_max = ground_points[:, 0].max() + margin
                    y_min = ground_points[:, 1].min() - margin
                    y_max = ground_points[:, 1].max() + margin

                    Z_min = ground_Z - 20.0
                    Z_max = C_cam[2] + 20.0

                    mask_xy = (
                        (fp_coords[:, 0] >= x_min) & (fp_coords[:, 0] <= x_max) &
                        (fp_coords[:, 1] >= y_min) & (fp_coords[:, 1] <= y_max)
                    )
                    mask_band = mask_xy & (fp_coords[:, 2] >= Z_min) & (fp_coords[:, 2] <= Z_max)

                    band_idx = np.where(mask_band)[0]
                    if len(band_idx) == 0:
                        no_points_count += 1
                        if do_debug:
                            num_xy = mask_xy.sum()
                            print(f"      [DEBUG] No points in band: row {row}, cols [{col_start}:{col_end}] (XY band has {num_xy} points, but 0 after Z filter)")
                            debug_count += 1
                        continue

                    band_coords_gpu = cp.asarray(fp_coords[band_idx], dtype=cp.float64)

                    if do_debug:
                        print(f"      [DEBUG] Row {row}, cols [{col_start}:{col_end}]: {len(x_idx)} rays, {len(band_idx)} points in band")
                        print(f"      [DEBUG] Band: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]")
                        print(f"      [DEBUG] Z filter: [{Z_min:.1f}, {Z_max:.1f}]")
                        debug_count += 1

                    pixels = np.stack([x_idx, y_idx, np.ones(len(x_idx))], axis=1)
                    rays_cam = (K_inv @ pixels.T).T
                    rays_world = (R_c2w @ rays_cam.T).T
                    rays_world = rays_world / np.linalg.norm(rays_world, axis=1, keepdims=True)

                    up_mask = rays_world[:, 2] > 0
                    if np.any(up_mask):
                        rays_world[up_mask] *= -1

                    with gpu_sem:
                        rays_gpu = cp.asarray(rays_world, dtype=cp.float64)
                        hits = process_rays_gpu(rays_gpu, C_cam_gpu, band_coords_gpu, h_tol, v_tol)

                    hits_cpu = cp.asnumpy(hits)
                    valid = hits_cpu >= 0
                    num_hits = valid.sum()
                    if num_hits > 0:
                        global_idx = fp_indices[band_idx[hits_cpu[valid]]]
                        with votes_lock:
                            np.add.at(votes, global_idx, 1)

                    local_total_rays += len(x_idx)

        if band_none_count > 0 or no_points_count > 0:
            print(f"      [SUMMARY] {img_name}: Band None={band_none_count}, No points={no_points_count}, Rays={local_total_rays}")

        if (idx + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = checkpoint_dir / f"votes_checkpoint_{idx+1:04d}.npy"
            with votes_lock:
                np.save(checkpoint_path, votes)
                print(f"\n   [CHECKPOINT] Saved at image {idx+1}: {checkpoint_path}")
                print(f"   Current votes: {votes.sum():,} (max={votes.max()})\n", flush=True)

        return local_total_rays

    # 실행 (단일/멀티스레드)
    image_list = list(images.keys())
    if num_workers == 1:
        for idx, img_name in enumerate(tqdm(image_list, desc="   Images")):
            tr = _process_one(idx, img_name)
            total_rays += tr
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(_process_one, idx, name): name for idx, name in enumerate(image_list)}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="   Images (parallel)"):
                try:
                    tr = fut.result()
                    total_rays += tr
                except Exception as e:
                    print(f"[ERROR] {futures[fut]}: {e}")

    # GPU 정리
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()

    print(f"   Total rays: {total_rays:,}")
    print(f"   Votes: {votes.sum():,}")

    return votes


# ==========================================
# 3. BACKWARD PASS
# ==========================================

def backward_projection(site_name: str, coords: np.ndarray, colors: np.ndarray,
                       forward_votes: np.ndarray, threshold: int = 7) -> np.ndarray:
    """Backward pass - Forward vote가 높은 points만 역투영"""
    import cupy as cp

    print(f"\n[BACKWARD] {site_name}")

    # Threshold 이상인 points만
    valid_mask = forward_votes >= threshold
    valid_indices = np.where(valid_mask)[0]
    valid_coords = coords[valid_indices]

    print(f"   Vote >= {threshold}: {len(valid_coords):,} points")

    if len(valid_coords) == 0:
        return np.zeros(len(coords), dtype=np.int32)

    # 설정
    det_dir = C.DETECTION_DIRS.get(site_name)
    cam_db = load_camera_db(site_name)
    K = get_camera_matrix(site_name)

    # Backward votes
    backward_votes = np.zeros(len(coords), dtype=np.int32)

    # GPU 준비
    valid_coords_gpu = cp.asarray(valid_coords, dtype=cp.float64)
    K_gpu = cp.asarray(K, dtype=cp.float64)

    # 이미지별 처리
    images = cam_db['images']

    for img_name in tqdm(list(images.keys())[:100], desc="   Images"):
        cam = images[img_name]

        # 마스크 로드
        img_path = det_dir / img_name
        if not img_path.exists():
            continue

        mask = img_mask_diff(str(img_path))
        if mask is None or not mask.any():
            continue

        H, W = mask.shape

        # 카메라 파라미터
        C_cam = np.array(cam['C'], dtype=np.float64)
        R_wc = r_from_opk(cam['omega'], cam['phi'], cam['kappa'])

        C_cam_gpu = cp.asarray(C_cam, dtype=cp.float64)
        R_wc_gpu = cp.asarray(R_wc, dtype=cp.float64)

        # Points → Image 투영
        P_world = valid_coords_gpu - C_cam_gpu
        P_cam = (R_wc_gpu @ P_world.T).T

        # 카메라 앞만
        front_mask = P_cam[:, 2] > 0.1
        if not front_mask.any():
            continue

        P_cam_front = P_cam[front_mask]
        front_idx = cp.where(front_mask)[0]

        # Image plane
        P_img = (K_gpu @ P_cam_front.T).T
        P_img[:, 0] /= P_img[:, 2]
        P_img[:, 1] /= P_img[:, 2]

        # 이미지 범위 내
        x = P_img[:, 0]
        y = P_img[:, 1]

        in_img = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if not in_img.any():
            continue

        x_valid = x[in_img].astype(cp.int32)
        y_valid = y[in_img].astype(cp.int32)
        img_idx = front_idx[in_img]

        # CPU로 전송
        x_cpu = cp.asnumpy(x_valid)
        y_cpu = cp.asnumpy(y_valid)
        img_idx_cpu = cp.asnumpy(img_idx)

        # Mask 체크
        for i in range(len(x_cpu)):
            if mask[y_cpu[i], x_cpu[i]]:
                global_idx = valid_indices[img_idx_cpu[i]]
                backward_votes[global_idx] += 1

    # GPU 정리
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()

    print(f"   Backward votes: {backward_votes.sum():,}")

    return backward_votes


# ==========================================
# 4. LAS 파일 저장
# ==========================================

def save_las(coords: np.ndarray, colors: np.ndarray, votes: np.ndarray,
            threshold: int, output_path: Path) -> None:
    """Vote threshold 이상인 points를 LAS로 저장"""
    mask = votes >= threshold
    if not mask.any():
        print(f"   [SKIP] vote >= {threshold}: 0 points")
        return

    filtered_coords = coords[mask]
    filtered_colors = colors[mask]
    filtered_votes = votes[mask]

    # LAS 생성
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(filtered_coords, axis=0)
    header.scales = [0.001, 0.001, 0.001]

    las = laspy.LasData(header)
    las.x = filtered_coords[:, 0]
    las.y = filtered_coords[:, 1]
    las.z = filtered_coords[:, 2]

    # RGB (16-bit)
    las.red = (filtered_colors[:, 0] * 257).astype(np.uint16)
    las.green = (filtered_colors[:, 1] * 257).astype(np.uint16)
    las.blue = (filtered_colors[:, 2] * 257).astype(np.uint16)

    # Vote count
    las.point_source_id = np.clip(filtered_votes, 0, 65535).astype(np.uint16)

    las.write(str(output_path))

    print(f"   [OK] vote >= {threshold}: {len(filtered_coords):,} points → {output_path.name}")


# ==========================================
# 5. 메인 파이프라인
# ==========================================

def process_site(site_name: str) -> None:
    """사이트 1개 전체 처리 (Forward + Backward + LAS)"""
    print(f"\n{'='*70}")
    print(f"Processing: {site_name}")
    print(f"{'='*70}")

    # 경로
    las_dir = C.PART2_DIR / site_name
    output_dir = C.PART3_DIR / site_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 포인트 클라우드 로드
    try:
        coords, colors = load_point_cloud(las_dir)
    except FileNotFoundError:
        print(f"   [ERROR] 포인트 클라우드 없음")
        return

    # 1. Forward pass
    start = time.time()
    forward_votes = forward_scanline(site_name, coords, colors)
    forward_time = time.time() - start

    # Forward 저장
    forward_path = output_dir / "forward_votes.npy"
    np.save(forward_path, forward_votes)
    print(f"   Forward 시간: {forward_time:.1f}초")

    # 2. Backward pass
    start = time.time()
    backward_votes = backward_projection(site_name, coords, colors, forward_votes, threshold=7)
    backward_time = time.time() - start

    # Backward 저장
    backward_path = output_dir / "backward_votes.npy"
    np.save(backward_path, backward_votes)
    print(f"   Backward 시간: {backward_time:.1f}초")

    # 3. 최종 votes (Forward + Backward)
    final_votes = forward_votes + backward_votes
    final_path = output_dir / "final_votes.npy"
    np.save(final_path, final_votes)

    # 4. LAS 저장 (threshold별)
    print(f"\n[LAS 저장]")
    for threshold in [7, 15, 30]:
        las_path = output_dir / f"vote_{threshold}.las"
        save_las(coords, colors, final_votes, threshold, las_path)


def process_all_sites() -> None:
    """Site A/B/C 전체 처리 (Zenmuse/P4R/Joint)"""
    sites = [
        # Site A
        "Zenmuse_AI_Site_A",
        "P4R_Site_A_Solid",
        "P4R_Zenmuse_Joint_AI_Site_A",
        # Zenmuse B/C
        "Zenmuse_AI_Site_B",
        "Zenmuse_AI_Site_C",
        # P4R B/C
        "P4R_Site_B_Solid_Merge_V2",
        "P4R_Site_C_Solid_Merge_V2",
        # Joint B/C
        "P4R_Zenmuse_Joint_AI_Site_B",
        "P4R_Zenmuse_Joint_AI_Site_C",
    ]

    print(f"\n{'='*70}")
    print(f"전체 파이프라인: {len(sites)}개 사이트")
    print(f"{'='*70}")

    total_start = time.time()

    for site in sites:
        try:
            process_site(site)
        except Exception as e:
            print(f"\n[ERROR] {site} 실패: {e}")
            continue

    total_time = time.time() - total_start

    print(f"\n{'='*70}")
    print(f"전체 완료: {total_time/60:.1f}분")
    print(f"{'='*70}")


if __name__ == "__main__":
    # GPU 확인
    if not C.USE_GPU:
        print("[ERROR] GPU 비활성화 상태")
        exit(1)

    import sys

    if len(sys.argv) > 1:
        # 특정 사이트만
        site = sys.argv[1]
        process_site(site)
    else:
        # 전체 사이트
        process_all_sites()
