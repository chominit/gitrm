# -*- coding: utf-8 -*-
"""
part3_forward_scanline.py
Scanline 방식 Forward Ray Casting (Row-by-Row 처리)
메모리 효율적이고 진행상황 저장 가능
"""

from pathlib import Path
import numpy as np
from typing import List, Tuple
import time
import constants as C
from camera_io import load_camera_db
from camera_calibration import get_camera_matrix, compute_camera_footprint_bbox
from img_mask import img_mask_diff
from gsd_parser import get_tolerance
from geometry import r_from_opk
from tqdm import tqdm

def load_point_cloud(las_dir: Path, use_sampling: bool, sample_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """포인트 클라우드 로드"""
    coords_path = las_dir / "coords_float64.npy"
    colors_path = las_dir / "colors_uint8.npy"

    if coords_path.exists() and colors_path.exists():
        print(f"   [OK] NPY 캐시 로드 중...")
        coords = np.load(coords_path)
        colors = np.load(colors_path)
        print(f"   원본 포인트: {len(coords):,}")
        return coords, colors
    else:
        raise FileNotFoundError(f"NPY 캐시 없음: {las_dir}")


def compute_row_ground_band(row_idx: int, img_height: int, img_width: int,
                             C_cam: np.ndarray, R_c2w: np.ndarray, K: np.ndarray,
                             ground_Z: float) -> Tuple[float, float, float, float]:
    """
    Row의 rays가 지면과 만나는 영역(band) 계산

    Args:
        row_idx: 현재 row 번호
        img_height: 이미지 높이
        img_width: 이미지 너비
        C_cam: 카메라 위치
        R_c2w: 카메라 회전 행렬
        K: 카메라 내부 파라미터
        ground_Z: 지면 Z 좌표

    Returns:
        (x_min, x_max, y_min, y_max): Ground band 범위
    """
    import cupy as cp

    K_inv = np.linalg.inv(K)

    # Row의 시작/끝 픽셀 (양 끝 + 중간)
    sample_cols = [0, img_width // 4, img_width // 2, 3 * img_width // 4, img_width - 1]

    ground_points = []

    for col in sample_cols:
        # Pixel → Ray
        pixel = np.array([col, row_idx, 1.0])
        ray_cam = K_inv @ pixel
        ray_world = R_c2w @ ray_cam
        # Ray 방향: z>0이면(위쪽) 뒤집어서 항상 지면(음의 z) 쪽을 보도록 보정
        if ray_world[2] > 0:
            ray_world = -ray_world
        ray_world = ray_world / np.linalg.norm(ray_world)

        # Ray와 ground plane 교점
        # C_cam + t * ray_world = (x, y, ground_Z)
        # C_cam[2] + t * ray_world[2] = ground_Z
        # t = (ground_Z - C_cam[2]) / ray_world[2]

        if abs(ray_world[2]) > 0.001:  # 거의 수평이 아닌 경우
            t = (ground_Z - C_cam[2]) / ray_world[2]
            if t > 0:  # 카메라 앞
                ground_point = C_cam + t * ray_world
                ground_points.append(ground_point[:2])  # x, y만

    if len(ground_points) == 0:
        return None

    ground_points = np.array(ground_points)

    # Band 범위 (약간 여유 추가)
    margin = 2.0  # 2m 여유
    x_min = ground_points[:, 0].min() - margin
    x_max = ground_points[:, 0].max() + margin
    y_min = ground_points[:, 1].min() - margin
    y_max = ground_points[:, 1].max() + margin

    return (x_min, x_max, y_min, y_max)


def process_rays_batch_gpu(rays: 'cp.ndarray', origin: 'cp.ndarray',
                           coords: 'cp.ndarray', h_tol: float, v_tol: float) -> 'cp.ndarray':
    """
    GPU에서 rays와 points의 distance 계산 및 hit 판정

    Returns:
        hit_indices: (N,) hit된 point의 index (-1이면 miss)
    """
    import cupy as cp

    N = len(rays)
    M = len(coords)

    if M == 0:
        return cp.full(N, -1, dtype=cp.int32)

    # Point batching (메모리 관리)
    MAX_POINTS_PER_BATCH = 200_000  # 기본값 (더 큰 배치로 GPU 활용 극대화)

    # GPU 메모리 상태에 따라 동적으로 MAX_POINTS_PER_BATCH 조정
    try:
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        max_ratio = getattr(C, "MAX_GPU_MEMORY_USAGE_RATIO", 0.9)
        max_allowed_bytes = int(total_bytes * max_ratio)
        current_used = total_bytes - free_bytes
        available_for_us = max_allowed_bytes - current_used

        if available_for_us > 0:
            # float64 기준 per (ray, point) 조합 대략 메모리(보수적으로 64 bytes 가정)
            BYTES_PER_PAIR = 64
            # 한 배치에서 사용하는 메모리 ≈ N * MAX_POINTS_PER_BATCH * BYTES_PER_PAIR
            max_points = max(1000, int(available_for_us / max(1, N * BYTES_PER_PAIR)))
            MAX_POINTS_PER_BATCH = min(MAX_POINTS_PER_BATCH, max_points)
    except Exception:
        # memGetInfo 사용 불가 시 기본값 유지
        pass

    best_depths = cp.full(N, cp.inf, dtype=cp.float64)
    best_indices = cp.full(N, -1, dtype=cp.int32)

    num_batches = (M + MAX_POINTS_PER_BATCH - 1) // MAX_POINTS_PER_BATCH

    for batch_idx in range(num_batches):
        start = batch_idx * MAX_POINTS_PER_BATCH
        end = min(start + MAX_POINTS_PER_BATCH, M)
        coords_batch = coords[start:end]

        # Distance 계산
        P = coords_batch - origin  # (M_batch, 3)
        proj_dist = cp.dot(rays, P.T)  # (N, M_batch)

        # 카메라 뒤 제외
        valid_mask = proj_dist > 0.1

        # 투영점
        proj_points = origin + rays[:, None, :] * proj_dist[:, :, None]

        # Residuals
        residuals = coords_batch[None, :, :] - proj_points

        # 수평/수직 거리
        h_dist = cp.linalg.norm(residuals[:, :, :2], axis=2)
        v_dist = cp.abs(residuals[:, :, 2])

        # GSD 허용오차 체크
        candidates = valid_mask & (h_dist <= h_tol) & (v_dist <= v_tol)

        # 각 ray별 최소 거리 업데이트
        for ray_idx in range(N):
            valid = cp.where(candidates[ray_idx])[0]
            if len(valid) > 0:
                depths = proj_dist[ray_idx, valid]
                min_idx = cp.argmin(depths)
                depth = depths[min_idx]

                if depth < best_depths[ray_idx]:
                    best_depths[ray_idx] = depth
                    best_indices[ray_idx] = start + valid[min_idx]

    return best_indices


def process_image_scanline(img_name: str, cam_info: dict, coords: np.ndarray,
                           det_dir: Path, site_name: str, ground_Z: float,
                           K: np.ndarray, h_tol: float, v_tol: float,
                           votes: np.ndarray) -> int:
    """
    이미지 1장을 Scanline 방식으로 처리

    Returns:
        processed_rays: 처리된 ray 개수
    """
    import cupy as cp

    # 1. 이미지 마스크 로드
    img_path = det_dir / img_name
    if not img_path.exists():
        return 0

    mask = img_mask_diff(str(img_path))
    if mask is None or not mask.any():
        return 0

    H, W = mask.shape

    # 2. 카메라 파라미터
    C_cam = np.array(cam_info['C'], dtype=np.float64)
    omega, phi, kappa = cam_info['omega'], cam_info['phi'], cam_info['kappa']
    R_wc = r_from_opk(omega, phi, kappa)
    R_c2w = R_wc.T

    K_inv = np.linalg.inv(K)

    # 3. Footprint 계산 (margin 최소)
    try:
        # FOOTPRINT_MARGIN = 0.0 사용
        bbox = compute_camera_footprint_bbox(C_cam, site_name, margin=0.0)
    except Exception as e:
        print(f"   [SKIP] {img_name}: Footprint 계산 실패 - {e}")
        return 0

    x_min_fp, x_max_fp, y_min_fp, y_max_fp = bbox

    # Footprint 내 포인트 (XY만, Z 필터 나중에)
    mask_footprint = (
        (coords[:, 0] >= x_min_fp) & (coords[:, 0] <= x_max_fp) &
        (coords[:, 1] >= y_min_fp) & (coords[:, 1] <= y_max_fp)
    )
    footprint_indices = np.where(mask_footprint)[0]
    footprint_coords = coords[footprint_indices]

    if len(footprint_coords) == 0:
        return 0

    # 4. Row-by-Row 처리
    total_rays = 0

    # Row별로 diff 픽셀이 있는지 확인
    rows_with_pixels = []
    for row in range(H):
        if mask[row, :].any():
            rows_with_pixels.append(row)

    if len(rows_with_pixels) == 0:
        return 0

    print(f"   Rows with diff pixels: {len(rows_with_pixels)}/{H}")

    # GPU 전송 (한번만)
    C_cam_gpu = cp.asarray(C_cam, dtype=cp.float64)
    R_c2w_gpu = cp.asarray(R_c2w, dtype=cp.float64)
    K_inv_gpu = cp.asarray(K_inv, dtype=cp.float64)

    for row in tqdm(rows_with_pixels, desc=f"   Processing {img_name}", leave=False):
        # Row에서 diff 픽셀만 추출
        row_mask = mask[row, :]
        if not row_mask.any():
            continue

        cols = np.where(row_mask)[0]
        num_rays = len(cols)

        if num_rays == 0:
            continue

        # 5. Row의 ground band 계산
        band = compute_row_ground_band(row, H, W, C_cam, R_c2w, K, ground_Z)

        if band is None:
            continue

        x_min_band, x_max_band, y_min_band, y_max_band = band

        # 6. Band 내 포인트 필터링 + Z 필터
        Z_min = ground_Z - 10.0
        Z_max = C_cam[2] + 10.0

        mask_band = (
            (footprint_coords[:, 0] >= x_min_band) & (footprint_coords[:, 0] <= x_max_band) &
            (footprint_coords[:, 1] >= y_min_band) & (footprint_coords[:, 1] <= y_max_band) &
            (footprint_coords[:, 2] >= Z_min) & (footprint_coords[:, 2] <= Z_max)
        )

        band_indices = footprint_indices[mask_band]
        band_coords = footprint_coords[mask_band]

        if len(band_coords) == 0:
            continue

        # 7. Rays 생성
        y_indices = np.full(num_rays, row, dtype=np.int32)
        x_indices = cols

        pixels_hom = np.stack([x_indices, y_indices, np.ones(num_rays)], axis=1).astype(np.float64)

        rays_cam = (K_inv @ pixels_hom.T).T
        rays_world = (R_c2w @ rays_cam.T).T
        rays_world = rays_world / np.linalg.norm(rays_world, axis=1, keepdims=True)

        # Ray z>0(위쪽)을 향하는 경우 뒤집어서 지면(음의 z) 쪽을 보도록 보정
        up_mask = rays_world[:, 2] > 0
        if np.any(up_mask):
            rays_world[up_mask] *= -1

        # GPU 전송
        rays_gpu = cp.asarray(rays_world, dtype=cp.float64)
        band_coords_gpu = cp.asarray(band_coords, dtype=cp.float64)

        # 8. Ray casting
        hit_indices = process_rays_batch_gpu(rays_gpu, C_cam_gpu, band_coords_gpu, h_tol, v_tol)

        # 9. Vote 누적
        hit_indices_cpu = cp.asnumpy(hit_indices)
        valid_hits = hit_indices_cpu[hit_indices_cpu >= 0]

        if len(valid_hits) > 0:
            global_indices = band_indices[valid_hits]
            votes[global_indices] += 1

        total_rays += num_rays

        # 메모리 정리
        del rays_gpu, band_coords_gpu, hit_indices

    # GPU 메모리 정리
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()

    return total_rays


def run(site_name: str = "Zenmuse_AI_Site_B") -> None:
    """Scanline 방식 Forward 실행"""
    print(f"\n{'='*60}")
    print(f"Part3 Forward Scanline (Row-by-Row)")
    print(f"{'='*60}")
    print(f"사이트: {site_name}")
    print(f"GPU: {'활성화' if C.USE_GPU else '비활성화'}")

    # 경로 설정
    las_dir = C.PART2_DIR / site_name
    det_dir = C.DETECTION_DIRS.get(site_name)
    output_dir = C.PART3_DIR / site_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 포인트 클라우드 로드
    xyz, rgb = load_point_cloud(las_dir, False, 0)

    # 카메라 DB 로드
    cam_db = load_camera_db(site_name)

    # Ground Z 계산
    Z_values = [cam_info['C'][2] for cam_info in cam_db['images'].values()]
    Z_avg = float(sum(Z_values) / len(Z_values))
    H_site = C.FLIGHT_ALT_BY_SITE.get(site_name, 80.0)
    ground_Z = Z_avg - H_site

    print(f"\n[Ground Z 계산]")
    print(f"   카메라 평균 Z: {Z_avg:.2f} m")
    print(f"   H_site: {H_site:.1f} m")
    print(f"   Ground Z: {ground_Z:.2f} m")

    # 카메라 매트릭스
    K = get_camera_matrix(site_name)

    # GSD 허용오차
    h_tol, v_tol = get_tolerance(site_name)

    print(f"\n[Processing Parameters]")
    print(f"   GSD 허용오차: 수평={h_tol}m, 수직={v_tol}m")
    print(f"   Footprint margin: 0.0 (GSD 기반)")
    print(f"   Processing: Scanline (row-by-row)")

    # Vote 배열 초기화
    N = xyz.shape[0]
    votes = np.zeros(N, dtype=np.int32)

    # 이미지 처리
    images = cam_db['images']
    image_names = list(images.keys())

    print(f"\n[Processing {len(image_names)} images]")

    start_time = time.time()
    total_rays = 0
    processed = 0

    for img_name in tqdm(image_names, desc="Images"):
        cam_info = images[img_name]
        rays = process_image_scanline(
            img_name, cam_info, xyz, det_dir, site_name,
            ground_Z, K, h_tol, v_tol, votes
        )

        if rays > 0:
            total_rays += rays
            processed += 1

    elapsed = time.time() - start_time

    # 결과 저장
    votes_path = output_dir / "forward_votes_scanline.npy"
    np.save(votes_path, votes)

    print(f"\n{'='*60}")
    print(f"[완료]")
    print(f"{'='*60}")
    print(f"처리 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    print(f"처리된 이미지: {processed}/{len(image_names)}")
    print(f"총 rays: {total_rays:,}")
    print(f"총 투표: {votes.sum():,}")
    print(f"평균 투표: {votes.mean():.2f}")
    print(f"최대 투표: {votes.max()}")
    print(f"\n저장: {votes_path}")

    # Vote threshold별 통계
    print(f"\n[Vote Statistics]")
    for threshold in C.VOTE_THRESHOLDS:
        count = (votes >= threshold).sum()
        print(f"   vote >= {threshold}: {count:,} points ({100*count/N:.2f}%)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        site = sys.argv[1]
    else:
        site = "Zenmuse_AI_Site_B"

    run(site)
