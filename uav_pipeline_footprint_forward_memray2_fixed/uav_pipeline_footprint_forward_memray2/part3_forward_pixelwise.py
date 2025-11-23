# -*- coding: utf-8 -*-
"""
part3_forward_pixelwise.py
Footprint-based Forward Pixel-wise Z-buffer (MD 문서 기반 구현)
작성일: 2025-11-21
"""

from pathlib import Path
import numpy as np
from typing import List, Tuple
import time
import constants as C
from camera_io import load_camera_db
from camera_calibration import get_camera_matrix, compute_camera_footprint_bbox
from img_mask import img_mask, img_mask_diff  # 차분 이미지용 마스크 추가
from gsd_parser import get_tolerance
from geometry import r_from_opk, points_in_polygon

def load_point_cloud(las_dir: Path, use_sampling: bool, sample_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    포인트 클라우드 로드 (NPY 캐시 우선)

    Returns:
        coords: (N, 3) XYZ
        colors: (N, 3) RGB
    """
    coords_path = las_dir / "coords_float64.npy"
    colors_path = las_dir / "colors_uint8.npy"

    if coords_path.exists() and colors_path.exists():
        print(f"   [OK] NPY 캐시 로드 중...")
        coords = np.load(coords_path)
        colors = np.load(colors_path)

        print(f"   원본 포인트: {len(coords):,}")

        if use_sampling and sample_size > 0 and len(coords) > sample_size:
            indices = np.random.choice(len(coords), sample_size, replace=False)
            coords = coords[indices]
            colors = colors[indices]
            print(f"   샘플링 후: {len(coords):,}")

        return coords, colors

    raise FileNotFoundError(f"NPY 캐시 없음: {las_dir}")

def forward_vote_pixelwise_gpu(
    coords: np.ndarray,
    cam_db: dict,
    det_dir: Path,
    site_name: str,
    ground_Z: float,
    k_max: int = 1
) -> np.ndarray:
    """
    GPU Pixel-wise Forward 투표 (Footprint 기반)

    MD 문서 알고리즘:
    1. 각 이미지마다 footprint 계산
    2. Footprint 내 포인트만 선택
    3. Ray-point distance 계산
    4. 가장 가까운 K개 선택
    5. 투표

    Args:
        coords: (N, 3) 포인트 좌표
        cam_db: 카메라 DB
        det_dir: 검출 이미지 디렉터리
        site_name: 사이트 이름
        ground_Z: 지면 Z 좌표
        k_max: 최대 hit 수

    Returns:
        votes: (N,) 투표 수
    """
    import cupy as cp
    from tqdm import tqdm

    N = coords.shape[0]
    votes = np.zeros(N, dtype=np.int32)

    # GPU 전송
    coords_gpu = cp.asarray(coords, dtype=cp.float32)

    # 카메라 매트릭스
    K = get_camera_matrix(site_name)
    K_gpu = cp.asarray(K, dtype=cp.float32)
    K_inv_gpu = cp.linalg.inv(K_gpu)

    # GSD 허용오차
    h_tol, v_tol = get_tolerance(site_name)

    images = cam_db['images']
    image_names = list(images.keys())

    print(f"\n[GPU] Pixel-wise Forward 시작 (K_max={k_max})...")
    print(f"   GSD 허용오차: 수평={h_tol}m, 수직={v_tol}m")
    print(f"   Footprint-based filtering 사용 (전역 Grid 제거)")
    print(f"   포인트: {N:,}")
    print(f"   모드: {'Single-Hit' if k_max == 1 else f'Top-{k_max} Hit'}")
    print(f"   {len(image_names)}개 이미지 처리 중...")

    total_skipped = 0
    total_processed = 0

    for img_idx, img_name in enumerate(tqdm(image_names, desc="   [GPU] Footprint-based", unit="img")):
        cam = images[img_name]

        # 1. Camera footprint 계산
        C_cam_np = np.array(cam['C'], dtype=np.float64)
        omega, phi, kappa = cam['omega'], cam['phi'], cam['kappa']
        R_wc_np = r_from_opk(omega, phi, kappa)
        R_c2w_np = R_wc_np.T

        try:
            bbox = compute_camera_footprint_bbox(C_cam_np, site_name, margin=C.FOOTPRINT_MARGIN)
        except Exception as e:
            print(f"\n[WARNING] Footprint 계산 실패 ({img_name}): {e}")
            total_skipped += 1
            continue

        # 2. Footprint 내 포인트 필터링
        x_min, x_max, y_min, y_max = bbox

        # Footprint 크기 디버깅 (첫 5개 이미지만)
        if img_idx < 5:
            fp_width = x_max - x_min
            fp_height = y_max - y_min
            fp_area = fp_width * fp_height
            print(f"\n[DEBUG] {img_name}:")
            print(f"   Camera: ({C_cam_np[0]:.1f}, {C_cam_np[1]:.1f}, {C_cam_np[2]:.1f})")
            print(f"   Footprint: {fp_width:.1f}m × {fp_height:.1f}m = {fp_area:.0f}m²")

        # 2-1. XY Footprint 필터
        mask_footprint = (
            (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) &
            (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max)
        )

        # 2-2. Z 필터 추가 (카메라가 볼 수 있는 높이 범위만)
        # Ground ~ Camera 사이 + 여유 범위
        Z_min = ground_Z - 10.0  # 지면 아래 10m
        Z_max = C_cam_np[2] + 10.0  # 카메라 위 10m
        mask_z = (coords[:, 2] >= Z_min) & (coords[:, 2] <= Z_max)

        # XY + Z 조합
        mask_combined = mask_footprint & mask_z

        subset_indices = np.where(mask_combined)[0]
        subset_coords = coords[subset_indices]

        if img_idx < 5:  # 첫 5개 이미지만 출력
            footprint_count = np.sum(mask_footprint)
            filtered_count = len(subset_coords)
            reduction = 100 * (1 - filtered_count / footprint_count) if footprint_count > 0 else 0
            print(f"   Z-filter: {footprint_count:,} → {filtered_count:,} points ({reduction:.1f}% reduction)")

        if len(subset_coords) == 0:
            total_skipped += 1
            continue

        # MD 문서: 10M 하드 리미트 제거, 메모리 예측으로만 판단
        # (디버깅용 경고만 출력)
        if len(subset_coords) > 20_000_000:
            print(f"\n[WARNING] {img_name}: Footprint 포인트 {len(subset_coords):,}개 (대량)")

        # 이전 하드 리미트 제거됨 - 메모리 예측으로 처리 여부 결정

        # 3. 이미지 마스크 로드 (차분 이미지용)
        img_path = det_dir / img_name
        if not img_path.exists():
            total_skipped += 1
            continue

        mask = img_mask_diff(str(img_path))  # 차분 이미지: 비검정 픽셀 감지
        if mask is None or not mask.any():
            total_skipped += 1
            continue

        mask_gpu = cp.asarray(mask, dtype=cp.bool_)

        # 마스크된 픽셀 좌표
        y_gpu, x_gpu = cp.where(mask_gpu)
        if len(x_gpu) == 0:
            total_skipped += 1
            continue

        num_pixels = len(x_gpu)

        # 4. 메모리 체크 제거 (배치 처리가 이미 메모리 관리함)
        # MAX_POINTS_PER_BATCH=2M으로 배치 처리되므로 실제 메모리 사용량은 ~1GB 이하

        # 5. Ray 생성 (GPU)
        C_cam_gpu = cp.asarray(C_cam_np, dtype=cp.float32)
        R_c2w_gpu = cp.asarray(R_c2w_np, dtype=cp.float32)

        ones = cp.ones(num_pixels, dtype=cp.float32)
        pixels_hom = cp.stack([x_gpu, y_gpu, ones], axis=1)
        rays_cam = (K_inv_gpu @ pixels_hom.T).T
        rays_world = (R_c2w_gpu @ rays_cam.T).T
        rays_world /= cp.linalg.norm(rays_world, axis=1, keepdims=True)

        # Ray이 위쪽(z>0)을 향하면 뒤집어서 항상 지면(음의 z)을 보도록 보정
        up_mask = rays_world[:, 2] > 0
        if cp.any(up_mask):
            rays_world[up_mask] *= -1

        # 6. Subset을 GPU로 전송
        subset_coords_gpu = cp.asarray(subset_coords, dtype=cp.float32)

        # 7. Ray도 배치 처리 (전용 VRAM 최대 활용, 공유 메모리 회피)
        MAX_RAYS_PER_BATCH = 35_000  # 3.5만 ray (35K×10K×56B ≈ 19.6GB, 안전)
        num_ray_batches = (num_pixels + MAX_RAYS_PER_BATCH - 1) // MAX_RAYS_PER_BATCH

        if img_idx < 5:  # 첫 5개 이미지만 출력
            print(f"   Ray batches: {num_ray_batches} × {MAX_RAYS_PER_BATCH:,} rays")

        all_hits = [cp.array([], dtype=cp.int32)] * num_pixels

        for ray_batch_idx in range(num_ray_batches):
            ray_start = ray_batch_idx * MAX_RAYS_PER_BATCH
            ray_end = min(ray_start + MAX_RAYS_PER_BATCH, num_pixels)

            rays_batch = rays_world[ray_start:ray_end]

            # Ray-point distance 계산 (배치 처리)
            hits = process_rays_brute_force_gpu(
                rays_batch, C_cam_gpu, subset_coords_gpu,
                h_tol, v_tol, k_max
            )

            # 결과 저장
            all_hits[ray_start:ray_end] = hits

        # 8. 투표 (global index로 변환)
        for ray_hits in all_hits:
            if len(ray_hits) > 0:
                global_indices = subset_indices[cp.asnumpy(ray_hits)]
                votes[global_indices] += 1

        total_processed += 1

        # 메모리 정리
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()

    print(f"\n[OK] Forward 완료!")
    print(f"   처리됨: {total_processed}장")
    print(f"   스킵됨: {total_skipped}장")
    print(f"   총 투표: {votes.sum():,}")
    print(f"   평균 투표: {votes.mean():.2f}")
    print(f"   최대 투표: {votes.max()}")

    return votes

def process_rays_brute_force_gpu(
    rays: "cp.ndarray",
    origin: "cp.ndarray",
    coords: "cp.ndarray",
    h_tol: float,
    v_tol: float,
    k_max: int
) -> List["cp.ndarray"]:
    """
    Ray-Point distance brute-force 계산 (GPU 배치 처리)

    Args:
        rays: (N, 3) 정규화된 ray 방향
        origin: (3,) 카메라 위치
        coords: (M, 3) 포인트 좌표 (footprint subset)
        h_tol, v_tol: GSD 허용오차
        k_max: 최대 선택 개수

    Returns:
        hits: List of arrays, 각 ray별 hit point indices
    """
    import cupy as cp

    N = len(rays)
    M = len(coords)

    if M == 0:
        return [cp.array([], dtype=cp.int32)] * N

    # 동적 배치 크기 (GPU 메모리 안전 계산)
    # proj_dist(N,M), proj_points(N,M,3), residuals(N,M,3) 모두 고려
    # float32 기준 필요 메모리: N × M_batch × 4 × (1 + 3 + 3) = N × M_batch × 28 bytes
    BYTES_PER_ELEMENT = 28  # proj_dist + proj_points + residuals

    # GPU 메모리 상태 기반 SAFE_GPU_BYTES 계산
    try:
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        max_ratio = getattr(C, "MAX_GPU_MEMORY_USAGE_RATIO", 0.9)
        max_allowed_bytes = int(total_bytes * max_ratio)
        current_used = total_bytes - free_bytes
        available_for_us = max_allowed_bytes - current_used

        if available_for_us <= 0:
            # 여유 메모리가 거의 없을 때는 전체의 5%만 사용
            SAFE_GPU_BYTES = int(total_bytes * 0.05)
        else:
            # 현재 여유의 90%만 사용 (안전 마진)
            SAFE_GPU_BYTES = int(available_for_us * 0.9)
    except Exception:
        # 실패 시 보수적인 기본값 (예: 18GB)
        SAFE_GPU_BYTES = 18 * 1024**3

    MAX_POINTS_PER_BATCH = max(
        10_000,
        min(
            C.MAX_POINTS_PER_BATCH,
            int(SAFE_GPU_BYTES / max(1, N * BYTES_PER_ELEMENT)),
        ),
    )

    hits_per_ray = []
    num_point_batches = (M + MAX_POINTS_PER_BATCH - 1) // MAX_POINTS_PER_BATCH

    # 배치 정보 출력 (디버깅)
    if N > 1_000_000:  # 1M rays 이상일 때만
        est_mem_gb = (N * MAX_POINTS_PER_BATCH * BYTES_PER_ELEMENT) / (1024**3)
        print(f"      [Batch] Rays: {N:,}, Points/batch: {MAX_POINTS_PER_BATCH:,}, Batches: {num_point_batches}, Est mem: {est_mem_gb:.1f}GB")

    # Ray별 best hit 추적
    best_depths = cp.full(N, cp.inf, dtype=cp.float32)
    best_indices = cp.full(N, -1, dtype=cp.int32)

    for batch_idx in range(num_point_batches):
        point_start = batch_idx * MAX_POINTS_PER_BATCH
        point_end = min(point_start + MAX_POINTS_PER_BATCH, M)
        coords_batch = coords[point_start:point_end]
        M_batch = len(coords_batch)

        # 벡터화된 distance 계산
        P = coords_batch - origin  # (M_batch, 3)
        proj_dist = cp.dot(rays, P.T)  # (N, M_batch)

        # 카메라 뒤 제외
        valid_mask = proj_dist > 0.1

        # 투영점 계산
        proj_points = origin + rays[:, None, :] * proj_dist[:, :, None]

        # Residual 계산
        residuals = coords_batch[None, :, :] - proj_points  # (N, M_batch, 3)

        # 수평/수직 거리
        h_dist = cp.linalg.norm(residuals[:, :, :2], axis=2)
        v_dist = cp.abs(residuals[:, :, 2])

        # GSD 허용오차 체크
        candidates_mask = valid_mask & (h_dist <= h_tol) & (v_dist <= v_tol)

        # 각 ray별 최소 거리 업데이트
        for ray_idx in range(N):
            valid = cp.where(candidates_mask[ray_idx])[0]
            if len(valid) > 0:
                dists = proj_dist[ray_idx, valid]
                min_idx = cp.argmin(dists)
                depth = dists[min_idx]

                if depth < best_depths[ray_idx]:
                    best_depths[ray_idx] = depth
                    best_indices[ray_idx] = point_start + valid[min_idx]

    # Ray별 hits 변환
    for ray_idx in range(N):
        if best_indices[ray_idx] >= 0:
            hits_per_ray.append(cp.array([best_indices[ray_idx]], dtype=cp.int32))
        else:
            hits_per_ray.append(cp.array([], dtype=cp.int32))

    return hits_per_ray

def run(site_name: str = "Zenmuse_AI_Site_B",
        k_max: int = 1,
        use_sampling: bool = False,
        sample_size: int = 0) -> None:
    """
    Footprint-based Forward 실행

    Args:
        site_name: 사이트 이름
        k_max: 최대 hit 수
        use_sampling: 샘플링 사용
        sample_size: 샘플 크기
    """
    print(f"\n{'='*60}")
    print(f"Part3 Forward Pixel-wise Z-buffer")
    print(f"{'='*60}")
    print(f"사이트: {site_name}")
    print(f"모드: {'Single-Hit' if k_max == 1 else f'Top-{k_max}'}")
    print(f"GPU: {'활성화' if C.USE_GPU else '비활성화'}")

    # 경로 설정
    las_dir = C.PART2_DIR / site_name
    det_dir = C.DETECTION_DIRS.get(site_name)
    output_dir = C.PART3_DIR / site_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 포인트 클라우드 로드
    xyz, rgb = load_point_cloud(las_dir, use_sampling, sample_size)

    # 카메라 DB 로드
    cam_db = load_camera_db(site_name)

    # Ground Z 계산 (Z_avg - H_site)
    Z_values = [cam_info['C'][2] for cam_info in cam_db['images'].values()]
    Z_avg = float(sum(Z_values) / len(Z_values))
    H_site = C.FLIGHT_ALT_BY_SITE.get(site_name, 80.0)
    ground_Z = Z_avg - H_site

    print(f"\n[Ground Z 계산]")
    print(f"   카메라 평균 Z: {Z_avg:.2f} m")
    print(f"   H_site: {H_site:.1f} m")
    print(f"   Ground Z: {ground_Z:.2f} m")

    # MD 문서: Z 필터링 제거 (스펙 반영)
    # 모든 포인트 사용

    # Forward 투표
    start_time = time.time()

    if C.USE_GPU:
        votes = forward_vote_pixelwise_gpu(xyz, cam_db, det_dir, site_name, ground_Z, k_max)
    else:
        print(f"\n[ERROR] CPU 모드 미구현. GPU를 사용하세요.")
        return

    elapsed = time.time() - start_time

    # 결과 저장
    votes_path = output_dir / "forward_votes.npy"
    np.save(votes_path, votes)
    print(f"\n[OK] 저장: {votes_path}")

    # 임계값별 LAS 저장 (간단 버전)
    for threshold in C.VOTE_THRESHOLDS:
        mask = votes >= threshold
        if mask.sum() > 0:
            las_path = output_dir / f"vote_{threshold}.las"
            print(f"   vote_{threshold}: {mask.sum():,} points")
            # TODO: LAS 저장 구현
        else:
            print(f"   vote_{threshold}: 0 points")

    print(f"\n[OK] 완료 ({elapsed:.1f}초)")

if __name__ == "__main__":
    run()
