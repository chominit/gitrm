# -*- coding: utf-8 -*-
"""scanline_engine.py
Scanline 방식 Ray Casting 핵심 로직 (Row chunking + GPU 동적 배치).
coords 전체를 GPU에 올리지 않고, footprint 내 포인트만 CPU에서 유지하면서
각 이미지/row-chunk 밴드에 해당하는 포인트 subset만 GPU로 전송한다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import constants as C
from img_mask import img_mask_diff
from memory_utils import get_dynamic_point_batch, cleanup_gpu_memory
from geometry import r_from_opk


def compute_row_ground_band(row_idx: int,
                            img_width: int,
                            C_cam: np.ndarray,
                            R_c2w: np.ndarray,
                            K_inv: np.ndarray,
                            ground_Z: float,
                            margin: float = 2.0) -> Optional[Tuple[float, float, float, float]]:
    """단일 row에 대해 지면 투영 영역(XY band)을 근사적으로 계산한다."""
    sample_cols = [0, img_width // 4, img_width // 2, 3 * img_width // 4, img_width - 1]
    ground_points = []

    for col in sample_cols:
        pixel = np.array([col, row_idx, 1.0], dtype=np.float64)
        ray_cam = K_inv @ pixel
        ray_world = R_c2w @ ray_cam

        # 위쪽(z>0)을 보면 뒤집어서 항상 지면(음의 z) 쪽을 향하게 보정
        if ray_world[2] > 0:
            ray_world = -ray_world

        norm = np.linalg.norm(ray_world)
        if norm < 1e-6:
            continue
        ray_world /= norm

        # UAV는 아래를 촬영한다고 가정: ray_world[2]가 음수일 때 지면과 교차
        if abs(ray_world[2]) > 0.001:
            t = (ground_Z - C_cam[2]) / ray_world[2]
            if t > 0.1:  # 카메라 앞쪽만 사용
                pt = C_cam + t * ray_world
                ground_points.append(pt[:2])  # X, Y

    if not ground_points:
        return None

    g_pts = np.vstack(ground_points)
    return (
        float(g_pts[:, 0].min() - margin),
        float(g_pts[:, 0].max() + margin),
        float(g_pts[:, 1].min() - margin),
        float(g_pts[:, 1].max() + margin),
    )


def process_rays_chunk_gpu(rays_cpu: np.ndarray,
                           origin_cpu: np.ndarray,
                           coords_cpu: np.ndarray,
                           h_tol: float,
                           v_tol: float,
                           device_id: int = 0) -> np.ndarray:
    """GPU에서 Ray Casting 수행 (coords_cpu를 작은 batch로 잘라 처리)."""
    import cupy as cp  # type: ignore

    N = int(len(rays_cpu))
    M_total = int(len(coords_cpu))

    if N == 0 or M_total == 0:
        return np.full(N, -1, dtype=np.int32)

    # Rays / origin을 GPU로 전송
    rays_gpu = cp.asarray(rays_cpu, dtype=cp.float32)
    origin_gpu = cp.asarray(origin_cpu, dtype=cp.float32)

    best_depths = cp.full(N, cp.inf, dtype=cp.float32)
    best_indices = cp.full(N, -1, dtype=cp.int32)

    # 현재 GPU 메모리 상태를 고려해 point batch 크기 결정
    point_batch_size = get_dynamic_point_batch(N, base_max_points=C.MAX_POINTS_PER_BATCH, device_id=device_id)

    for start in range(0, M_total, point_batch_size):
        end = min(start + point_batch_size, M_total)
        if end <= start:
            break

        coords_batch_cpu = coords_cpu[start:end]
        pts_batch = cp.asarray(coords_batch_cpu, dtype=cp.float32)  # (M_batch, 3)

        # P = Point - Origin  -> (M_batch, 3)
        P = pts_batch - origin_gpu  # broadcasting
        # proj_dist = Ray · P  -> (N, M_batch)
        proj_dist = cp.dot(rays_gpu, P.T)

        # 카메라 앞쪽으로만
        valid_mask = proj_dist > 0.1

        # 투영점
        proj_pts = origin_gpu + rays_gpu[:, None, :] * proj_dist[:, :, None]  # (N, M_batch, 3)

        # 잔차
        res = pts_batch[None, :, :] - proj_pts  # (N, M_batch, 3)
        h_dist = cp.linalg.norm(res[:, :, :2], axis=2)  # (N, M_batch)
        v_dist = cp.abs(res[:, :, 2])  # (N, M_batch)

        candidates = valid_mask & (h_dist <= h_tol) & (v_dist <= v_tol)

        # hit가 있는 ray만 처리
        has_hit = cp.any(candidates, axis=1)
        hit_ray_indices = cp.where(has_hit)[0]

        for r_idx in hit_ray_indices:
            r = int(r_idx)
            valid_cols = cp.where(candidates[r])[0]
            depths = proj_dist[r, valid_cols]
            min_local_idx = cp.argmin(depths)
            min_depth = depths[min_local_idx]

            if min_depth < best_depths[r]:
                best_depths[r] = min_depth
                best_indices[r] = start + valid_cols[min_local_idx]

        # batch 단위로 임시 메모리 정리
        del pts_batch, P, proj_dist, proj_pts, res, h_dist, v_dist, candidates
        cp.get_default_memory_pool().free_all_blocks()

    hit_indices_cpu = np.asarray(cp.asnumpy(best_indices), dtype=np.int32)

    # GPU 메모리 정리
    del rays_gpu, origin_gpu, best_depths, best_indices
    cleanup_gpu_memory()

    return hit_indices_cpu


def process_image_chunked(img_name: str,
                          cam_info: dict,
                          fp_coords: np.ndarray,
                          fp_indices: np.ndarray,
                          det_dir: Path,
                          ground_Z: float,
                          K: np.ndarray,
                          h_tol: float,
                          v_tol: float,
                          votes: np.ndarray,
                          chunk_size: int = 32,
                          device_id: int = 0) -> int:
    """이미지를 row chunk 단위로 묶어서 처리한다.

    fp_coords / fp_indices:
        - footprint 내 포인트 좌표 및 원본 coords 인덱스.
        - 이 함수에서는 fp_coords만 CPU에 둔 상태에서 band subset을 GPU로 전송한다.
    """
    img_path = det_dir / img_name
    if not img_path.exists():
        return 0

    mask = img_mask_diff(str(img_path))
    if mask is None or not mask.any():
        return 0

    IMG_HEIGHT, IMG_WIDTH = mask.shape

    # 카메라 파라미터
    C_cam = np.array(cam_info['C'], dtype=np.float64)
    R_wc = r_from_opk(cam_info['omega'], cam_info['phi'], cam_info['kappa'])
    R_c2w = R_wc.T
    K_inv = np.linalg.inv(K)

    total_rays = 0

    # Row chunk 루프
    for row_start in range(0, IMG_HEIGHT, chunk_size):
        row_end = min(row_start + chunk_size, IMG_HEIGHT)
        chunk_mask = mask[row_start:row_end, :]

        if not chunk_mask.any():
            continue

        # Chunk 내 대표 row들로 ground band 근사
        rows_to_check = {row_start, (row_start + row_end) // 2, row_end - 1}
        x_mins, x_maxs, y_mins, y_maxs = [], [], [], []

        for r in rows_to_check:
            band = compute_row_ground_band(r, IMG_WIDTH, C_cam, R_c2w, K_inv, ground_Z)
            if band is not None:
                x_min, x_max, y_min, y_max = band
                x_mins.append(x_min)
                x_maxs.append(x_max)
                y_mins.append(y_min)
                y_maxs.append(y_max)

        if not x_mins:
            continue

        u_xmin, u_xmax = min(x_mins), max(x_maxs)
        u_ymin, u_ymax = min(y_mins), max(y_maxs)

        # Z 필터 포함 band 포인트 (CPU에서 처리)
        Z_min = ground_Z - 10.0
        Z_max = C_cam[2] + 10.0

        mask_band = (
            (fp_coords[:, 0] >= u_xmin) & (fp_coords[:, 0] <= u_xmax) &
            (fp_coords[:, 1] >= u_ymin) & (fp_coords[:, 1] <= u_ymax) &
            (fp_coords[:, 2] >= Z_min) & (fp_coords[:, 2] <= Z_max)
        )
        band_idx = np.where(mask_band)[0]
        if len(band_idx) == 0:
            continue

        band_coords = fp_coords[band_idx]

        # Chunk 내 유효 픽셀 위치 (row_start 기준 offset)
        y_rel, x_inds = np.where(chunk_mask)
        if len(x_inds) == 0:
            continue

        y_inds = y_rel + row_start
        num_rays = len(x_inds)

        # Rays 생성 (CPU)
        pixels = np.stack([x_inds, y_inds, np.ones(num_rays)], axis=1).astype(np.float64)
        rays_cam = (K_inv @ pixels.T).T
        rays_world = (R_c2w @ rays_cam.T).T
        rays_world /= np.linalg.norm(rays_world, axis=1, keepdims=True)

        # 위쪽(z>0)을 향하면 뒤집어서 지면(음의 z) 쪽을 보도록 보정
        up_mask = rays_world[:, 2] > 0
        if np.any(up_mask):
            rays_world[up_mask] *= -1

        # GPU ray casting (band_coords는 CPU)
        hit_local = process_rays_chunk_gpu(
            rays_world, C_cam, band_coords, h_tol, v_tol, device_id=device_id
        )

        # 결과 집계
        valid_hits = hit_local >= 0
        if np.any(valid_hits):
            local_idx = hit_local[valid_hits]              # band_coords 내 인덱스
            fp_local_idx = band_idx[local_idx]            # fp_coords 내 인덱스
            global_idx = fp_indices[fp_local_idx]         # 원본 coords 내 인덱스

            # numpy의 in-place 집계
            np.add.at(votes, global_idx, 1)

        total_rays += num_rays

    # 이미지 단위로 메모리 정리
    cleanup_gpu_memory()
    return int(total_rays)
