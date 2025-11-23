# -*- coding: utf-8 -*-
"""
forward_gpu_memory_guard_example.py

part3_complete_pipeline.process_rays_gpu() 안에
GPU 메모리 한도(90%)를 반영하는 방법 예시.

실제 적용 시에는 해당 함수 내부 코드를 적절히 대체/수정해서 사용.
"""

import cupy as cp
from mem_limits import get_dynamic_point_batch


def process_rays_gpu_example(rays: "cp.ndarray", origin: "cp.ndarray",
                             coords: "cp.ndarray", h_tol: float, v_tol: float) -> "cp.ndarray":
    N = len(rays)
    M = len(coords)

    if M == 0:
        return cp.full(N, -1, dtype=cp.int32)

    # 기본 배치 크기
    MAX_RAY_BATCH = 10_000
    BASE_MAX_POINT_BATCH = 30_000

    best_indices = cp.full(N, -1, dtype=cp.int32)
    best_depths = cp.full(N, cp.inf, dtype=cp.float32)

    for ray_start in range(0, N, MAX_RAY_BATCH):
        ray_end = min(ray_start + MAX_RAY_BATCH, N)
        ray_batch = rays[ray_start:ray_end]
        N_batch = len(ray_batch)

        batch_indices = cp.full(N_batch, -1, dtype=cp.int32)
        batch_depths = cp.full(N_batch, cp.inf, dtype=cp.float32)

        # 현재 VRAM 상태를 보고 point batch 크기 조정
        max_point_batch = get_dynamic_point_batch(N_batch, BASE_MAX_POINT_BATCH)

        for pt_start in range(0, M, max_point_batch):
            pt_end = min(pt_start + max_point_batch, M)
            pt_batch = coords[pt_start:pt_end]

            P = pt_batch - origin  # (M_batch, 3)
            proj_dist = cp.dot(ray_batch, P.T)  # (N_batch, M_batch)

            valid = proj_dist > 0.1
            proj_pts = origin + ray_batch[:, None, :] * proj_dist[:, :, None]
            res = pt_batch[None, :, :] - proj_pts

            h_dist = cp.linalg.norm(res[:, :, :2], axis=2)
            v_dist = cp.abs(res[:, :, 2])

            hits = valid & (h_dist <= h_tol) & (v_dist <= v_tol)

            for i in range(N_batch):
                hit_idx = cp.where(hits[i])[0]
                if len(hit_idx) > 0:
                    depths = proj_dist[i, hit_idx]
                    min_idx = cp.argmin(depths)
                    depth = depths[min_idx]

                    if depth < batch_depths[i]:
                        batch_depths[i] = depth
                        batch_indices[i] = pt_start + hit_idx[min_idx]

            # 임시 배열 해제
            del P, proj_dist, valid, proj_pts, res, h_dist, v_dist, hits

        best_depths[ray_start:ray_end] = batch_depths
        best_indices[ray_start:ray_end] = batch_indices

        del batch_indices, batch_depths

    return best_indices
