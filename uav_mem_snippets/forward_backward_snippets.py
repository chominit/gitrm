# -*- coding: utf-8 -*-
"""
forward_backward_snippets.py

- GPU 메모리 상한(예: 24GB 의 90%)을 고려해 Ray-Point 배치 크기를 조절하는
  예시 함수 `process_rays_gpu_memsafe`
- CPU 메모리(예: 250GB)와 멀티프로세싱을 사용할 때의 패턴 예시

실제 파이프라인(`part3_complete_pipeline.py`)에 그대로 복사/붙여넣기
하거나, 필요에 맞게 수정해서 사용하면 됩니다.
"""

from __future__ import annotations
from typing import List
import numpy as np

# memory_utils 에 정의된 함수들
from memory_utils import (
    get_safe_gpu_bytes,
    get_safe_cpu_bytes,
    compute_max_point_batch,
)


# ---------------------------------------------------------------------------
# 1. GPU 메모리 상한을 반영한 Ray-Point 처리 함수
# ---------------------------------------------------------------------------

def process_rays_gpu_memsafe(
    rays: "cp.ndarray",
    origin: "cp.ndarray",
    coords: "cp.ndarray",
    h_tol: float,
    v_tol: float,
    max_ray_batch: int = 10_000,
    min_point_batch: int = 5_000,
    max_point_batch: int = 50_000,
    device_id: int = 0,
) -> "cp.ndarray":
    """
    기존 `process_rays_gpu`와 인터페이스는 동일하되,
    GPU 전용 메모리의 90% 이내에서만 동작하도록 point batch 크기를
    동적으로 조절하는 버전.

    실제 코드에서는 `part3_complete_pipeline.process_rays_gpu`를
    이 함수 내용으로 교체하면 됩니다.
    """
    import cupy as cp

    N = len(rays)
    M = len(coords)

    if M == 0:
        return cp.full(N, -1, dtype=cp.int32)

    # 결과 버퍼 (float32로 변경)
    best_indices = cp.full(N, -1, dtype=cp.int32)
    best_depths = cp.full(N, cp.inf, dtype=cp.float32)

    # float32 기준 pair 당 대략 32~40 bytes 정도 사용한다고 가정
    PER_PAIR_BYTES = 32

    # GPU에서 사용해도 되는 메모리 상한 추정 (24GB * 0.9 수준)
    safe_gpu_bytes = get_safe_gpu_bytes(device_id=device_id)
    if safe_gpu_bytes is None:
        # 정보를 얻지 못하면 보수적인 기본값 (18GB) 사용
        safe_gpu_bytes = 18 * 1024**3

    # Ray batch 루프
    for ray_start in range(0, N, max_ray_batch):
        ray_end = min(ray_start + max_ray_batch, N)
        ray_batch = rays[ray_start:ray_end]
        N_batch = len(ray_batch)

        # Ray batch 크기에 따라 point batch 크기 결정
        dynamic_point_batch = compute_max_point_batch(
            num_rays=N_batch,
            safe_gpu_bytes=safe_gpu_bytes,
            per_pair_bytes=PER_PAIR_BYTES,
            min_points=min_point_batch,
            max_points=max_point_batch,
        )

        batch_indices = cp.full(N_batch, -1, dtype=cp.int32)
        batch_depths = cp.full(N_batch, cp.inf, dtype=cp.float32)

        # Point batch 루프
        for pt_start in range(0, M, dynamic_point_batch):
            pt_end = min(pt_start + dynamic_point_batch, M)
            coords_batch = coords[pt_start:pt_end]

            # (N_batch, M_batch, 3)
            P = coords_batch - origin
            proj_dist = cp.dot(ray_batch, P.T)

            valid_mask = proj_dist > 0.1

            proj_points = origin + ray_batch[:, None, :] * proj_dist[:, :, None]

            residuals = coords_batch[None, :, :] - proj_points
            h_dist = cp.linalg.norm(residuals[:, :, :2], axis=2)
            v_dist = cp.abs(residuals[:, :, 2])

            candidates = valid_mask & (h_dist <= h_tol) & (v_dist <= v_tol)

            for ray_idx in range(N_batch):
                valid = cp.where(candidates[ray_idx])[0]
                if len(valid) == 0:
                    continue

                depths = proj_dist[ray_idx, valid]
                min_idx = cp.argmin(depths)
                depth = depths[min_idx]

                if depth < batch_depths[ray_idx]:
                    batch_depths[ray_idx] = depth
                    batch_indices[ray_idx] = pt_start + valid[min_idx]

            # 배치 내 임시 배열 정리
            del P, proj_dist, proj_points, residuals, h_dist, v_dist, candidates
            cp.get_default_memory_pool().free_all_blocks()

        best_depths[ray_start:ray_end] = batch_depths
        best_indices[ray_start:ray_end] = batch_indices

        del batch_indices, batch_depths
        cp.get_default_memory_pool().free_all_blocks()

    return best_indices


# ---------------------------------------------------------------------------
# 2. CPU 메모리 상한(250GB) 체크 예시
# ---------------------------------------------------------------------------

def check_point_cloud_memory(coords: np.ndarray, colors: np.ndarray) -> None:
    """
    coords / colors 가 대략 얼마나 메모리를 쓰는지 계산하고,
    250GB 한계를 넘는지 로그만 찍어주는 예시.

    실제 파이프라인에서는 이 함수를 `process_site` 초반부에서 호출해서
    경고만 띄우거나, 필요시 샘플링 비율을 조절하는 데 사용할 수 있다.
    """
    coords_bytes = int(coords.nbytes)
    colors_bytes = int(colors.nbytes)
    total_bytes = coords_bytes + colors_bytes

    safe_cpu_bytes = get_safe_cpu_bytes()
    gb = 1024 ** 3

    print(f"   coords: {coords_bytes / gb:.2f} GB, colors: {colors_bytes / gb:.2f} GB, total: {total_bytes / gb:.2f} GB")

    if safe_cpu_bytes is None:
        print("   [WARN] psutil 미설치 → CPU 메모리 한계 체크 불가 (수동으로 모니터링 필요)")
        return

    print(f"   [CPU] Safe limit ~ {safe_cpu_bytes / gb:.1f} GB (<= 250GB 및 현재 available 기준)")

    if total_bytes > safe_cpu_bytes:
        ratio = safe_cpu_bytes / total_bytes
        print(f"   [WARN] 예상 사용량이 안전 한계를 초과함 → 약 {ratio:.2f} 배 샘플링 필요")
        # 예시: 샘플링 비율을 적용하고 싶다면 아래와 같이 구현 가능
        # n = coords.shape[0]
        # sample_n = int(n * ratio)
        # idx = np.random.choice(n, sample_n, replace=False)
        # coords = coords[idx]
        # colors = colors[idx]
        # return coords, colors


# ---------------------------------------------------------------------------
# 3. CPU 멀티프로세싱 패턴 (GPU 작업과 분리하는 것을 권장)
# ---------------------------------------------------------------------------

def process_all_sites_cpu_parallel(site_names: List[str], max_workers: int = 4) -> None:
    """
    사이트 단위로 CPU 위주의 작업을 멀티프로세싱으로 돌릴 때의 예시 패턴.

    GPU 를 쓰는 구간과 멀티프로세싱을 섞으면 GPU 메모리가
    서로 경쟁해서 성능이 떨어질 수 있으므로,
    - CPU-only 전처리
    - LAS 후처리
    같은 부분에만 적용하는 것을 권장.
    """
    from concurrent.futures import ProcessPoolExecutor

    def _worker(site: str) -> None:
        from part3_complete_pipeline import process_site  # 실제 프로젝트 경로에 맞게 수정
        process_site(site)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for _ in ex.map(_worker, site_names):
            pass
