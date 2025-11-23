# -*- coding: utf-8 -*-
"""
mem_limits.py

CPU/RAM 및 GPU/VRAM 사용량을 제어하기 위한 공통 유틸 모듈.
- RAM 한도: 기본 250 GB (필요 시 수정)
- VRAM 한도: 기본 90 % (GPU 전용 메모리 기준)

part3_complete_pipeline.py, part3_forward_pixelwise.py 등에서 import 해서 사용.

    from mem_limits import (
        MAX_CPU_RAM_GB,
        GPU_MEM_USAGE_RATIO,
        can_allocate_cpu_bytes,
        get_dynamic_point_batch,
    )
"""

from __future__ import annotations

import math

BYTES_PER_GB = 1024 ** 3

# 기본값 (필요하면 constants.py 에서 override 하도록 구현해도 됨)
MAX_CPU_RAM_GB: float = 250.0       # 최대 사용 RAM (GB)
GPU_MEM_USAGE_RATIO: float = 0.90   # GPU 전용 메모리 사용 상한 비율 (예: 0.9 -> 90%)


# -------------------------------
# CPU 메모리 사용량 가드
# -------------------------------

def can_allocate_cpu_bytes(needed_bytes: int, label: str = "") -> bool:
    """
    예상되는 추가 메모리 사용량(바이트)을 받아서,
    MAX_CPU_RAM_GB 를 넘을 것 같으면 False 를 반환.

    psutil 이 없으면 항상 True 를 반환 (가드 비활성화).
    """
    try:
        import psutil
    except ImportError:
        # psutil 없으면 체크하지 않음
        return True

    vm = psutil.virtual_memory()
    limit = int(MAX_CPU_RAM_GB * BYTES_PER_GB)
    used = vm.total - vm.available

    if used + needed_bytes > limit:
        gb_needed = needed_bytes / BYTES_PER_GB
        gb_used = used / BYTES_PER_GB
        gb_limit = limit / BYTES_PER_GB
        if label:
            label = f"[{label}] "
        print(
            f"[WARNING] {label}예상 CPU 메모리 초과: +{gb_needed:.1f}GB, "
            f"현재 {gb_used:.1f}GB / 한도 {gb_limit:.1f}GB"
        )
        return False

    return True


# -------------------------------
# GPU 메모리 사용량 가드
# -------------------------------

def _get_gpu_mem_info():
    """현재 GPU free/total 메모리를 바이트 단위로 반환.
    CuPy 또는 CUDA 가 없으면 (None, None) 반환.
    """
    try:
        import cupy as cp
    except Exception:
        return None, None

    try:
        free_b, total_b = cp.cuda.runtime.memGetInfo()
        return int(free_b), int(total_b)
    except Exception:
        return None, None


def can_allocate_gpu_bytes(needed_bytes: int, label: str = "") -> bool:
    """
    예상되는 추가 GPU 메모리 사용량(바이트)을 받아서,
    전체 VRAM 의 GPU_MEM_USAGE_RATIO 비율을 넘을 것 같으면 False.
    """
    free_b, total_b = _get_gpu_mem_info()
    if free_b is None or total_b is None:
        # GPU 정보를 알 수 없으면 체크하지 않음
        return True

    limit_b = int(total_b * GPU_MEM_USAGE_RATIO)
    used_b = total_b - free_b

    if used_b + needed_bytes > limit_b:
        gb_needed = needed_bytes / BYTES_PER_GB
        gb_used = used_b / BYTES_PER_GB
        gb_limit = limit_b / BYTES_PER_GB
        if label:
            label = f"[{label}] "
        print(
            f"[WARNING] {label}예상 GPU 메모리 초과: +{gb_needed:.2f}GB, "
            f"현재 {gb_used:.2f}GB / 한도 {gb_limit:.2f}GB"
        )
        return False

    return True


def get_dynamic_point_batch(n_rays: int, base_max_points: int = 30_000) -> int:
    """
    Ray 개수(n_rays)와 현재 GPU 메모리 상태를 기반으로
    point batch 크기를 동적으로 조정.

    - base_max_points: 설계 상 최대 포인트 배치 크기 (예: 30,000)
    - 반환값: 실제로 사용할 포인트 배치 크기
    """
    if n_rays <= 0:
        return base_max_points

    free_b, total_b = _get_gpu_mem_info()
    if free_b is None or total_b is None:
        return base_max_points

    limit_b = int(total_b * GPU_MEM_USAGE_RATIO)
    used_b = total_b - free_b
    remaining = max(limit_b - used_b, 0)

    # float32 기준 대략적인 per-ray/point 메모리 사용량 (안전하게 여유 있게 잡음)
    # proj_dist, proj_pts, res, h_dist, v_dist 등을 고려하여 약 48 bytes 정도로 가정.
    BYTES_PER_PAIR = 48

    if remaining <= 0:
        # 여유가 전혀 없으면 아주 작은 배치만 허용
        return max(1_000, min(base_max_points, 5_000))

    max_points_est = remaining // (max(n_rays, 1) * BYTES_PER_PAIR)

    if max_points_est <= 0:
        return max(1_000, min(base_max_points, 5_000))

    # 너무 큰 값이 나오지 않도록 base_max_points 로 상한
    max_points = int(max(1_000, min(base_max_points, max_points_est)))
    return max_points
