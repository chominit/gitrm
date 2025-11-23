# -*- coding: utf-8 -*-
"""
memory_utils.py

CPU / GPU 메모리 상태를 조회하고,
배치 크기를 동적으로 조절하기 위한 보조 유틸리티.

- CPU: psutil 이 설치되어 있으면 실제 메모리 사용량 기준으로 동작
- GPU: CuPy 를 통해 현재 디바이스의 메모리 정보를 조회

이 모듈은 독립적으로 동작하도록 설계되었으며,
기존 파이프라인 코드에서 import 해서 사용하면 됩니다.
"""

from __future__ import annotations
from typing import Optional, Tuple


# 하드 제한(논리적인 상한선)
CPU_HARD_LIMIT_GB: float = 250.0   # 사용자가 보유한 RAM 상한
GPU_USE_RATIO: float = 0.9         # GPU 전체 메모리의 90%까지만 사용


# ---------------------------------------------------------------------------
# CPU 메모리
# ---------------------------------------------------------------------------

def get_cpu_memory_state() -> Tuple[int, int]:
    """
    현재 CPU 메모리 상태를 반환한다.

    Returns
    -------
    total : int
        전체 메모리 용량 (bytes)
    available : int
        현재 가용 메모리 (bytes)

    psutil 이 없는 경우 (ImportError 발생 시) (0, 0) 을 반환한다.
    """
    try:
        import psutil  # type: ignore
    except ImportError:
        return 0, 0

    vm = psutil.virtual_memory()
    return int(vm.total), int(vm.available)


def get_safe_cpu_bytes(
    hard_limit_gb: float = CPU_HARD_LIMIT_GB,
    safety_ratio: float = 0.9,
) -> Optional[int]:
    """
    파이프라인에서 사용해도 안전한 CPU 메모리 상한(바이트)을 추정한다.

    Parameters
    ----------
    hard_limit_gb : float
        논리적인 상한선 (예: 250GB).
    safety_ratio : float
        현재 available 메모리의 몇 %까지만 사용할지 (기본 90%).

    Returns
    -------
    safe_bytes : Optional[int]
        사용 가능한 메모리 상한 (bytes).
        psutil 이 없거나 정보를 얻지 못하면 None.
    """
    total, available = get_cpu_memory_state()
    if total == 0:
        return None

    hard_limit_bytes = int(min(total, hard_limit_gb * (1024 ** 3)))
    soft_limit_bytes = int(available * safety_ratio)

    return min(hard_limit_bytes, soft_limit_bytes)


# ---------------------------------------------------------------------------
# GPU 메모리 (CuPy)
# ---------------------------------------------------------------------------

def get_gpu_memory_state(device_id: int = 0) -> Tuple[int, int]:
    """
    현재 GPU(device_id)의 메모리 상태를 반환한다.

    Returns
    -------
    total : int
        GPU 전체 메모리 (bytes)
    free : int
        현재 가용 메모리 (bytes)

    CuPy 가 없으면 (0, 0) 반환.
    """
    try:
        import cupy as cp  # type: ignore
    except ImportError:
        return 0, 0

    dev = cp.cuda.Device(device_id)
    free_bytes, total_bytes = dev.mem_info  # free, total
    return int(total_bytes), int(free_bytes)


def get_safe_gpu_bytes(
    device_id: int = 0,
    use_ratio: float = GPU_USE_RATIO,
    safety_ratio: float = 0.9,
) -> Optional[int]:
    """
    파이프라인에서 사용해도 안전한 GPU 메모리 상한(바이트)을 추정한다.

    Parameters
    ----------
    device_id : int
        사용할 GPU ID
    use_ratio : float
        GPU 전체 메모리의 몇 %까지만 쓸지 (예: 0.9 → 90%)
    safety_ratio : float
        현재 free 메모리의 몇 %까지만 쓸지 (기본 90%)

    Returns
    -------
    safe_bytes : Optional[int]
        사용 가능한 메모리 상한 (bytes).
        CuPy 가 없거나 정보를 얻지 못하면 None.
    """
    total, free = get_gpu_memory_state(device_id=device_id)
    if total == 0:
        return None

    hard_limit = int(total * use_ratio)
    soft_limit = int(free * safety_ratio)
    return min(hard_limit, soft_limit)


# ---------------------------------------------------------------------------
# 배치 크기 계산 보조 함수
# ---------------------------------------------------------------------------

def compute_max_point_batch(
    num_rays: int,
    safe_gpu_bytes: int,
    per_pair_bytes: int,
    min_points: int = 5_000,
    max_points: int = 50_000,
) -> int:
    """
    Ray 개수와 안전한 GPU 메모리 한계를 기반으로,
    한 번에 처리할 point 개수 (point batch size)를 계산한다.

    Parameters
    ----------
    num_rays : int
        한 번에 처리할 ray 개수 (N_batch)
    safe_gpu_bytes : int
        사용 가능한 GPU 메모리 상한 (bytes)
    per_pair_bytes : int
        Ray-Point pair 하나를 처리하는 데 필요한 대략적인 메모리 (bytes).
        (예: float32 기준 32~40 bytes 정도로 잡으면 됨)
    min_points : int
        최소 point batch 크기
    max_points : int
        최대 point batch 크기 (알고리즘 상의 하드 캡)

    Returns
    -------
    int
        결정된 point batch 크기
    """
    if safe_gpu_bytes <= 0 or num_rays <= 0:
        return max_points

    max_pairs = safe_gpu_bytes // per_pair_bytes
    if max_pairs <= 0:
        return min_points

    est_points = max_pairs // num_rays
    if est_points <= 0:
        return min_points

    return int(max(min_points, min(max_points, est_points)))
