# -*- coding: utf-8 -*-
"""memory_utils.py
GPU/CPU 메모리 상태 조회 및 동적 배치 크기 계산 유틸리티.
대용량 포인트 클라우드 처리 시 VRAM 90% 이내에서만 동작하도록 배치 크기를 조정한다.
"""
from __future__ import annotations

from typing import Tuple, Optional

# ==========================================
# Constants
# ==========================================

# 시스템 RAM 상한 (현재는 참고용 상수)
CPU_HARD_LIMIT_GB: float = 250.0

# GPU VRAM 사용 비율 (예: 0.90 = 총 VRAM의 90%까지만 사용)
GPU_MEM_USAGE_RATIO: float = 0.90

# Ray–Point 쌍당 예상 메모리 사용량 (bytes, float32 기준 보수적 추정)
BYTES_PER_PAIR_ESTIMATE: int = 48

# 기본 포인트 배치 크기
DEFAULT_BATCH_SIZE: int = 30_000


# ==========================================
# GPU 메모리 유틸
# ==========================================

def get_gpu_memory_state(device_id: int = 0) -> Tuple[int, int]:
    """GPU(device_id)의 (free_bytes, total_bytes)를 바이트 단위로 반환.
    CuPy가 없거나 오류가 나면 (0, 0)을 반환한다.
    """
    try:
        import cupy as cp  # type: ignore

        dev = cp.cuda.Device(device_id)
        free_bytes, total_bytes = dev.mem_info
        return int(free_bytes), int(total_bytes)
    except Exception:
        return 0, 0


def get_dynamic_point_batch(num_rays: int,
                            base_max_points: int = DEFAULT_BATCH_SIZE,
                            device_id: int = 0) -> int:
    """현재 GPU 메모리 상태와 ray 개수를 기반으로 안전한 point batch 크기를 계산한다.

    메모리 사용량을
        num_rays × num_points × BYTES_PER_PAIR_ESTIMATE
    로 근사하고, VRAM의 GPU_MEM_USAGE_RATIO(예: 90%) 이내에서만 동작하도록
    num_points를 조정한다.
    """
    if num_rays <= 0:
        return base_max_points

    free_bytes, total_bytes = get_gpu_memory_state(device_id=device_id)

    # GPU 정보를 얻지 못하면 기본값의 절반 정도로 보수적으로 반환
    if total_bytes == 0:
        return max(1_000, base_max_points // 2)

    # 총 VRAM 중 우리가 사용해도 되는 상한
    limit_bytes = int(total_bytes * GPU_MEM_USAGE_RATIO)
    used_bytes = total_bytes - free_bytes
    available_for_us = max(0, limit_bytes - used_bytes)

    # 여유가 거의 없으면 아주 작은 배치만 허용
    if available_for_us < 100 * 1024 * 1024:  # 100MB 미만
        return 1_000

    # 메모리 사용량 ≈ num_rays × num_points × BYTES_PER_PAIR_ESTIMATE
    max_points_est = available_for_us // max(num_rays * BYTES_PER_PAIR_ESTIMATE, 1)

    # 너무 작거나 크지 않게 클리핑
    final_batch = int(max(1_000, min(base_max_points, max_points_est)))
    return final_batch


# ==========================================
# CPU 메모리 유틸 (선택사항)
# ==========================================

def get_safe_cpu_bytes(hard_limit_gb: float = CPU_HARD_LIMIT_GB,
                       safety_ratio: float = 0.9) -> Optional[int]:
    """현재 시스템에서 안전하게 사용할 수 있는 CPU 메모리 (bytes)를 반환.

    hard_limit_gb: 절대 상한 (예: 250GB)
    safety_ratio: 현재 available 메모리의 몇 %까지 쓸지 (예: 0.9)
    """
    try:
        import psutil  # type: ignore
    except Exception:
        return None

    vm = psutil.virtual_memory()
    total_bytes = vm.total
    avail_bytes = vm.available

    hard_bytes = int(min(total_bytes, hard_limit_gb * 1024**3))
    soft_bytes = int(avail_bytes * safety_ratio)
    return min(hard_bytes, soft_bytes)


# ==========================================
# GPU 메모리 정리
# ==========================================

def cleanup_gpu_memory() -> None:
    """CuPy 메모리 풀을 명시적으로 정리한다.
    CPU 모드에서는 아무 일도 하지 않는다.
    """
    try:
        import cupy as cp  # type: ignore

        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    except Exception:
        # CuPy 미설치 또는 기타 오류 시 무시
        pass
