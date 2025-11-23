# -*- coding: utf-8 -*-
"""
multiproc_example.py

process_all_sites() 에 CPU 멀티프로세싱을 적용하는 예시 코드.
- 한 사이트당 GPU 를 사용하므로, 실제 워커 수는 1~2 정도가 안전.
- RAM 사용량을 고려해서 병렬 사이트 수를 결정해야 함.

실제 코드에 적용할 때는:
  - part3_complete_pipeline.py 의 process_all_sites() 를 본 예시처럼 교체.
"""

from __future__ import annotations

from multiprocessing import Pool, cpu_count
from typing import List
import constants as C
from part3_complete_pipeline import process_site  # 실제 모듈 경로에 맞게 수정
from mem_limits import can_allocate_cpu_bytes, BYTES_PER_GB


def process_all_sites_multiproc(sites: List[str]) -> None:
    """CPU 멀티프로세싱으로 여러 사이트를 처리하는 예시.

    constants.py 에 다음과 같은 설정을 추가해 두는 것을 권장:

        CPU_WORKERS = 2          # 동시에 돌릴 워커 수
        MAX_CPU_RAM_GB = 250.0   # 전체 RAM 상한

    """
    max_workers = getattr(C, "CPU_WORKERS", 1)
    max_workers = max(1, min(max_workers, len(sites), cpu_count()))

    if max_workers == 1:
        # 단일 프로세스 모드 (기존과 동일)
        for site in sites:
            print(f"\n[Single] Processing {site}")
            process_site(site)
        return

    print(f"\n[Multi] CPU 워커 수: {max_workers}")

    # RAM 여유가 충분한지 대략 확인 (필요 시 더 정교한 체크 가능)
    # 여기서는 단순히 "각 프로세스가 대략 50GB 쓴다"고 가정한 예.
    est_per_site_gb = getattr(C, "EST_RAM_PER_SITE_GB", 50.0)
    needed_bytes = int(est_per_site_gb * max_workers * BYTES_PER_GB)
    if not can_allocate_cpu_bytes(needed_bytes, label="process_all_sites_multiproc"):
        print("[WARNING] 예상 RAM 사용량이 한도를 넘을 수 있어, 단일 프로세스로 실행합니다.")
        for site in sites:
            process_site(site)
        return

    def _worker(site_name: str) -> None:
        try:
            process_site(site_name)
        except Exception as e:
            print(f"[ERROR] Worker for {site_name}: {e}")

    with Pool(processes=max_workers) as pool:
        pool.map(_worker, sites)
