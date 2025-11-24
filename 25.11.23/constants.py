# -*- coding: utf-8 -*-
"""
constants.py - GPU 최적화 UAV 파이프라인 전역 상수 및 경로 설정
GPU(CuPy) 기반 고속 처리를 위한 설정 모듈
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union
import logging

# ==============================
# GPU 설정 및 초기화
# ==============================
USE_GPU = True
GPU_AVAILABLE = False
CUDA_DEVICE = 0  # 기본 GPU 디바이스 번호

try:
    import cupy as cp
    import cupyx
    from cupyx.scipy import spatial as cp_spatial
    
    # GPU 메모리 풀 설정 (메모리 할당 최적화)
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    
    # GPU 사용 가능 여부 확인
    try:
        # GPU 테스트
        test_array = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        result = cp.sum(test_array).get()
        
        GPU_AVAILABLE = True
        
        # GPU 정보 출력
        device = cp.cuda.Device(CUDA_DEVICE)
        gpu_props = device.attributes
        gpu_name = gpu_props.get('DeviceName', b'Unknown').decode('utf-8')
        
        # GPU 메모리 정보
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        free_gb = free_mem / (1024**3)
        total_gb = total_mem / (1024**3)
        
        print("=" * 60)
        print("[OK] GPU mode activated")
        print(f"   - GPU device: {gpu_name}")
        print(f"   - VRAM: {free_gb:.1f}/{total_gb:.1f} GB (free/total)")
        print(f"   - CUDA capability: {gpu_props.get('ComputeCapabilityMajor', 0)}.{gpu_props.get('ComputeCapabilityMinor', 0)}")
        print(f"   - CuPy version: {cp.__version__}")
        print("=" * 60)

    except Exception as e:
        print(f"[WARNING] GPU initialization failed: {e}")
        print("   Check system requirements.")
        GPU_AVAILABLE = False
        USE_GPU = False
        
except ImportError as e:
    print("[ERROR] CuPy not installed.")
    print("   To use GPU acceleration, install:")
    print("   pip install cupy-cuda11x  # CUDA 11.x")
    print("   pip install cupy-cuda12x  # CUDA 12.x")
    GPU_AVAILABLE = False
    USE_GPU = False

# GPU를 사용할 수 없는 경우 NumPy 폴백
if not GPU_AVAILABLE:
    import numpy as cp  # type: ignore
    print("[WARNING] Falling back to NumPy mode (performance degradation expected)")

# ==============================
# 출력 디렉터리 설정 (절대 경로)
# ==============================
BASE_OUTPUT_DIR: Path = Path(r"C:\Users\jscool\uav_pipeline_outputs")

# 각 Part별 출력 디렉터리
PART1_DIR: Path = BASE_OUTPUT_DIR / "part1_io"
PART2_DIR: Path = BASE_OUTPUT_DIR / "part2_las"
PART3_DIR: Path = BASE_OUTPUT_DIR / "part3_las"
PART4_DIR: Path = BASE_OUTPUT_DIR / "part4_las"
PART5_DIR: Path = BASE_OUTPUT_DIR / "part5_las"

# 출력 디렉터리 자동 생성
for _dir in (PART1_DIR, PART2_DIR, PART3_DIR, PART4_DIR, PART5_DIR):
    try:
        _dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"디렉터리 생성 실패: {_dir} - {e}")

# ==============================
# 처리 대상 사이트 (Original 제외)
# ==============================
SITE_WHITELIST: Tuple[str, ...] = (
    # Original 사이트들은 주석 처리
    # "Site_A_Original",
    # "Site_B_Original", 
    # "Site_C_Original",
    
    # 실제 처리 대상
    "Zenmuse_AI_Site_A",
    "Zenmuse_AI_Site_B",
    "Zenmuse_AI_Site_C",
    "P4R_Site_A_Solid",
    "P4R_Site_B_Solid_Merge_V2",
    "P4R_Site_C_Solid_Merge_V2",
    "P4R_Zenmuse_Joint_AI_Site_A",
    "P4R_Zenmuse_Joint_AI_Site_B",
    "P4R_Zenmuse_Joint_AI_Site_C",
)

# ==============================
# LAS 파일 경로 매핑
# ==============================
SITE_LAS_PATHS: Dict[str, Path] = {
    # Original 사이트들 (주석 처리)
    # "Site_A_Original": PART2_DIR / "Site_A_Original",
    # "Site_B_Original": PART2_DIR / "Site_B_Original",
    # "Site_C_Original": PART2_DIR / "Site_C_Original",
    
    # 실제 처리 대상
    "Zenmuse_AI_Site_A": PART2_DIR / "Zenmuse_AI_Site_A",
    "Zenmuse_AI_Site_B": PART2_DIR / "Zenmuse_AI_Site_B",
    "Zenmuse_AI_Site_C": PART2_DIR / "Zenmuse_AI_Site_C",
    "P4R_Site_A_Solid": PART2_DIR / "P4R_Site_A_Solid",
    "P4R_Site_B_Solid_Merge_V2": PART2_DIR / "P4R_Site_B_Solid_Merge_V2",
    "P4R_Site_C_Solid_Merge_V2": PART2_DIR / "P4R_Site_C_Solid_Merge_V2",
    "P4R_Zenmuse_Joint_AI_Site_A": PART2_DIR / "P4R_Zenmuse_Joint_AI_Site_A",
    "P4R_Zenmuse_Joint_AI_Site_B": PART2_DIR / "P4R_Zenmuse_Joint_AI_Site_B",
    "P4R_Zenmuse_Joint_AI_Site_C": PART2_DIR / "P4R_Zenmuse_Joint_AI_Site_C",
}

# ==============================
# Pix4D Report XML 경로 매핑
# ==============================
SITE_REPORT_PATHS: Dict[str, Path] = {
    # Original 사이트들 (주석 처리)
    # "Site_A_Original": Path(r"G:\UAV_RANSAC\Pix4d\Site_A_Original\1_initial\report\report.xml"),
    # "Site_B_Original": Path(r"G:\UAV_RANSAC\Pix4d\Site_B_Original\1_initial\report\report.xml"),
    # "Site_C_Original": Path(r"G:\UAV_RANSAC\Pix4d\Site_C_Original\1_initial\report\report.xml"),
    
    # 실제 처리 대상
    "Zenmuse_AI_Site_A": Path(r"G:\UAV_RANSAC\Pix4d\Zenmuse_AI_Site_A\1_initial\report\report.xml"),
    "Zenmuse_AI_Site_B": Path(r"G:\UAV_RANSAC\Pix4d\Zenmuse_AI_Site_B\1_initial\report\report.xml"),
    "Zenmuse_AI_Site_C": Path(r"G:\UAV_RANSAC\Pix4d\Zenmuse_AI_Site_C\1_initial\report\report.xml"),
    
    # P4R Site들 - 프로젝트 내 report.xml 사용 또는 EOP 파일로 대체
    "P4R_Site_A_Solid": Path(r"C:\Users\jscool\uav_pipeline_outputs\P4R_Site_A_report.xml"),  # 프로젝트 파일 사용
    "P4R_Site_B_Solid_Merge_V2": Path(r"C:\Users\jscool\uav_pipeline_outputs\P4R_Site_B_report.xml"),
    "P4R_Site_C_Solid_Merge_V2": Path(r"C:\Users\jscool\uav_pipeline_outputs\P4R_Site_C_report.xml"),
    
    "P4R_Zenmuse_Joint_AI_Site_A": Path(r"G:\UAV_RANSAC\Pix4d\P4R_Zenmuse_Joint_AI_Site_A\1_initial\report\report.xml"),
    "P4R_Zenmuse_Joint_AI_Site_B": Path(r"G:\UAV_RANSAC\Pix4d\P4R_Zenmuse_Joint_AI_Site_B\1_initial\report\report.xml"),
    "P4R_Zenmuse_Joint_AI_Site_C": Path(r"G:\UAV_RANSAC\Pix4d\P4R_Zenmuse_Joint_AI_Site_C\1_initial\report\report.xml"),
}

# ==============================
# 이미지 디렉터리 경로 매핑 (F 드라이브 사용)
# ==============================
SITE_IMAGE_PATHS: Dict[str, Path] = {
    # Original 사이트들 (주석 처리)
    # "Site_A_Original": Path(r"F:\Images\Raw_Image_Data\Site_A_Original"),
    # "Site_B_Original": Path(r"F:\Images\Raw_Image_Data\Site_B_Original"),
    # "Site_C_Original": Path(r"F:\Images\Raw_Image_Data\Site_C_Original"),
    
    # 실제 처리 대상 - F 드라이브의 병합된 이미지 사용
    "Zenmuse_AI_Site_A": Path(r"F:\Images\병합된 이미지\Zenmuse_AI_Site_A"),
    "Zenmuse_AI_Site_B": Path(r"F:\Images\병합된 이미지\Zenmuse_AI_Site_B"),
    "Zenmuse_AI_Site_C": Path(r"F:\Images\병합된 이미지\Zenmuse_AI_Site_C"),
    "P4R_Site_A_Solid": Path(r"F:\Images\병합된 이미지\P4R_Site_A_Solid"),
    "P4R_Site_B_Solid_Merge_V2": Path(r"F:\Images\병합된 이미지\P4R_Site_B_Solid_Merge_V2"),
    "P4R_Site_C_Solid_Merge_V2": Path(r"F:\Images\병합된 이미지\P4R_Site_C_Solid_Merge_V2"),
    "P4R_Zenmuse_Joint_AI_Site_A": Path(r"F:\Images\병합된 이미지\P4R_Zenmuse_Joint_AI_Site_A"),
    "P4R_Zenmuse_Joint_AI_Site_B": Path(r"F:\Images\병합된 이미지\P4R_Zenmuse_Joint_AI_Site_B"),
    "P4R_Zenmuse_Joint_AI_Site_C": Path(r"F:\Images\병합된 이미지\P4R_Zenmuse_Joint_AI_Site_C"),
}

# ==============================
# EOP (External Orientation Parameters) 파일 경로
# ==============================
EOP_FILES: Dict[str, Path] = {
    "P4R_Site_A": Path(r"C:\Users\jscool\uav_pipeline_outputs\P4R Image.txt"),
    "Site_BC": Path(r"C:\Users\jscool\uav_pipeline_outputs\Zenmuse P1 Image 젠뮤즈 P1 이미지 좌표.txt"),
}

# ==============================
# Vote 임계값 설정
# ==============================
VOTE_THRESHOLDS: List[int] = [7, 15, 30]  # Vote count 임계값
DEFAULT_VOTE_THRESHOLD: int = 7

# ==============================
# Ray Casting 파라미터
# ==============================
RAY_PARAMS = {
    "max_distance": 1000.0,      # 최대 ray 거리 (미터)
    "distance_threshold": 0.5,   # 점 매칭 거리 임계값 (미터)
    "angle_threshold": 5.0,      # 각도 임계값 (도)
    "z_buffer_tolerance": 0.1,   # Z-버퍼 허용 오차
}

# ==============================
# RGB 필터링 파라미터
# ==============================
COLOR_FILTER = {
    "enabled": True,
    "r_range": (30, 225),  # R 채널 범위
    "g_range": (30, 225),  # G 채널 범위  
    "b_range": (30, 225),  # B 채널 범위
    "tolerance": 20,       # 색상 매칭 허용 오차
}

# ==============================
# GPU 메모리 관리 파라미터
# ==============================
GPU_MEMORY = {
    "batch_size": 10000,      # GPU 배치 크기
    "max_points": 10000000,   # 최대 포인트 수
    "chunk_size": 50000,      # 청크 크기
    "memory_limit_gb": 8.0,   # GPU 메모리 제한 (GB)
}

# ==============================
# 병렬 처리 설정
# ==============================
PARALLEL_CONFIG = {
    "num_workers": 4,         # 워커 스레드 수
    "use_multiprocessing": False,  # GPU에서는 멀티프로세싱 비활성화
    "batch_images": 5,        # 동시 처리 이미지 수
}

# ==============================
# 로깅 설정
# ==============================
LOG_CONFIG = {
    "level": logging.INFO,
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    "file": BASE_OUTPUT_DIR / "pipeline.log"
}

# 로깅 초기화
logging.basicConfig(
    level=LOG_CONFIG["level"],
    format=LOG_CONFIG["format"],
    handlers=[
        logging.FileHandler(LOG_CONFIG["file"]),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"GPU 파이프라인 초기화 - GPU 사용: {GPU_AVAILABLE}")
# 아래 내용을 constants.py 끝에 추가:

# ==============================
# Footprint 및 Forward Pipeline 설정
# ==============================
FOOTPRINT_MARGIN = 0.0  # Footprint 마진 제거 (GSD × FOV만 사용)
MEMORY_FACTOR = 3  # 메모리 안전 계수
MAX_MEM_GB = 10000  # GPU 메모리 최대 사용량 (배치 처리로 실제 사용량 훨씬 적음)
MAX_POINTS_PER_BATCH = 2_000_000  # 배치당 최대 포인트 수

# 메모리 제한 설정 (사용자 환경 기준)
# CPU RAM 최대 사용량 (bytes) - 250GB
MAX_CPU_MEMORY_BYTES: int = 250 * 1024**3

# GPU VRAM 최대 사용 비율 (예: 0.9 = 90%)
MAX_GPU_MEMORY_USAGE_RATIO: float = 0.9


# DETECTION_DIRS 추가 (차분된 이미지 디렉토리 - 검정 배경 + segmentation 부분만)
IMAGES_DIR = Path(r"F:\Images\차분된 이미지")
DETECTION_DIRS: Dict[str, Path] = {
    "P4R_Site_A_Solid": IMAGES_DIR / "P4R_Site_A_Solid",
    "P4R_Site_B_Solid_Merge_V2": IMAGES_DIR / "P4R_Site_B_Solid_Merge_V2",
    "P4R_Site_C_Solid_Merge_V2": IMAGES_DIR / "P4R_Site_C_Solid_Merge_V2",
    "Zenmuse_AI_Site_A": IMAGES_DIR / "Zenmuse_AI_Site_A",
    "Zenmuse_AI_Site_B": IMAGES_DIR / "Zenmuse_AI_Site_B",
    "Zenmuse_AI_Site_C": IMAGES_DIR / "Zenmuse_AI_Site_C",
    "P4R_Zenmuse_Joint_AI_Site_A": IMAGES_DIR / "P4R_Zenmuse_Joint_AI_Site_A",
    "P4R_Zenmuse_Joint_AI_Site_B": IMAGES_DIR / "P4R_Zenmuse_Joint_AI_Site_B",
    "P4R_Zenmuse_Joint_AI_Site_C": IMAGES_DIR / "P4R_Zenmuse_Joint_AI_Site_C",
}

# FLIGHT_ALT_BY_SITE (비행 고도)
FLIGHT_ALT_BY_SITE: Dict[str, float] = {
    "P4R_Site_A_Solid": 35.0,
    "Zenmuse_AI_Site_A": 35.0,
    "Zenmuse_AI_Site_B": 80.0,
    "Zenmuse_AI_Site_C": 80.0,
    "P4R_Site_B_Solid_Merge_V2": 80.0,
    "P4R_Site_C_Solid_Merge_V2": 80.0,
}

# VOTE_THRESHOLDS (투표 임계값)
VOTE_THRESHOLDS = (7, 15, 30)

# TARGET_EPSG (좌표계)
TARGET_EPSG = 5186
# GSD 허용오차
GSD_H = 0.01  # 수평: 1cm
GSD_V = 0.03  # 수직: 3cm

