# -*- coding: utf-8 -*-
"""
UAV Pipeline GPU v26 - GPU 최적화 포인트 클라우드 처리 파이프라인

이 패키지는 UAV 이미지와 LiDAR 데이터를 처리하는 GPU 최적화 파이프라인입니다.
정방향/역방향 Ray Casting을 통해 포인트 클라우드를 필터링하고 검증합니다.
"""

__version__ = "26.0.0"
__author__ = "UAV Pipeline Team"
__license__ = "MIT"

# GPU 가용성 확인
try:
    import cupy as cp
    GPU_AVAILABLE = True
    GPU_VERSION = cp.__version__
except ImportError:
    GPU_AVAILABLE = False
    GPU_VERSION = None

# 주요 모듈 임포트
from . import constants
from . import las_utils
from . import geometry
from . import camera_io
from . import ray_casting
from . import color_gate
from . import memory_utils
from . import logging_utils
from . import part3_forward
from . import part4_backward

# 주요 클래스 및 함수 노출
from .ray_casting import RayCaster
from .camera_io import CameraParameters, parse_pix4d_report, parse_eop_file
from .las_utils import read_las_file, write_las_file, load_las_directory
from .color_gate import ColorGate
from .memory_utils import GPUMemoryManager, MemoryProfiler
from .logging_utils import Timer, setup_logger

# 버전 정보 출력
def print_version():
    """버전 정보 출력"""
    print(f"UAV Pipeline GPU v{__version__}")
    if GPU_AVAILABLE:
        print(f"GPU: 활성화 (CuPy {GPU_VERSION})")
    else:
        print("GPU: 비활성화 (CPU 모드)")

# 초기화 메시지
import logging
logger = logging.getLogger(__name__)

if GPU_AVAILABLE:
    logger.info(f"UAV Pipeline GPU v{__version__} 초기화 - GPU 모드")
else:
    logger.info(f"UAV Pipeline GPU v{__version__} 초기화 - CPU 모드")

__all__ = [
    # 버전
    "__version__",
    "GPU_AVAILABLE",
    "print_version",
    
    # 주요 클래스
    "RayCaster",
    "CameraParameters",
    "ColorGate",
    "GPUMemoryManager",
    "MemoryProfiler",
    "Timer",
    
    # 주요 함수
    "parse_pix4d_report",
    "parse_eop_file",
    "read_las_file",
    "write_las_file",
    "load_las_directory",
    "setup_logger",
    
    # 모듈
    "constants",
    "las_utils",
    "geometry",
    "camera_io",
    "ray_casting",
    "color_gate",
    "memory_utils",
    "logging_utils",
    "part3_forward",
    "part4_backward",
]
