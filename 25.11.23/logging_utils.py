# -*- coding: utf-8 -*-
"""
logging_utils.py - 로깅 유틸리티
파이프라인 실행 로깅 및 모니터링
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

# 로거 설정
def setup_logger(
    name: str = "uav_pipeline",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    로거 설정
    
    Args:
        name: 로거 이름
        log_file: 로그 파일 경로
        level: 로그 레벨
        format_string: 로그 포맷
    
    Returns:
        설정된 로거
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 포맷 설정
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class Timer:
    """실행 시간 측정 클래스"""
    
    def __init__(self, name: str = "Timer", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.elapsed = 0
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"{self.name} 시작")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        self.logger.info(f"{self.name} 완료: {self.elapsed:.2f}초")
    
    def get_elapsed(self) -> float:
        """경과 시간 반환"""
        if self.start_time:
            return time.time() - self.start_time
        return self.elapsed

def log_progress(
    current: int,
    total: int,
    prefix: str = "진행",
    bar_length: int = 50,
    logger: Optional[logging.Logger] = None
):
    """
    진행 상황 로깅
    
    Args:
        current: 현재 값
        total: 전체 값
        prefix: 접두사
        bar_length: 진행 바 길이
        logger: 로거
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    percent = 100 * current / total
    filled = int(bar_length * current / total)
    bar = "█" * filled + "-" * (bar_length - filled)
    
    message = f"{prefix}: |{bar}| {percent:.1f}% ({current}/{total})"
    logger.info(message)

def format_size(bytes_size: int) -> str:
    """
    바이트 크기를 사람이 읽기 쉬운 형식으로 변환
    
    Args:
        bytes_size: 바이트 크기
    
    Returns:
        포맷된 문자열
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"

def format_time(seconds: float) -> str:
    """
    초를 사람이 읽기 쉬운 형식으로 변환
    
    Args:
        seconds: 초
    
    Returns:
        포맷된 문자열
    """
    if seconds < 60:
        return f"{seconds:.2f}초"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}분"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}시간"

# 전역 로거
logger = setup_logger("uav_pipeline")

# 간편 로깅 함수
def debug(msg: str):
    """디버그 로그"""
    logger.debug(msg)

def info(msg: str):
    """정보 로그"""
    logger.info(msg)

def warning(msg: str):
    """경고 로그"""
    logger.warning(msg)

def error(msg: str):
    """에러 로그"""
    logger.error(msg)

def critical(msg: str):
    """치명적 에러 로그"""
    logger.critical(msg)
