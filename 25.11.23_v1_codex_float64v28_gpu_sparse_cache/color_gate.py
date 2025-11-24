# -*- coding: utf-8 -*-
"""
color_gate.py - GPU 최적화 색상 필터링 모듈
RGB 색상 기반 포인트 필터링
"""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import logging

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp  # type: ignore
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)

class ColorGate:
    """RGB 색상 필터링 클래스"""
    
    def __init__(
        self,
        r_range: Tuple[int, int] = (204, 255),
        g_range: Tuple[int, int] = (0, 51),
        b_range: Tuple[int, int] = (172, 210),
        tolerance: int = 1
    ):
        """
        초기화
        
        Args:
            r_range: R 채널 유효 범위
            g_range: G 채널 유효 범위
            b_range: B 채널 유효 범위
            tolerance: 색상 매칭 허용 오차
        """
        self.r_range = r_range
        self.g_range = g_range
        self.b_range = b_range
        self.tolerance = tolerance
        
        logger.info(f"ColorGate 초기화: R{r_range}, G{g_range}, B{b_range}, 허용오차={tolerance}")
    
    def filter_by_range(
        self,
        rgb: cp.ndarray
    ) -> cp.ndarray:
        """
        RGB 범위로 필터링
        
        Args:
            rgb: RGB 배열 (N, 3)
        
        Returns:
            유효한 포인트 마스크 (N,)
        """
        mask = (
            (rgb[:, 0] >= self.r_range[0]) & (rgb[:, 0] <= self.r_range[1]) &
            (rgb[:, 1] >= self.g_range[0]) & (rgb[:, 1] <= self.g_range[1]) &
            (rgb[:, 2] >= self.b_range[0]) & (rgb[:, 2] <= self.b_range[1])
        )
        
        valid_count = mask.sum()
        logger.debug(f"색상 범위 필터링: {len(mask):,} -> {valid_count:,} ({100*valid_count/len(mask):.1f}%)")
        
        return mask
    
    def match_colors(
        self,
        rgb1: cp.ndarray,
        rgb2: cp.ndarray,
        tolerance: Optional[int] = None
    ) -> cp.ndarray:
        """
        두 색상 집합 매칭
        
        Args:
            rgb1: 첫 번째 RGB 배열 (N, 3)
            rgb2: 두 번째 RGB 배열 (N, 3) 또는 (3,)
            tolerance: 허용 오차 (None이면 기본값 사용)
        
        Returns:
            매칭 마스크 (N,)
        """
        if tolerance is None:
            tolerance = self.tolerance
        
        # rgb2가 단일 색상인 경우 브로드캐스팅
        if rgb2.ndim == 1:
            rgb2 = rgb2[cp.newaxis, :]
        
        # 색상 차이 계산
        diff = cp.abs(rgb1.astype(cp.float64) - rgb2.astype(cp.float64))
        
        # 모든 채널이 허용 오차 내에 있는지 확인
        mask = cp.all(diff <= tolerance, axis=1)
        
        return mask
    
    def filter_outliers(
        self,
        rgb: cp.ndarray,
        percentile_low: float = 2.5,
        percentile_high: float = 97.5
    ) -> cp.ndarray:
        """
        통계적 이상치 제거
        
        Args:
            rgb: RGB 배열 (N, 3)
            percentile_low: 하위 백분위수
            percentile_high: 상위 백분위수
        
        Returns:
            이상치가 아닌 포인트 마스크 (N,)
        """
        masks = []
        
        for channel in range(3):
            values = rgb[:, channel]
            
            # 백분위수 계산
            if GPU_AVAILABLE:
                p_low = cp.percentile(values, percentile_low)
                p_high = cp.percentile(values, percentile_high)
            else:
                p_low = np.percentile(values, percentile_low)
                p_high = np.percentile(values, percentile_high)
            
            # 채널별 마스크
            channel_mask = (values >= p_low) & (values <= p_high)
            masks.append(channel_mask)
        
        # 모든 채널이 유효한 경우만
        final_mask = masks[0] & masks[1] & masks[2]
        
        valid_count = final_mask.sum()
        logger.debug(f"이상치 제거: {len(rgb):,} -> {valid_count:,} ({100*valid_count/len(rgb):.1f}%)")
        
        return final_mask
    
    def compute_color_histogram(
        self,
        rgb: cp.ndarray,
        bins: int = 32
    ) -> dict:
        """
        색상 히스토그램 계산
        
        Args:
            rgb: RGB 배열 (N, 3)
            bins: 히스토그램 빈 수
        
        Returns:
            히스토그램 정보
        """
        histograms = {}
        
        for i, channel in enumerate(['R', 'G', 'B']):
            if GPU_AVAILABLE:
                hist, edges = cp.histogram(rgb[:, i], bins=bins, range=(0, 256))
                histograms[channel] = {
                    'counts': hist.get().tolist(),
                    'edges': edges.get().tolist()
                }
            else:
                hist, edges = np.histogram(rgb[:, i], bins=bins, range=(0, 256))
                histograms[channel] = {
                    'counts': hist.tolist(),
                    'edges': edges.tolist()
                }
        
        return histograms
    
    def adaptive_filter(
        self,
        rgb: cp.ndarray,
        reference_rgb: cp.ndarray,
        adaptive_factor: float = 1.5
    ) -> cp.ndarray:
        """
        적응적 색상 필터링 (참조 색상 기반)
        
        Args:
            rgb: 필터링할 RGB 배열 (N, 3)
            reference_rgb: 참조 RGB 배열 (M, 3)
            adaptive_factor: 적응 계수
        
        Returns:
            필터 마스크 (N,)
        """
        # 참조 색상의 통계 계산
        if GPU_AVAILABLE:
            ref_mean = cp.mean(reference_rgb, axis=0)
            ref_std = cp.std(reference_rgb, axis=0)
        else:
            ref_mean = np.mean(reference_rgb, axis=0)
            ref_std = np.std(reference_rgb, axis=0)
        
        # 적응적 범위 설정
        lower_bound = ref_mean - adaptive_factor * ref_std
        upper_bound = ref_mean + adaptive_factor * ref_std
        
        # 범위 클리핑
        lower_bound = cp.maximum(lower_bound, 0)
        upper_bound = cp.minimum(upper_bound, 255)
        
        # 필터링
        mask = cp.all(
            (rgb >= lower_bound[cp.newaxis, :]) & 
            (rgb <= upper_bound[cp.newaxis, :]),
            axis=1
        )
        
        valid_count = mask.sum()
        logger.debug(f"적응적 필터링: {len(rgb):,} -> {valid_count:,} ({100*valid_count/len(rgb):.1f}%)")
        
        return mask

def apply_color_filter(
    xyz: cp.ndarray,
    rgb: cp.ndarray,
    config: dict
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    색상 필터 적용 (간편 함수)
    
    Args:
        xyz: 3D 좌표 배열 (N, 3)
        rgb: RGB 색상 배열 (N, 3)
        config: 필터 설정
    
    Returns:
        필터링된 xyz, rgb
    """
    if not config.get('enabled', False):
        return xyz, rgb
    
    # ColorGate 생성
    gate = ColorGate(
        r_range=config.get('r_range', (30, 225)),
        g_range=config.get('g_range', (30, 225)),
        b_range=config.get('b_range', (30, 225)),
        tolerance=config.get('tolerance', 20)
    )
    
    # 범위 필터링
    mask = gate.filter_by_range(rgb)
    
    # 이상치 제거 (선택적)
    if config.get('remove_outliers', False):
        outlier_mask = gate.filter_outliers(rgb)
        mask = mask & outlier_mask
    
    # 필터링 적용
    filtered_xyz = xyz[mask]
    filtered_rgb = rgb[mask]
    
    logger.info(f"색상 필터 적용: {len(xyz):,} -> {len(filtered_xyz):,} 포인트")
    
    return filtered_xyz, filtered_rgb

def compute_color_distance(
    rgb1: cp.ndarray,
    rgb2: cp.ndarray,
    metric: str = "euclidean"
) -> cp.ndarray:
    """
    색상 거리 계산
    
    Args:
        rgb1: 첫 번째 RGB 배열
        rgb2: 두 번째 RGB 배열
        metric: 거리 메트릭 ("euclidean", "manhattan", "chebyshev")
    
    Returns:
        거리 배열
    """
    diff = rgb1.astype(cp.float64) - rgb2.astype(cp.float64)
    
    if metric == "euclidean":
        distance = cp.sqrt(cp.sum(diff**2, axis=-1))
    elif metric == "manhattan":
        distance = cp.sum(cp.abs(diff), axis=-1)
    elif metric == "chebyshev":
        distance = cp.max(cp.abs(diff), axis=-1)
    else:
        raise ValueError(f"지원하지 않는 메트릭: {metric}")
    
    return distance
