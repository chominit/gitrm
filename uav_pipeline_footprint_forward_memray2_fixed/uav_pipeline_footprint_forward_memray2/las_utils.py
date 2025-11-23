# -*- coding: utf-8 -*-
"""
las_utils.py - GPU 최적화 LAS/LAZ 파일 처리 유틸리티
포인트 클라우드 데이터의 고속 로드/저장 처리
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import logging

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp  # type: ignore
    GPU_AVAILABLE = False

import laspy
import laspy.file

logger = logging.getLogger(__name__)

def read_las_file(
    filepath: Path,
    use_gpu: bool = True,
    return_header: bool = False
) -> Tuple[cp.ndarray, cp.ndarray, Optional[Dict[str, Any]]]:
    """
    LAS/LAZ 파일을 읽어 GPU 메모리로 로드
    
    Args:
        filepath: LAS/LAZ 파일 경로
        use_gpu: GPU 사용 여부
        return_header: 헤더 정보 반환 여부
    
    Returns:
        xyz: 3D 좌표 배열 (N, 3)
        rgb: RGB 색상 배열 (N, 3)  
        header: 헤더 정보 (선택적)
    """
    if not filepath.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")
    
    logger.info(f"LAS 파일 로드 중: {filepath}")
    
    try:
        # LAS 파일 읽기
        with laspy.open(filepath, mode='r') as las_file:
            las_data = las_file.read()
            
            # 좌표 추출
            xyz_np = np.column_stack([
                las_data.x,
                las_data.y,
                las_data.z
            ]).astype(np.float32)
            
            # RGB 색상 추출 (있는 경우)
            if hasattr(las_data, 'red') and hasattr(las_data, 'green') and hasattr(las_data, 'blue'):
                # 16-bit RGB를 8-bit로 변환
                rgb_np = np.column_stack([
                    (las_data.red / 256).astype(np.uint8),
                    (las_data.green / 256).astype(np.uint8),
                    (las_data.blue / 256).astype(np.uint8)
                ])
            else:
                # RGB가 없으면 기본값 사용
                logger.warning(f"RGB 정보 없음, 기본값 사용: {filepath}")
                rgb_np = np.full((len(xyz_np), 3), 128, dtype=np.uint8)
            
            # GPU로 전송 (필요한 경우)
            if use_gpu and GPU_AVAILABLE:
                xyz = cp.asarray(xyz_np, dtype=cp.float32)
                rgb = cp.asarray(rgb_np, dtype=cp.uint8)
                logger.info(f"데이터를 GPU로 전송 완료: {len(xyz):,} 포인트")
            else:
                xyz = xyz_np
                rgb = rgb_np
            
            # 헤더 정보
            header = None
            if return_header:
                header = {
                    'point_count': len(xyz),
                    'min_bounds': [las_data.header.x_min, las_data.header.y_min, las_data.header.z_min],
                    'max_bounds': [las_data.header.x_max, las_data.header.y_max, las_data.header.z_max],
                    'scale': [las_data.header.x_scale, las_data.header.y_scale, las_data.header.z_scale],
                    'offset': [las_data.header.x_offset, las_data.header.y_offset, las_data.header.z_offset],
                }
            
            logger.info(f"LAS 로드 완료: {len(xyz):,} 포인트")
            return xyz, rgb, header
            
    except Exception as e:
        logger.error(f"LAS 파일 로드 실패: {filepath} - {e}")
        raise

def write_las_file(
    filepath: Path,
    xyz: cp.ndarray,
    rgb: Optional[cp.ndarray] = None,
    vote_counts: Optional[cp.ndarray] = None,
    header_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    GPU 메모리에서 LAS 파일로 저장
    
    Args:
        filepath: 저장할 LAS 파일 경로
        xyz: 3D 좌표 배열 (N, 3)
        rgb: RGB 색상 배열 (N, 3)
        vote_counts: Vote count 배열 (N,)
        header_info: 헤더 정보
    """
    logger.info(f"LAS 파일 저장 중: {filepath}")
    
    # GPU 메모리에서 CPU로 전송
    if hasattr(xyz, 'get'):  # CuPy 배열인지 확인
        xyz_np = xyz.get().astype(np.float64)
        rgb_np = rgb.get() if rgb is not None else None
        vote_np = vote_counts.get() if vote_counts is not None else None
    else:
        xyz_np = xyz.astype(np.float64)
        rgb_np = rgb
        vote_np = vote_counts
    
    # 디렉터리 생성
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # LAS 파일 생성
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = [0.001, 0.001, 0.001]  # 밀리미터 정밀도
        
        # 좌표 범위 설정
        header.mins = xyz_np.min(axis=0)
        header.maxs = xyz_np.max(axis=0)
        
        # LAS 데이터 생성
        las = laspy.LasData(header)
        
        # 좌표 설정
        las.x = xyz_np[:, 0]
        las.y = xyz_np[:, 1]
        las.z = xyz_np[:, 2]
        
        # RGB 설정 (있는 경우)
        if rgb_np is not None:
            # 8-bit를 16-bit로 변환
            las.red = (rgb_np[:, 0] * 256).astype(np.uint16)
            las.green = (rgb_np[:, 1] * 256).astype(np.uint16)
            las.blue = (rgb_np[:, 2] * 256).astype(np.uint16)
        
        # Vote count를 intensity로 저장 (있는 경우)
        if vote_np is not None:
            # Vote count를 0-65535 범위로 정규화
            max_vote = vote_np.max() if vote_np.max() > 0 else 1
            las.intensity = (vote_np / max_vote * 65535).astype(np.uint16)
        
        # 파일 저장
        las.write(str(filepath))
        logger.info(f"LAS 저장 완료: {len(xyz_np):,} 포인트 -> {filepath}")
        
    except Exception as e:
        logger.error(f"LAS 파일 저장 실패: {filepath} - {e}")
        raise

def filter_points_by_vote(
    xyz: cp.ndarray,
    rgb: cp.ndarray,
    vote_counts: cp.ndarray,
    threshold: int
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Vote count 임계값으로 포인트 필터링
    
    Args:
        xyz: 3D 좌표 배열
        rgb: RGB 색상 배열
        vote_counts: Vote count 배열
        threshold: Vote count 임계값
    
    Returns:
        필터링된 xyz, rgb, vote_counts
    """
    # Vote count가 임계값 이상인 포인트만 선택
    mask = vote_counts >= threshold
    
    filtered_xyz = xyz[mask]
    filtered_rgb = rgb[mask] if rgb is not None else None
    filtered_votes = vote_counts[mask]
    
    logger.info(f"Vote 필터링: {len(xyz):,} -> {len(filtered_xyz):,} 포인트 (임계값: {threshold})")
    
    return filtered_xyz, filtered_rgb, filtered_votes

def merge_point_clouds(
    point_clouds: list[Tuple[cp.ndarray, cp.ndarray, cp.ndarray]]
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    여러 포인트 클라우드 병합
    
    Args:
        point_clouds: [(xyz, rgb, votes), ...] 리스트
    
    Returns:
        병합된 xyz, rgb, votes
    """
    if not point_clouds:
        return cp.empty((0, 3)), cp.empty((0, 3)), cp.empty(0)
    
    # 각 배열 병합
    xyz_list = [pc[0] for pc in point_clouds if pc[0] is not None]
    rgb_list = [pc[1] for pc in point_clouds if pc[1] is not None]
    vote_list = [pc[2] for pc in point_clouds if pc[2] is not None]
    
    if GPU_AVAILABLE:
        merged_xyz = cp.vstack(xyz_list) if xyz_list else cp.empty((0, 3))
        merged_rgb = cp.vstack(rgb_list) if rgb_list else cp.empty((0, 3))
        merged_votes = cp.hstack(vote_list) if vote_list else cp.empty(0)
    else:
        merged_xyz = np.vstack(xyz_list) if xyz_list else np.empty((0, 3))
        merged_rgb = np.vstack(rgb_list) if rgb_list else np.empty((0, 3))
        merged_votes = np.hstack(vote_list) if vote_list else np.empty(0)
    
    logger.info(f"포인트 클라우드 병합: {len(merged_xyz):,} 총 포인트")
    
    return merged_xyz, merged_rgb, merged_votes

def save_point_cloud_stats(
    filepath: Path,
    xyz: cp.ndarray,
    vote_counts: cp.ndarray,
    processing_time: float
) -> None:
    """
    포인트 클라우드 통계 저장
    
    Args:
        filepath: 통계 파일 경로
        xyz: 3D 좌표 배열
        vote_counts: Vote count 배열
        processing_time: 처리 시간
    """
    # GPU 메모리에서 CPU로 전송 (필요한 경우)
    if hasattr(xyz, 'get'):
        xyz_np = xyz.get()
        vote_np = vote_counts.get() 
    else:
        xyz_np = xyz
        vote_np = vote_counts
    
    stats = {
        'total_points': len(xyz_np),
        'min_coords': xyz_np.min(axis=0).tolist(),
        'max_coords': xyz_np.max(axis=0).tolist(),
        'mean_coords': xyz_np.mean(axis=0).tolist(),
        'vote_stats': {
            'min': int(vote_np.min()),
            'max': int(vote_np.max()),
            'mean': float(vote_np.mean()),
            'std': float(vote_np.std()),
        },
        'vote_distribution': {
            f'>={t}': int((vote_np >= t).sum()) 
            for t in [1, 5, 7, 10, 15, 20, 30, 50]
        },
        'processing_time_sec': processing_time,
    }
    
    # JSON 형식으로 저장
    import json
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"통계 저장 완료: {filepath}")

def load_las_directory(
    directory: Path,
    use_gpu: bool = True,
    pattern: str = "*.las"
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    디렉터리의 모든 LAS 파일 로드 및 병합
    
    Args:
        directory: LAS 파일이 있는 디렉터리
        use_gpu: GPU 사용 여부
        pattern: 파일 패턴
    
    Returns:
        병합된 xyz, rgb 배열
    """
    las_files = sorted(directory.glob(pattern))
    if not las_files:
        logger.warning(f"LAS 파일 없음: {directory}/{pattern}")
        return cp.empty((0, 3)), cp.empty((0, 3))
    
    logger.info(f"LAS 파일 {len(las_files)}개 로드 중: {directory}")
    
    point_clouds = []
    for las_file in las_files:
        try:
            xyz, rgb, _ = read_las_file(las_file, use_gpu=use_gpu)
            point_clouds.append((xyz, rgb, cp.zeros(len(xyz))))
        except Exception as e:
            logger.error(f"파일 로드 실패: {las_file} - {e}")
            continue
    
    if point_clouds:
        merged_xyz, merged_rgb, _ = merge_point_clouds(point_clouds)
        return merged_xyz, merged_rgb
    else:
        return cp.empty((0, 3)), cp.empty((0, 3))

# GPU 메모리 정리 함수
def cleanup_gpu_memory():
    """GPU 메모리 정리"""
    if GPU_AVAILABLE:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        logger.info("GPU 메모리 정리 완료")
