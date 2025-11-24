# -*- coding: utf-8 -*-
"""
ray_casting.py - GPU 최적화 Ray Casting 모듈
정방향/역방향 Ray Casting 처리
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
import logging
import time

try:
    import cupy as cp
    from cupyx.scipy import spatial
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp  # type: ignore
    from scipy import spatial  # type: ignore
    GPU_AVAILABLE = False

from camera_io import CameraParameters
from geometry import (
    world_to_camera,
    camera_to_image,
    image_to_ray,
    ray_point_distance,
    compute_z_buffer,
    build_kdtree,
    query_kdtree
)

logger = logging.getLogger(__name__)

class RayCaster:
    """GPU 최적화 Ray Casting 클래스"""
    
    def __init__(
        self,
        point_cloud_xyz: cp.ndarray,
        point_cloud_rgb: cp.ndarray,
        camera_params: Dict[str, CameraParameters],
        config: dict
    ):
        """
        초기화
        
        Args:
            point_cloud_xyz: 포인트 클라우드 좌표 (N, 3)
            point_cloud_rgb: 포인트 클라우드 색상 (N, 3)
            camera_params: 카메라 파라미터 딕셔너리
            config: 설정 딕셔너리
        """
        self.point_cloud_xyz = point_cloud_xyz
        self.point_cloud_rgb = point_cloud_rgb
        self.camera_params = camera_params
        self.config = config
        
        # Vote counting 배열 초기화
        self.vote_counts = cp.zeros(len(point_cloud_xyz), dtype=cp.int32)
        
        # KD-Tree 구축
        self.kdtree = build_kdtree(point_cloud_xyz)
        logger.info(f"Ray Caster 초기화: {len(point_cloud_xyz):,} 포인트, {len(camera_params)} 카메라")
    
    def forward_ray_cast(
        self,
        image_path: Path,
        camera: CameraParameters,
        mode: str = "kdtree"  # "kdtree" 또는 "zbuffer"
    ) -> cp.ndarray:
        """
        정방향 Ray Casting - 이미지에서 포인트 클라우드로
        
        Args:
            image_path: 이미지 파일 경로
            camera: 카메라 파라미터
            mode: 매칭 모드 ("kdtree" 또는 "zbuffer")
        
        Returns:
            매칭된 포인트 인덱스
        """
        start_time = time.time()
        
        # 이미지 로드 및 전처리
        image_pixels = self._load_and_filter_image(image_path)
        if len(image_pixels) == 0:
            logger.warning(f"필터링 후 픽셀 없음: {image_path}")
            return cp.array([], dtype=cp.int32)
        
        logger.info(f"정방향 Ray Casting: {image_path.name}, {len(image_pixels):,} 픽셀")
        
        # Ray 생성
        ray_origins, ray_directions = image_to_ray(
            image_pixels[:, :2],  # x, y 좌표만
            camera.focal_length,
            camera.principal_point,
            camera.rotation_matrix,
            camera.position
        )
        
        # 매칭 수행
        if mode == "kdtree":
            matched_indices = self._match_rays_kdtree(
                ray_origins,
                ray_directions,
                image_pixels[:, 2:]  # RGB 값
            )
        else:  # zbuffer
            matched_indices = self._match_rays_zbuffer(
                ray_origins,
                ray_directions,
                camera,
                image_pixels[:, 2:]
            )
        
        # Vote counting 업데이트
        if len(matched_indices) > 0:
            unique_indices, counts = cp.unique(matched_indices, return_counts=True)
            self.vote_counts[unique_indices] += counts
        
        elapsed = time.time() - start_time
        logger.info(f"정방향 완료: {len(matched_indices):,} 매칭, {elapsed:.2f}초")
        
        return matched_indices
    
    def backward_ray_cast(
        self,
        matched_indices: cp.ndarray,
        camera: CameraParameters,
        image_path: Path
    ) -> cp.ndarray:
        """
        역방향 Ray Casting - 포인트 클라우드에서 이미지로
        
        Args:
            matched_indices: 정방향에서 매칭된 포인트 인덱스
            camera: 카메라 파라미터
            image_path: 이미지 파일 경로
        
        Returns:
            역방향 검증된 포인트 인덱스
        """
        if len(matched_indices) == 0:
            return cp.array([], dtype=cp.int32)
        
        start_time = time.time()
        logger.info(f"역방향 Ray Casting: {len(matched_indices):,} 포인트")
        
        # 매칭된 포인트들의 좌표와 색상
        matched_xyz = self.point_cloud_xyz[matched_indices]
        matched_rgb = self.point_cloud_rgb[matched_indices]
        
        # 카메라 좌표계로 변환
        camera_points = world_to_camera(
            matched_xyz,
            camera.position,
            camera.rotation_matrix
        )
        
        # 이미지 좌표로 투영
        image_coords = camera_to_image(
            camera_points,
            camera.focal_length,
            camera.principal_point,
            camera.distortion
        )
        
        # 이미지 범위 내에 있는 포인트만 선택
        image_width = camera.sensor_width
        image_height = camera.sensor_height
        
        valid_mask = (
            (image_coords[:, 0] >= 0) & 
            (image_coords[:, 0] < image_width) &
            (image_coords[:, 1] >= 0) & 
            (image_coords[:, 1] < image_height)
        )
        
        # 색상 필터링 적용
        if self.config.get('color_filter', {}).get('enabled', False):
            color_mask = self._apply_color_filter(matched_rgb)
            valid_mask = valid_mask & color_mask
        
        verified_indices = matched_indices[valid_mask]
        
        elapsed = time.time() - start_time
        logger.info(f"역방향 완료: {len(verified_indices):,}/{len(matched_indices):,} 검증, {elapsed:.2f}초")
        
        return verified_indices
    
    def _load_and_filter_image(
        self,
        image_path: Path
    ) -> cp.ndarray:
        """
        이미지 로드 및 색상 필터링
        
        Args:
            image_path: 이미지 파일 경로
        
        Returns:
            필터링된 픽셀 배열 (N, 5) - [x, y, r, g, b]
        """
        try:
            # OpenCV 또는 PIL로 이미지 로드
            import cv2
            image_np = cv2.imread(str(image_path))
            if image_np is None:
                raise ValueError(f"이미지 로드 실패: {image_path}")
            
            # BGR -> RGB 변환
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            
            # GPU로 전송
            if GPU_AVAILABLE:
                image = cp.asarray(image_np)
            else:
                image = image_np
            
            height, width, _ = image.shape
            
            # 픽셀 좌표 생성
            y_coords, x_coords = cp.meshgrid(
                cp.arange(height, dtype=cp.float64),
                cp.arange(width, dtype=cp.float64),
                indexing='ij'
            )
            
            # 평탄화
            x_flat = x_coords.ravel()
            y_flat = y_coords.ravel()
            rgb_flat = image.reshape(-1, 3)
            
            # 색상 필터링
            if self.config.get('color_filter', {}).get('enabled', False):
                mask = self._apply_color_filter(rgb_flat)
                x_flat = x_flat[mask]
                y_flat = y_flat[mask]
                rgb_flat = rgb_flat[mask]
            
            # 결합
            pixels = cp.column_stack([x_flat, y_flat, rgb_flat])
            
            # 서브샘플링 (필요한 경우)
            max_pixels = self.config.get('max_pixels_per_image', 100000)
            if len(pixels) > max_pixels:
                indices = cp.random.choice(len(pixels), max_pixels, replace=False)
                pixels = pixels[indices]
            
            return pixels
            
        except Exception as e:
            logger.error(f"이미지 로드 실패: {image_path} - {e}")
            return cp.array([], dtype=cp.float64).reshape(0, 5)
    
    def _apply_color_filter(
        self,
        rgb: cp.ndarray
    ) -> cp.ndarray:
        """
        색상 필터링 적용
        
        Args:
            rgb: RGB 배열 (N, 3)
        
        Returns:
            필터 마스크 (N,)
        """
        config = self.config.get('color_filter', {})
        r_range = config.get('r_range', (30, 225))
        g_range = config.get('g_range', (30, 225))
        b_range = config.get('b_range', (30, 225))
        
        mask = (
            (rgb[:, 0] >= r_range[0]) & (rgb[:, 0] <= r_range[1]) &
            (rgb[:, 1] >= g_range[0]) & (rgb[:, 1] <= g_range[1]) &
            (rgb[:, 2] >= b_range[0]) & (rgb[:, 2] <= b_range[1])
        )
        
        return mask
    
    def _match_rays_kdtree(
        self,
        ray_origins: cp.ndarray,
        ray_directions: cp.ndarray,
        pixel_colors: cp.ndarray
    ) -> cp.ndarray:
        """
        KD-Tree를 사용한 Ray 매칭
        
        Args:
            ray_origins: Ray 시작점들
            ray_directions: Ray 방향들
            pixel_colors: 픽셀 색상들
        
        Returns:
            매칭된 포인트 인덱스
        """
        matched_indices = []
        distance_threshold = self.config.get('ray_params', {}).get('distance_threshold', 0.5)
        color_tolerance = self.config.get('color_filter', {}).get('tolerance', 20)
        
        # 배치 처리
        batch_size = self.config.get('gpu_memory', {}).get('batch_size', 10000)
        num_rays = len(ray_origins)
        
        for i in range(0, num_rays, batch_size):
            batch_end = min(i + batch_size, num_rays)
            batch_origins = ray_origins[i:batch_end]
            batch_directions = ray_directions[i:batch_end]
            batch_colors = pixel_colors[i:batch_end]
            
            # 각 Ray에 대해 가장 가까운 포인트 찾기
            for j in range(len(batch_origins)):
                # Ray를 따라 샘플링
                max_distance = self.config.get('ray_params', {}).get('max_distance', 1000.0)
                num_samples = 100
                
                t_values = cp.linspace(0.1, max_distance, num_samples)
                sample_points = batch_origins[j] + t_values[:, cp.newaxis] * batch_directions[j]
                
                # KD-Tree 질의
                distances, indices = query_kdtree(
                    self.kdtree,
                    sample_points,
                    k=1,
                    distance_upper_bound=distance_threshold
                )
                
                # 유효한 매칭 찾기
                valid_mask = distances < distance_threshold
                if valid_mask.any():
                    valid_indices = indices[valid_mask]
                    
                    # 색상 검증
                    if color_tolerance > 0:
                        point_colors = self.point_cloud_rgb[valid_indices]
                        color_diff = cp.abs(point_colors - batch_colors[j])
                        color_mask = cp.all(color_diff <= color_tolerance, axis=1)
                        valid_indices = valid_indices[color_mask]
                    
                    if len(valid_indices) > 0:
                        # 가장 가까운 포인트 선택
                        matched_indices.append(valid_indices[0])
        
        return cp.array(matched_indices, dtype=cp.int32)
    
    def _match_rays_zbuffer(
        self,
        ray_origins: cp.ndarray,
        ray_directions: cp.ndarray,
        camera: CameraParameters,
        pixel_colors: cp.ndarray
    ) -> cp.ndarray:
        """
        Z-버퍼를 사용한 Ray 매칭
        
        Args:
            ray_origins: Ray 시작점들
            ray_directions: Ray 방향들
            camera: 카메라 파라미터
            pixel_colors: 픽셀 색상들
        
        Returns:
            매칭된 포인트 인덱스
        """
        # Z-버퍼 생성
        image_size = (int(camera.sensor_height), int(camera.sensor_width))
        z_buffer = compute_z_buffer(
            self.point_cloud_xyz,
            camera.position,
            camera.rotation_matrix,
            image_size,
            camera.focal_length,
            camera.principal_point
        )
        
        # 카메라 좌표계로 변환
        camera_points = world_to_camera(
            self.point_cloud_xyz,
            camera.position,
            camera.rotation_matrix
        )
        
        # 이미지 좌표로 투영
        image_coords = camera_to_image(
            camera_points,
            camera.focal_length,
            camera.principal_point,
            camera.distortion
        )
        
        matched_indices = []
        z_tolerance = self.config.get('ray_params', {}).get('z_buffer_tolerance', 0.1)
        
        # 각 포인트에 대해 Z-버퍼 검증
        for i in range(len(self.point_cloud_xyz)):
            x, y = image_coords[i]
            if x < 0 or x >= image_size[1] or y < 0 or y >= image_size[0]:
                continue
            
            # Z-버퍼 값과 비교
            x_int, y_int = int(x), int(y)
            z_value = camera_points[i, 2]
            z_buffer_value = z_buffer[y_int, x_int]
            
            if cp.abs(z_value - z_buffer_value) <= z_tolerance:
                matched_indices.append(i)
        
        return cp.array(matched_indices, dtype=cp.int32)
    
    def get_vote_filtered_points(
        self,
        vote_threshold: int
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        Vote count로 필터링된 포인트 반환
        
        Args:
            vote_threshold: Vote count 임계값
        
        Returns:
            필터링된 xyz, rgb, vote_counts
        """
        mask = self.vote_counts >= vote_threshold
        
        filtered_xyz = self.point_cloud_xyz[mask]
        filtered_rgb = self.point_cloud_rgb[mask]
        filtered_votes = self.vote_counts[mask]
        
        logger.info(f"Vote 필터링 (>={vote_threshold}): {mask.sum():,}/{len(mask):,} 포인트")
        
        return filtered_xyz, filtered_rgb, filtered_votes
