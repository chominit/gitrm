# -*- coding: utf-8 -*-
"""
geometry.py - GPU 최적화 기하학적 계산 유틸리티
3D 좌표 변환, Ray Casting, 투영 연산 등
"""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import logging

try:
    import cupy as cp
    from cupyx.scipy import spatial
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp  # type: ignore
    from scipy import spatial  # type: ignore
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)

def create_rotation_matrix(
    omega: float,
    phi: float,
    kappa: float,
    use_gpu: bool = True
) -> cp.ndarray:
    """
    오메가, 파이, 카파 각도로 회전 행렬 생성
    
    Args:
        omega: X축 회전각 (라디안)
        phi: Y축 회전각 (라디안)
        kappa: Z축 회전각 (라디안)
        use_gpu: GPU 사용 여부
    
    Returns:
        3x3 회전 행렬
    """
    array_module = cp if use_gpu and GPU_AVAILABLE else np
    
    # X축 회전 (오메가)
    Rx = array_module.array([
        [1, 0, 0],
        [0, array_module.cos(omega), -array_module.sin(omega)],
        [0, array_module.sin(omega), array_module.cos(omega)]
    ], dtype=array_module.float32)
    
    # Y축 회전 (파이)
    Ry = array_module.array([
        [array_module.cos(phi), 0, array_module.sin(phi)],
        [0, 1, 0],
        [-array_module.sin(phi), 0, array_module.cos(phi)]
    ], dtype=array_module.float32)
    
    # Z축 회전 (카파)
    Rz = array_module.array([
        [array_module.cos(kappa), -array_module.sin(kappa), 0],
        [array_module.sin(kappa), array_module.cos(kappa), 0],
        [0, 0, 1]
    ], dtype=array_module.float32)
    
    # 회전 행렬 결합: R = Rz * Ry * Rx
    R = array_module.dot(Rz, array_module.dot(Ry, Rx))
    
    return R

def world_to_camera(
    world_points: cp.ndarray,
    camera_position: cp.ndarray,
    rotation_matrix: cp.ndarray
) -> cp.ndarray:
    """
    월드 좌표계를 카메라 좌표계로 변환
    
    Args:
        world_points: 월드 좌표 점들 (N, 3)
        camera_position: 카메라 위치 (3,)
        rotation_matrix: 회전 행렬 (3, 3)
    
    Returns:
        카메라 좌표계 점들 (N, 3)
    """
    # 평행 이동
    translated = world_points - camera_position
    
    # 회전 적용
    camera_points = cp.dot(translated, rotation_matrix.T)
    
    return camera_points

def camera_to_image(
    camera_points: cp.ndarray,
    focal_length: float,
    principal_point: Tuple[float, float],
    distortion_coeffs: Optional[cp.ndarray] = None
) -> cp.ndarray:
    """
    카메라 좌표계를 이미지 좌표계로 투영
    
    Args:
        camera_points: 카메라 좌표 점들 (N, 3)
        focal_length: 초점 거리
        principal_point: 주점 (cx, cy)
        distortion_coeffs: 왜곡 계수 [k1, k2, p1, p2, k3]
    
    Returns:
        이미지 좌표 (N, 2)
    """
    # Z 좌표가 0보다 작은 점들은 카메라 뒤에 있음
    valid_mask = camera_points[:, 2] > 0.001
    
    # 정규화된 이미지 좌표
    normalized = cp.zeros((len(camera_points), 2), dtype=cp.float32)
    normalized[valid_mask, 0] = camera_points[valid_mask, 0] / camera_points[valid_mask, 2]
    normalized[valid_mask, 1] = camera_points[valid_mask, 1] / camera_points[valid_mask, 2]
    
    # 렌즈 왜곡 적용 (있는 경우)
    if distortion_coeffs is not None:
        r2 = normalized[:, 0]**2 + normalized[:, 1]**2
        k1, k2, p1, p2, k3 = distortion_coeffs
        
        # 방사 왜곡
        radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
        
        # 접선 왜곡
        x_distorted = normalized[:, 0] * radial + 2*p1*normalized[:, 0]*normalized[:, 1] + p2*(r2 + 2*normalized[:, 0]**2)
        y_distorted = normalized[:, 1] * radial + p1*(r2 + 2*normalized[:, 1]**2) + 2*p2*normalized[:, 0]*normalized[:, 1]
        
        normalized[:, 0] = x_distorted
        normalized[:, 1] = y_distorted
    
    # 픽셀 좌표로 변환
    image_points = cp.zeros_like(normalized)
    image_points[:, 0] = focal_length * normalized[:, 0] + principal_point[0]
    image_points[:, 1] = focal_length * normalized[:, 1] + principal_point[1]
    
    # 유효하지 않은 점들은 -1로 설정
    image_points[~valid_mask] = -1
    
    return image_points

def image_to_ray(
    image_points: cp.ndarray,
    focal_length: float,
    principal_point: Tuple[float, float],
    rotation_matrix: cp.ndarray,
    camera_position: cp.ndarray
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    이미지 좌표에서 3D Ray 생성
    
    Args:
        image_points: 이미지 좌표 (N, 2)
        focal_length: 초점 거리
        principal_point: 주점 (cx, cy)
        rotation_matrix: 회전 행렬 (3, 3)
        camera_position: 카메라 위치 (3,)
    
    Returns:
        ray_origins: Ray 시작점들 (N, 3)
        ray_directions: Ray 방향 벡터들 (N, 3)
    """
    # 정규화된 이미지 좌표
    normalized = cp.zeros((len(image_points), 3), dtype=cp.float32)
    normalized[:, 0] = (image_points[:, 0] - principal_point[0]) / focal_length
    normalized[:, 1] = (image_points[:, 1] - principal_point[1]) / focal_length
    normalized[:, 2] = 1.0
    
    # 카메라 좌표계에서 월드 좌표계로 변환
    ray_directions = cp.dot(normalized, rotation_matrix)
    
    # 정규화
    ray_directions = ray_directions / cp.linalg.norm(ray_directions, axis=1, keepdims=True)
    
    # Ray 시작점은 카메라 위치
    ray_origins = cp.tile(camera_position, (len(image_points), 1))
    
    return ray_origins, ray_directions

def ray_point_distance(
    ray_origin: cp.ndarray,
    ray_direction: cp.ndarray,
    points: cp.ndarray
) -> cp.ndarray:
    """
    Ray와 점들 사이의 최단 거리 계산
    
    Args:
        ray_origin: Ray 시작점 (3,)
        ray_direction: Ray 방향 (3,)
        points: 3D 점들 (N, 3)
    
    Returns:
        거리 배열 (N,)
    """
    # 점들에서 ray 시작점까지의 벡터
    to_points = points - ray_origin
    
    # ray 방향으로의 투영 길이
    projections = cp.dot(to_points, ray_direction)
    
    # 음수 투영은 ray 뒤에 있는 점들
    projections = cp.maximum(projections, 0)
    
    # ray 상의 가장 가까운 점
    closest_points_on_ray = ray_origin + projections[:, cp.newaxis] * ray_direction
    
    # 거리 계산
    distances = cp.linalg.norm(points - closest_points_on_ray, axis=1)
    
    return distances

def compute_z_buffer(
    points: cp.ndarray,
    camera_position: cp.ndarray,
    rotation_matrix: cp.ndarray,
    image_size: Tuple[int, int],
    focal_length: float,
    principal_point: Tuple[float, float]
) -> cp.ndarray:
    """
    Z-버퍼 계산 (깊이 맵)
    
    Args:
        points: 3D 점들 (N, 3)
        camera_position: 카메라 위치
        rotation_matrix: 회전 행렬
        image_size: 이미지 크기 (height, width)
        focal_length: 초점 거리
        principal_point: 주점
    
    Returns:
        Z-버퍼 (height, width)
    """
    # 카메라 좌표계로 변환
    camera_points = world_to_camera(points, camera_position, rotation_matrix)
    
    # 이미지 좌표로 투영
    image_coords = camera_to_image(camera_points, focal_length, principal_point)
    
    # Z-버퍼 초기화 (무한대)
    z_buffer = cp.full(image_size, cp.inf, dtype=cp.float32)
    
    # 유효한 투영만 처리
    valid_mask = (image_coords[:, 0] >= 0) & (image_coords[:, 0] < image_size[1]) & \
                 (image_coords[:, 1] >= 0) & (image_coords[:, 1] < image_size[0])
    
    if valid_mask.any():
        valid_coords = image_coords[valid_mask].astype(cp.int32)
        valid_depths = camera_points[valid_mask, 2]
        
        # 각 픽셀에 대해 최소 깊이 값 저장
        for i in range(len(valid_coords)):
            y, x = valid_coords[i, 1], valid_coords[i, 0]
            z_buffer[y, x] = cp.minimum(z_buffer[y, x], valid_depths[i])
    
    return z_buffer

def build_kdtree(
    points: cp.ndarray,
    leaf_size: int = 10
) -> spatial.KDTree:
    """
    KD-Tree 구축
    
    Args:
        points: 3D 점들 (N, 3)
        leaf_size: 리프 노드 크기
    
    Returns:
        KD-Tree 객체
    """
    # CPU로 전송하여 KD-Tree 구축 (scipy 사용)
    if hasattr(points, 'get'):
        points_np = points.get()
    else:
        points_np = points
    
    kdtree = spatial.KDTree(points_np, leafsize=leaf_size)
    logger.info(f"KD-Tree 구축 완료: {len(points_np):,} 포인트")
    
    return kdtree

def query_kdtree(
    kdtree: spatial.KDTree,
    query_points: cp.ndarray,
    k: int = 1,
    distance_upper_bound: float = cp.inf
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    KD-Tree 질의
    
    Args:
        kdtree: KD-Tree 객체
        query_points: 질의 점들 (N, 3)
        k: 찾을 최근접 이웃 수
        distance_upper_bound: 최대 거리
    
    Returns:
        distances: 거리 배열
        indices: 인덱스 배열
    """
    # CPU로 전송하여 질의
    if hasattr(query_points, 'get'):
        query_np = query_points.get()
    else:
        query_np = query_points
    
    distances, indices = kdtree.query(
        query_np,
        k=k,
        distance_upper_bound=distance_upper_bound
    )
    
    # GPU로 다시 전송
    if GPU_AVAILABLE:
        distances = cp.asarray(distances)
        indices = cp.asarray(indices)
    
    return distances, indices

def compute_normal_vectors(
    points: cp.ndarray,
    k_neighbors: int = 10
) -> cp.ndarray:
    """
    포인트 클라우드의 법선 벡터 계산
    
    Args:
        points: 3D 점들 (N, 3)
        k_neighbors: 이웃 점 개수
    
    Returns:
        법선 벡터 (N, 3)
    """
    # KD-Tree 구축
    kdtree = build_kdtree(points)
    
    # 각 점에 대해 이웃 찾기
    _, indices = query_kdtree(kdtree, points, k=k_neighbors)
    
    normals = cp.zeros_like(points)
    
    # 각 점에 대해 PCA로 법선 계산
    for i in range(len(points)):
        if hasattr(points, 'get'):
            neighbors = points.get()[indices.get()[i]]
        else:
            neighbors = points[indices[i]]
        
        # 중심 제거
        centered = neighbors - neighbors.mean(axis=0)
        
        # 공분산 행렬
        cov = cp.cov(centered.T) if GPU_AVAILABLE else np.cov(centered.T)
        
        # 고유값 분해
        eigenvalues, eigenvectors = cp.linalg.eigh(cov) if GPU_AVAILABLE else np.linalg.eigh(cov)
        
        # 가장 작은 고유값에 해당하는 고유벡터가 법선
        normals[i] = eigenvectors[:, 0]
    
    return normals
# 아래 내용을 geometry.py 끝에 추가:

# ==============================
# 간단한 기하학 함수들 (part3_forward_pixelwise.py용)
# ==============================

def r_from_opk(omega: float, phi: float, kappa: float) -> np.ndarray:
    """
    Omega-Phi-Kappa 각도로부터 World->Camera 회전 행렬 생성

    Args:
        omega: X축 회전 (degrees)
        phi: Y축 회전 (degrees)
        kappa: Z축 회전 (degrees)

    Returns:
        R_wc: 3x3 회전 행렬 (World to Camera)
    """
    # Degrees -> Radians
    o = np.radians(omega)
    p = np.radians(phi)
    k = np.radians(kappa)

    # 회전 행렬 (ZYX 순서)
    Ro = np.array([
        [1, 0, 0],
        [0, np.cos(o), -np.sin(o)],
        [0, np.sin(o), np.cos(o)]
    ])

    Rp = np.array([
        [np.cos(p), 0, np.sin(p)],
        [0, 1, 0],
        [-np.sin(p), 0, np.cos(p)]
    ])

    Rk = np.array([
        [np.cos(k), -np.sin(k), 0],
        [np.sin(k), np.cos(k), 0],
        [0, 0, 1]
    ])

    # R_wc = Rk @ Rp @ Ro
    R_wc = Rk @ Rp @ Ro

    return R_wc


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    Ray casting 알고리즘으로 점이 다각형 내부에 있는지 확인

    Args:
        point: (2,) 점 좌표 [x, y]
        polygon: (N, 2) 다각형 꼭짓점 좌표

    Returns:
        내부면 True, 외부면 False
    """
    x, y = point[0], point[1]
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """
    여러 점들이 다각형 내부에 있는지 벡터화 확인

    Args:
        points: (N, 2) 점 좌표들
        polygon: (M, 2) 다각형 꼭짓점 좌표

    Returns:
        (N,) boolean 배열 - 내부면 True
    """
    mask = np.array([point_in_polygon(pt, polygon) for pt in points], dtype=bool)
    return mask