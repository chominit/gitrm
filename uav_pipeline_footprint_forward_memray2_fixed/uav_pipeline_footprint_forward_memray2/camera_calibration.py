# -*- coding: utf-8 -*-
"""
camera_calibration.py
Camera Footprint 계산 (MD 문서 기반 정확한 구현)
작성일: 2025-11-21
"""

import numpy as np
from typing import Dict, Tuple, Optional
import constants as C

# ==============================
# 카메라 IOP (내부 파라미터)
# ==============================

# Zenmuse P1 35mm (Site B/C)
# - Image: 8192×5460
# - Focal length: ~8201 px
# - FOV: H=53.63°, V=36.96°
ZENMUSE_IOP = {
    "fx": 8201.013000,
    "fy": 8201.013000,
    "cx": 4087.2,
    "cy": 2743.7,
    "width": 8192,
    "height": 5460,
}

# P4R FC6310 (Site A)
# - Image: 5472×3648
# - Pixel size: 2.34527 µm
# - Focal length: 8.6604 mm → ~3692.71 px
# - FOV: H=73.07°, V=52.57°
P4R_IOP = {
    "fx": 3692.709155,
    "fy": 3692.709155,
    "cx": 2724.7,
    "cy": 1823.3,
    "width": 5472,
    "height": 3648,
}

SITE_CALIBRATION: Dict[str, Dict] = {
    "Zenmuse_AI_Site_A": P4R_IOP,
    "P4R_Site_A_Solid": P4R_IOP,
    "Zenmuse_AI_Site_B": ZENMUSE_IOP,
    "Zenmuse_AI_Site_C": ZENMUSE_IOP,
    "P4R_Site_B_Solid_Merge_V2": ZENMUSE_IOP,
    "P4R_Site_C_Solid_Merge_V2": ZENMUSE_IOP,
}

# ==============================
# 카메라 매트릭스
# ==============================
def get_camera_matrix(site_name: str) -> np.ndarray:
    """
    사이트별 카메라 내부 파라미터 행렬(K) 반환

    Returns:
        K: 3x3 numpy array
    """
    if site_name not in SITE_CALIBRATION:
        iop = ZENMUSE_IOP
    else:
        iop = SITE_CALIBRATION[site_name]

    K = np.array([
        [iop["fx"], 0,         iop["cx"]],
        [0,         iop["fy"], iop["cy"]],
        [0,         0,         1.0]
    ], dtype=np.float64)

    return K

def get_image_size(site_name: str) -> Tuple[int, int]:
    """
    Returns:
        (width, height) tuple
    """
    if site_name not in SITE_CALIBRATION:
        iop = ZENMUSE_IOP
    else:
        iop = SITE_CALIBRATION[site_name]

    return (iop["width"], iop["height"])

def get_fov_deg(site_name: str) -> Tuple[float, float]:
    """
    사이트별 FOV 반환 (degrees)

    Returns:
        (fov_x, fov_y) in degrees
    """
    # Site A 계열 → P4R
    if "Site_A" in site_name:
        return (C.P4R_FOV_X_DEG, C.P4R_FOV_Y_DEG)
    # Site B/C 계열 → P1 35mm
    else:
        return (C.P1_FOV_X_DEG, C.P1_FOV_Y_DEG)

# ==============================
# Footprint 계산 (MD 문서 사양)
# ==============================
def compute_camera_footprint(
    Cw: np.ndarray,     # (3,) camera center in world coords
    R_c2w: np.ndarray,  # (3,3) camera->world rotation
    K: np.ndarray,      # (3,3) intrinsics
    width: int,
    height: int,
    ground_Z: float,
    margin: float = C.FOOTPRINT_MARGIN
) -> np.ndarray:
    """
    카메라 footprint 4-corner polygon 계산 (MD 문서 기반)

    Ground plane과 ray intersection 기반 정확한 계산

    Args:
        Cw: 카메라 중심 [X, Y, Z] (월드 좌표계)
        R_c2w: 카메라->월드 회전 행렬 (3x3)
        K: 내부 파라미터 매트릭스 (3x3)
        width, height: 이미지 크기
        ground_Z: 지면 Z 좌표
        margin: 안전 마진 (기본 0.7)

    Returns:
        corners_xy: (4, 2) array of [X, Y] ground intersections
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 1. 이미지 4 코너 픽셀 좌표
    corners_uv = np.array([
        [0, 0],          # Top-left
        [width, 0],      # Top-right
        [width, height], # Bottom-right
        [0, height]      # Bottom-left
    ], dtype=np.float64)

    # 2. 픽셀 -> 카메라 좌표계 방향벡터
    u, v = corners_uv[:, 0], corners_uv[:, 1]
    dc_x = (u - cx) / fx
    dc_y = (v - cy) / fy
    dc_z = np.ones(4)

    dirs_cam = np.column_stack([dc_x, dc_y, dc_z])  # (4, 3)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)

    # 3. 카메라 -> 월드 좌표계로 회전
    dirs_world = (R_c2w @ dirs_cam.T).T  # (4, 3)

    # 4. Ground plane과 교차점 계산
    corners_xy = []
    for d_w in dirs_world:
        dz = d_w[2]

        # 수직 레이 체크
        if abs(dz) < 1e-6:
            # 거의 수직인 경우 큰 값으로 제한
            t = 10000.0
        else:
            # t = (ground_Z - Cw_z) / dz
            t = (ground_Z - Cw[2]) / dz

        # 교차점 계산
        P = Cw + t * d_w
        corners_xy.append([P[0], P[1]])

    corners_xy = np.array(corners_xy, dtype=np.float64)  # (4, 2)

    # 5. Margin 적용
    if margin != 1.0:
        center = corners_xy.mean(axis=0)
        corners_xy = center + margin * (corners_xy - center)

    return corners_xy

def compute_camera_footprint_bbox(
    camera_pos: np.ndarray,
    site_name: str,
    margin: float = C.FOOTPRINT_MARGIN
) -> Tuple[float, float, float, float]:
    """
    간단한 bbox 계산 (H_site 상수 사용)

    MD 문서 권장 방식: FLIGHT_ALT_BY_SITE 사용

    Returns:
        (x_min, x_max, y_min, y_max)
    """
    if site_name not in SITE_CALIBRATION:
        iop = ZENMUSE_IOP
    else:
        iop = SITE_CALIBRATION[site_name]

    fx, fy = iop["fx"], iop["fy"]
    width, height = iop["width"], iop["height"]

    # FOV 계산 (radians)
    fov_x = 2 * np.arctan(width / (2 * fx))
    fov_y = 2 * np.arctan(height / (2 * fy))

    # *** 핵심: 고도는 FLIGHT_ALT_BY_SITE 사용 ***
    if site_name in C.FLIGHT_ALT_BY_SITE:
        altitude = C.FLIGHT_ALT_BY_SITE[site_name]
    else:
        # Fallback: camera_pos[2] - 100 추정
        altitude = max(float(camera_pos[2] - 100.0), 10.0)
        print(f"[WARNING] {site_name} ALT 상수 없음. 추정값 {altitude:.1f}m 사용")

    # Footprint 크기
    half_width = altitude * np.tan(fov_x / 2) * margin
    half_height = altitude * np.tan(fov_y / 2) * margin

    # Bounding box
    cx, cy = float(camera_pos[0]), float(camera_pos[1])
    x_min = cx - half_width
    x_max = cx + half_width
    y_min = cy - half_height
    y_max = cy + half_height

    return (x_min, x_max, y_min, y_max)
