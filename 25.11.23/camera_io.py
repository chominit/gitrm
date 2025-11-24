# -*- coding: utf-8 -*-
"""
camera_io.py - GPU 최적화 카메라 파라미터 처리 모듈
Pix4D report.xml 및 EOP 파일 파싱
"""

from __future__ import annotations
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp  # type: ignore
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)

class CameraParameters:
    """카메라 파라미터 클래스"""
    
    def __init__(
        self,
        image_name: str,
        position: cp.ndarray,  # X, Y, Z
        rotation: cp.ndarray,   # Omega, Phi, Kappa
        focal_length: float,
        sensor_width: float,
        sensor_height: float,
        principal_point: Optional[Tuple[float, float]] = None,
        distortion: Optional[cp.ndarray] = None
    ):
        self.image_name = image_name
        self.position = position
        self.rotation = rotation
        self.focal_length = focal_length
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.principal_point = principal_point or (sensor_width/2, sensor_height/2)
        self.distortion = distortion
        
        # 회전 행렬 미리 계산
        self._rotation_matrix = None
        
    @property
    def rotation_matrix(self) -> cp.ndarray:
        """회전 행렬 반환 (캐시됨)"""
        if self._rotation_matrix is None:
            from geometry import create_rotation_matrix
            self._rotation_matrix = create_rotation_matrix(
                self.rotation[0],
                self.rotation[1],
                self.rotation[2],
                use_gpu=GPU_AVAILABLE
            )
        return self._rotation_matrix
    
    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            'image_name': self.image_name,
            'position': self.position.tolist() if hasattr(self.position, 'tolist') else self.position,
            'rotation': self.rotation.tolist() if hasattr(self.rotation, 'tolist') else self.rotation,
            'focal_length': self.focal_length,
            'sensor_width': self.sensor_width,
            'sensor_height': self.sensor_height,
            'principal_point': self.principal_point,
            'distortion': self.distortion.tolist() if self.distortion is not None else None
        }

def parse_pix4d_report(
    report_path: Path
) -> Dict[str, CameraParameters]:
    """
    Pix4D report.xml 파일 파싱
    
    Args:
        report_path: report.xml 파일 경로
    
    Returns:
        이미지명을 키로 하는 카메라 파라미터 딕셔너리
    """
    if not report_path.exists():
        logger.error(f"Report 파일 없음: {report_path}")
        return {}
    
    logger.info(f"Pix4D report 파싱 중: {report_path}")
    
    try:
        tree = ET.parse(report_path)
        root = tree.getroot()
        
        cameras = {}
        
        # 카메라 내부 파라미터 파싱
        camera_model = root.find('.//initialCameraModel')
        if camera_model is not None:
            focal_length = float(camera_model.find('f').text) if camera_model.find('f') is not None else 35.0
            sensor_width = float(camera_model.find('sensorWidth').text) if camera_model.find('sensorWidth') is not None else 36.0
            sensor_height = float(camera_model.find('sensorHeight').text) if camera_model.find('sensorHeight') is not None else 24.0
            
            # 주점
            cx = float(camera_model.find('cx').text) if camera_model.find('cx') is not None else sensor_width/2
            cy = float(camera_model.find('cy').text) if camera_model.find('cy') is not None else sensor_height/2
            principal_point = (cx, cy)
            
            # 왜곡 계수
            distortion = None
            if camera_model.find('k1') is not None:
                k1 = float(camera_model.find('k1').text)
                k2 = float(camera_model.find('k2').text) if camera_model.find('k2') is not None else 0.0
                k3 = float(camera_model.find('k3').text) if camera_model.find('k3') is not None else 0.0
                p1 = float(camera_model.find('p1').text) if camera_model.find('p1') is not None else 0.0
                p2 = float(camera_model.find('p2').text) if camera_model.find('p2') is not None else 0.0
                distortion = cp.array([k1, k2, p1, p2, k3], dtype=cp.float32)
        else:
            # 기본값 사용
            focal_length = 35.0
            sensor_width = 36.0
            sensor_height = 24.0
            principal_point = (18.0, 12.0)
            distortion = None
        
        # 각 이미지의 외부 파라미터 파싱
        calibrated_cameras = root.find('.//calibratedCameras')
        if calibrated_cameras is not None:
            for camera in calibrated_cameras.findall('camera'):
                image_name = camera.get('name')
                if not image_name:
                    continue
                
                # 위치 (X, Y, Z)
                position_elem = camera.find('position')
                if position_elem is not None:
                    x = float(position_elem.find('x').text)
                    y = float(position_elem.find('y').text)
                    z = float(position_elem.find('z').text)
                    position = cp.array([x, y, z], dtype=cp.float32)
                else:
                    position = cp.zeros(3, dtype=cp.float32)
                
                # 회전 (Omega, Phi, Kappa) - 라디안으로 변환
                rotation_elem = camera.find('rotation')
                if rotation_elem is not None:
                    omega = np.radians(float(rotation_elem.find('omega').text))
                    phi = np.radians(float(rotation_elem.find('phi').text))
                    kappa = np.radians(float(rotation_elem.find('kappa').text))
                    rotation = cp.array([omega, phi, kappa], dtype=cp.float32)
                else:
                    rotation = cp.zeros(3, dtype=cp.float32)
                
                # 카메라 파라미터 객체 생성
                cam_params = CameraParameters(
                    image_name=image_name,
                    position=position,
                    rotation=rotation,
                    focal_length=focal_length,
                    sensor_width=sensor_width,
                    sensor_height=sensor_height,
                    principal_point=principal_point,
                    distortion=distortion
                )
                
                cameras[image_name] = cam_params
        
        logger.info(f"카메라 파라미터 파싱 완료: {len(cameras)}개 이미지")
        return cameras
        
    except Exception as e:
        logger.error(f"Report 파싱 실패: {report_path} - {e}")
        return {}

def parse_eop_file(
    eop_path: Path,
    focal_length: float = 35.0,
    sensor_width: float = 36.0,
    sensor_height: float = 24.0,
    site_name: Optional[str] = None
) -> Dict[str, CameraParameters]:
    """
    EOP (External Orientation Parameters) 텍스트 파일 파싱
    
    파일 형식:
    이미지명 X Y Z Omega Phi Kappa
    
    Args:
        eop_path: EOP 파일 경로
        focal_length: 초점 거리
        sensor_width: 센서 너비
        sensor_height: 센서 높이
        site_name: 사이트명 (카메라 파라미터 선택용)
    
    Returns:
        이미지명을 키로 하는 카메라 파라미터 딕셔너리
    """
    if not eop_path.exists():
        logger.error(f"EOP 파일 없음: {eop_path}")
        return {}
    
    logger.info(f"EOP 파일 파싱 중: {eop_path}")
    
    # 사이트별 카메라 파라미터 설정
    if site_name:
        if "P4R" in site_name or "Site_A" in site_name:
            # DJI P4R 카메라 파라미터
            focal_length = 8.8  # mm
            sensor_width = 13.2  # mm
            sensor_height = 8.8  # mm
        elif "Zenmuse" in site_name:
            # DJI Zenmuse P1 카메라 파라미터
            focal_length = 35.0  # mm
            sensor_width = 36.0  # mm (Full Frame)
            sensor_height = 24.0  # mm
    
    cameras = {}
    
    try:
        with open(eop_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):  # 빈 줄이나 주석 무시
                continue
            
            parts = line.split()
            if len(parts) < 7:  # 최소 7개 필드 필요
                continue
            
            image_name = parts[0]
            
            # 위치 (X, Y, Z)
            position = cp.array([
                float(parts[1]),
                float(parts[2]),
                float(parts[3])
            ], dtype=cp.float32)
            
            # 회전 (Omega, Phi, Kappa) - 도에서 라디안으로 변환
            rotation = cp.array([
                np.radians(float(parts[4])),
                np.radians(float(parts[5])),
                np.radians(float(parts[6]))
            ], dtype=cp.float32)
            
            # 카메라 파라미터 객체 생성
            cam_params = CameraParameters(
                image_name=image_name,
                position=position,
                rotation=rotation,
                focal_length=focal_length,
                sensor_width=sensor_width,
                sensor_height=sensor_height,
                principal_point=(sensor_width/2, sensor_height/2),
                distortion=None
            )
            
            cameras[image_name] = cam_params
        
        logger.info(f"EOP 파싱 완료: {len(cameras)}개 이미지")
        logger.info(f"카메라 설정: f={focal_length}mm, 센서={sensor_width}x{sensor_height}mm")
        return cameras
        
    except Exception as e:
        logger.error(f"EOP 파싱 실패: {eop_path} - {e}")
        return {}

def merge_camera_parameters(
    *camera_dicts: Dict[str, CameraParameters]
) -> Dict[str, CameraParameters]:
    """
    여러 카메라 파라미터 딕셔너리 병합
    
    Args:
        camera_dicts: 카메라 파라미터 딕셔너리들
    
    Returns:
        병합된 카메라 파라미터 딕셔너리
    """
    merged = {}
    
    for cam_dict in camera_dicts:
        merged.update(cam_dict)
    
    logger.info(f"카메라 파라미터 병합 완료: 총 {len(merged)}개")
    return merged

def save_camera_parameters(
    cameras: Dict[str, CameraParameters],
    output_path: Path
) -> None:
    """
    카메라 파라미터를 JSON 파일로 저장
    
    Args:
        cameras: 카메라 파라미터 딕셔너리
        output_path: 저장할 JSON 파일 경로
    """
    import json
    
    # GPU 배열을 리스트로 변환
    camera_dict = {}
    for img_name, cam in cameras.items():
        camera_dict[img_name] = cam.to_dict()
    
    # JSON으로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(camera_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"카메라 파라미터 저장 완료: {output_path}")

def load_camera_parameters(
    json_path: Path
) -> Dict[str, CameraParameters]:
    """
    JSON 파일에서 카메라 파라미터 로드
    
    Args:
        json_path: JSON 파일 경로
    
    Returns:
        카메라 파라미터 딕셔너리
    """
    import json
    
    with open(json_path, 'r', encoding='utf-8') as f:
        camera_dict = json.load(f)
    
    cameras = {}
    for img_name, params in camera_dict.items():
        cam = CameraParameters(
            image_name=img_name,
            position=cp.array(params['position'], dtype=cp.float32),
            rotation=cp.array(params['rotation'], dtype=cp.float32),
            focal_length=params['focal_length'],
            sensor_width=params['sensor_width'],
            sensor_height=params['sensor_height'],
            principal_point=tuple(params['principal_point']),
            distortion=cp.array(params['distortion'], dtype=cp.float32) if params['distortion'] else None
        )
        cameras[img_name] = cam
    
    logger.info(f"카메라 파라미터 로드 완료: {len(cameras)}개")
    return cameras

def filter_cameras_by_site(
    cameras: Dict[str, CameraParameters],
    site_name: str
) -> Dict[str, CameraParameters]:
    """
    사이트명으로 카메라 필터링
    
    Args:
        cameras: 전체 카메라 파라미터
        site_name: 사이트명
    
    Returns:
        필터링된 카메라 파라미터
    """
    filtered = {}
    
    # 사이트명에 따른 패턴 매칭
    patterns = []
    if "Site_A" in site_name:
        patterns.extend(["DSC", "P4R", "A"])
    elif "Site_B" in site_name:
        patterns.extend(["Zenmuse", "B"])
    elif "Site_C" in site_name:
        patterns.extend(["Zenmuse", "C"])
    
    for img_name, cam in cameras.items():
        for pattern in patterns:
            if pattern.lower() in img_name.lower():
                filtered[img_name] = cam
                break
    
    if not filtered:
        # 패턴 매칭 실패 시 전체 반환
        filtered = cameras
    
    logger.info(f"사이트 {site_name} 필터링: {len(cameras)} -> {len(filtered)}개")
    return filtered
# 아래 내용을 camera_io.py 끝에 추가:

# ==============================
# 간단한 EOP 로더 (part3_forward_pixelwise.py용)
# ==============================
def load_camera_db(site_name: str) -> Dict:
    """
    카메라 DB 로드 (EOP 텍스트 기반)

    Returns:
        {'images': {img_name: {'C': [x,y,z], 'omega': ..., ...}}}
    """
    import json
    from coord_transform import wgs84_to_epsg5186

    # 현재 파일 경로
    current_dir = Path(__file__).parent

    # Site B용
    if "Site_B" in site_name:
        eop_path = current_dir / "Site_B_Images_EOP.txt"
        if eop_path.exists():
            logger.info(f"EOP 파싱: {eop_path.name} (724개)")
            return _parse_simple_eop(eop_path)

    # Site C용
    if "Site_C" in site_name:
        eop_path = current_dir / "Site_C_Images_EOP.txt"
        if eop_path.exists():
            logger.info(f"EOP 파싱: {eop_path.name} (3308개)")
            return _parse_simple_eop(eop_path)

    # Site A용
    if "Site_A" in site_name:
        eop_path = current_dir / "Site_A_Images_EOPs.txt"
        if eop_path.exists():
            logger.info(f"EOP 파싱: {eop_path.name}")
            return _parse_simple_eop(eop_path)

    raise FileNotFoundError(f"EOP 파일을 찾을 수 없음: {site_name}")


def _parse_simple_eop(eop_path: Path) -> Dict:
    """간단한 EOP 파싱 (내부 함수)"""
    from coord_transform import wgs84_to_epsg5186

    images = {}

    with open(eop_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) < 7:
                continue

            img_name = parts[0].strip()
            lat = float(parts[1])
            lon = float(parts[2])
            z = float(parts[3])
            omega = float(parts[4])
            phi = float(parts[5])
            kappa = float(parts[6])

            # WGS84 -> EPSG:5186
            x, y = wgs84_to_epsg5186(lat, lon)

            images[img_name] = {
                'C': [float(x), float(y), float(z)],
                'omega': omega,
                'phi': phi,
                'kappa': kappa,
            }

    return {'images': images}