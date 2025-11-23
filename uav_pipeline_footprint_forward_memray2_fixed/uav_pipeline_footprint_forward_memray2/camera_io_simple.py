# -*- coding: utf-8 -*-
"""
camera_io_simple.py
간단한 camera DB 로더 (part3_forward_pixelwise.py용)
"""

from pathlib import Path
from typing import Dict
import json
from coord_transform import wgs84_to_epsg5186

def parse_eop_text(eop_path: Path) -> Dict:
    """
    EOP 텍스트 파일 파싱
    Format: Image_name,Lat,Lon,Z,Omega,Phi,Kappa,X_error,Y_error
    """
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

            # WGS84 -> EPSG:5186 변환
            x, y = wgs84_to_epsg5186(lat, lon)

            images[img_name] = {
                'C': [float(x), float(y), float(z)],
                'omega': omega,
                'phi': phi,
                'kappa': kappa,
            }

    return {'images': images}


def load_camera_db(site_name: str) -> Dict:
    """
    카메라 DB 로드 (EOP 텍스트 기반)

    Returns:
        {'images': {img_name: {'C': [x,y,z], 'omega': ..., ...}}}
    """
    import constants as C

    # 1. DETECTION_DIRS에서 camera_db.json 찾기
    det_dir = C.DETECTION_DIRS.get(site_name)
    if det_dir:
        cam_db_path = det_dir / "camera_db.json"
        if cam_db_path.exists():
            print(f"[INFO] {site_name}: camera_db.json 로드")
            with open(cam_db_path, 'r', encoding='utf-8') as f:
                return json.load(f)

    # 2. part1_camera_db 폴더에서 찾기
    part1_db_path = Path(r"C:\Users\jscool\uav_pipeline_outputs\part1_camera_db") / site_name / "camera_db.json"
    if part1_db_path.exists():
        print(f"[INFO] {site_name}: part1 camera_db.json 로드")
        with open(part1_db_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # 3. EOP 텍스트 파싱
    print(f"[INFO] camera_db.json 없음. EOP 텍스트 파싱 중...")

    current_dir = Path(__file__).parent

    # Site B용
    if "Site_B" in site_name:
        eop_path = current_dir / "Site_B_Images_EOP.txt"
        if eop_path.exists():
            print(f"[INFO] EOP 파싱: {eop_path.name} (724개)")
            return parse_eop_text(eop_path)

    # Site C용
    if "Site_C" in site_name:
        eop_path = current_dir / "Site_C_Images_EOP.txt"
        if eop_path.exists():
            print(f"[INFO] EOP 파싱: {eop_path.name} (3308개)")
            return parse_eop_text(eop_path)

    # Site A용
    if "Site_A" in site_name:
        eop_path = current_dir / "Site_A_Images_EOPs.txt"
        if eop_path.exists():
            print(f"[INFO] EOP 파싱: {eop_path.name}")
            return parse_eop_text(eop_path)

    raise FileNotFoundError(f"EOP 파일을 찾을 수 없음: {site_name}")
