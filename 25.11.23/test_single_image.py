# -*- coding: utf-8 -*-
"""
test_single_image.py
단일 이미지 테스트 - 차분 이미지 ray casting
"""

from pathlib import Path
import numpy as np
import constants as C
from camera_io import load_camera_db
from img_mask import img_mask_diff

site_name = "Zenmuse_AI_Site_B"

print(f"\n{'='*60}")
print(f"Single Image Test - {site_name}")
print(f"{'='*60}")

# 경로 설정
las_dir = C.PART2_DIR / site_name
det_dir = C.DETECTION_DIRS.get(site_name)

print(f"\nLAS dir: {las_dir}")
print(f"Detection dir: {det_dir}")
print(f"Detection dir exists: {det_dir.exists()}")

# 포인트 클라우드 로드
coords_path = las_dir / "coords_float64.npy"
colors_path = las_dir / "colors_uint8.npy"

print(f"\n[Loading Point Cloud]")
if coords_path.exists() and colors_path.exists():
    coords = np.load(coords_path)
    colors = np.load(colors_path)
    print(f"  Points: {len(coords):,}")
    print(f"  Coords shape: {coords.shape}")
    print(f"  Colors shape: {colors.shape}")
else:
    print(f"  ERROR: NPY files not found")
    exit(1)

# 카메라 DB 로드
print(f"\n[Loading Camera DB]")
cam_db = load_camera_db(site_name)
images = cam_db['images']
print(f"  Images: {len(images)}")

# 첫 이미지 테스트
first_img_name = list(images.keys())[0]
print(f"\n[Testing First Image: {first_img_name}]")

# 이미지 경로
img_path = det_dir / first_img_name
print(f"  Image path: {img_path}")
print(f"  Image exists: {img_path.exists()}")

# 마스크 로드
if img_path.exists():
    print(f"\n[Loading Mask]")
    mask = img_mask_diff(str(img_path))

    if mask is not None:
        print(f"  Mask shape: {mask.shape}")
        print(f"  Mask dtype: {mask.dtype}")
        print(f"  Non-zero pixels: {mask.sum():,}")
        print(f"  Detection ratio: {100*mask.sum()/mask.size:.2f}%")

        # 픽셀 좌표
        y_coords, x_coords = np.where(mask)
        print(f"  Pixel coordinates: {len(x_coords):,}")

        if len(x_coords) > 0:
            print(f"\n  Sample pixels (first 10):")
            for i in range(min(10, len(x_coords))):
                print(f"    ({x_coords[i]}, {y_coords[i]})")
    else:
        print(f"  ERROR: Mask is None")
else:
    print(f"  ERROR: Image not found")

print(f"\n{'='*60}")
print(f"Test Complete")
print(f"{'='*60}\n")
