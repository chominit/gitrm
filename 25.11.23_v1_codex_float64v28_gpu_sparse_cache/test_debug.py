#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_debug.py
디버깅 테스트
"""

import numpy as np
from pathlib import Path

# 포인트 클라우드 확인
las_dir = Path(r"C:\Users\jscool\uav_pipeline_outputs\part2_las\Zenmuse_AI_Site_B")
coords_path = las_dir / "coords_float64.npy"

if coords_path.exists():
    coords = np.load(coords_path)
    print(f"포인트 수: {len(coords):,}")

    # 범위 확인
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()

    print(f"\nX 범위: {x_min:.1f} ~ {x_max:.1f} ({x_max-x_min:.1f}m)")
    print(f"Y 범위: {y_min:.1f} ~ {y_max:.1f} ({y_max-y_min:.1f}m)")
    print(f"Z 범위: {z_min:.1f} ~ {z_max:.1f} ({z_max-z_min:.1f}m)")

    area = (x_max - x_min) * (y_max - y_min)
    density = len(coords) / area
    print(f"\n전체 면적: {area:.0f}m²")
    print(f"평균 밀도: {density:.1f} points/m²")

    # 작은 영역 테스트 (55.9m × 37.3m)
    test_x_center = 201240
    test_y_center = 523020
    test_x_min = test_x_center - 55.9/2
    test_x_max = test_x_center + 55.9/2
    test_y_min = test_y_center - 37.3/2
    test_y_max = test_y_center + 37.3/2

    mask = (
        (coords[:, 0] >= test_x_min) & (coords[:, 0] <= test_x_max) &
        (coords[:, 1] >= test_y_min) & (coords[:, 1] <= test_y_max)
    )

    test_points = coords[mask]
    test_area = 55.9 * 37.3
    print(f"\n테스트 영역 (55.9×37.3m = {test_area:.0f}m²):")
    print(f"포인트 수: {len(test_points):,}")
    print(f"밀도: {len(test_points)/test_area:.1f} points/m²")