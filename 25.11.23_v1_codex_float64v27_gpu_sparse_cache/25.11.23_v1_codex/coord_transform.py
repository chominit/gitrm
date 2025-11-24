# -*- coding: utf-8 -*-
"""
coord_transform.py
WGS84 → EPSG:5186 좌표 변환 (순수 Python 구현)
"""

import math
from typing import Tuple

# ==============================
# EPSG:5186 (Korea 2000 / Central Belt) 파라미터
# ==============================
# GRS80 타원체
A = 6378137.0  # 장반경 (m)
F = 1.0 / 298.257222101  # 편평률
B = A * (1 - F)  # 단반경
E2 = 2 * F - F * F  # 제1이심률 제곱

# TM 투영 파라미터
LAT0 = 38.0  # 원점 위도 (degrees)
LON0 = 127.0  # 중앙 자오선 (degrees)
K0 = 1.0  # 축척 계수
FALSE_EASTING = 200000.0  # 동쪽 가산값 (m)
FALSE_NORTHING = 600000.0  # 북쪽 가산값 (m)


def wgs84_to_epsg5186(lat: float, lon: float) -> Tuple[float, float]:
    """
    WGS84 (위도, 경도) → EPSG:5186 (X, Y) 변환

    Transverse Mercator 투영 공식 사용

    Args:
        lat: 위도 (degrees)
        lon: 경도 (degrees)

    Returns:
        (X, Y): EPSG:5186 좌표 (meters)
    """
    # 라디안 변환
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lat0_rad = math.radians(LAT0)
    lon0_rad = math.radians(LON0)

    # 경도차
    dlon = lon_rad - lon0_rad

    # 위도 함수 계산
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    tan_lat = math.tan(lat_rad)

    # 곡률 반경
    n = A / math.sqrt(1 - E2 * sin_lat * sin_lat)

    # 자오선 호장
    e4 = E2 * E2
    e6 = e4 * E2

    A0 = 1 - E2 / 4 - 3 * e4 / 64 - 5 * e6 / 256
    A2 = 3 * (E2 + e4 / 4 + 15 * e6 / 128) / 8
    A4 = 15 * (e4 + 3 * e6 / 4) / 256
    A6 = 35 * e6 / 3072

    M = A * (
        A0 * lat_rad
        - A2 * math.sin(2 * lat_rad)
        + A4 * math.sin(4 * lat_rad)
        - A6 * math.sin(6 * lat_rad)
    )

    M0 = A * (
        A0 * lat0_rad
        - A2 * math.sin(2 * lat0_rad)
        + A4 * math.sin(4 * lat0_rad)
        - A6 * math.sin(6 * lat0_rad)
    )

    # TM 좌표 계산
    T = tan_lat * tan_lat
    C = E2 * cos_lat * cos_lat / (1 - E2)
    A_val = dlon * cos_lat

    # X (Easting)
    X = K0 * n * (
        A_val
        + (1 - T + C) * A_val**3 / 6
        + (5 - 18 * T + T**2 + 72 * C - 58 * E2 / (1 - E2)) * A_val**5 / 120
    ) + FALSE_EASTING

    # Y (Northing)
    Y = K0 * (
        M - M0
        + n * tan_lat * (
            A_val**2 / 2
            + (5 - T + 9 * C + 4 * C**2) * A_val**4 / 24
            + (61 - 58 * T + T**2 + 600 * C - 330 * E2 / (1 - E2)) * A_val**6 / 720
        )
    ) + FALSE_NORTHING

    return (X, Y)


# ==============================
# 테스트
# ==============================
if __name__ == "__main__":
    # 테스트: 서울 부근
    test_lat = 37.3064416388889
    test_lon = 127.013981

    X, Y = wgs84_to_epsg5186(test_lat, test_lon)

    print(f"WGS84 → EPSG:5186 변환 테스트")
    print(f"입력: Lat={test_lat:.6f}, Lon={test_lon:.6f}")
    print(f"출력: X={X:.2f}, Y={Y:.2f}")
