# -*- coding: utf-8 -*-
"""
gsd_parser.py
GSD 기반 허용오차 계산
"""

from typing import Tuple
import constants as C

def get_tolerance(site_name: str) -> Tuple[float, float]:
    """
    GSD 기반 허용오차

    Returns:
        (h_tol, v_tol): 수평/수직 허용오차 (미터)
    """
    # 사이트별 GSD (현재는 전역값 사용)
    gsd = C.GSD_H

    return (
        1.0 * gsd,  # 수평: 1 × GSD = 0.01m
        3.0 * gsd   # 수직: 3 × GSD = 0.03m
    )
