# -*- coding: utf-8 -*-
"""
gsd_parser.py
GSD(Ground Sample Distance) 파싱 모듈
- 우선 사이트별 캐시된 GSD 값을 사용
- 없으면 report.xml에서 파싱
- 그래도 없으면 기본값(C.GSD_H/C.GSD_V)
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict
from pathlib import Path
import re
import constants as C

# 캐시된 사이트별 GSD (m) - 2025-11-19 기준
SITE_GSD: Dict[str, float] = {
    "Zenmuse_AI_Site_A": 0.013964,
    "Zenmuse_AI_Site_B": 0.010127,
    "Zenmuse_AI_Site_C": 0.010494,
    "P4R_Site_A_Solid": 0.014314,
    "P4R_Site_B_Solid_Merge_V2": 0.010347,
    "P4R_Site_C_Solid_Merge_V2": 0.010497,
    "P4R_Zenmuse_Joint_AI_Site_A": 0.014282,
    "P4R_Zenmuse_Joint_AI_Site_B": 0.010152,
    "P4R_Zenmuse_Joint_AI_Site_C": 0.010489,
}


def _parse_gsd_from_report(report_path: Path) -> Optional[float]:
    """report.xml에서 GSD(m)를 추출. 실패 시 None."""
    if not report_path or not report_path.exists():
        return None

    try:
        text = report_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            text = report_path.read_text(errors="ignore")
        except Exception:
            return None

    patterns = [
        r"ground[_\s-]?resolution[^0-9]*([\d\.]+)\s*(mm|cm|m)",
        r"gsd[^0-9]*([\d\.]+)\s*(mm|cm|m)",
        r"GSD[^0-9]*([\d\.]+)\s*(mm|cm|m)",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            val = float(m.group(1))
            unit = m.group(2).lower()
            if unit == "mm":
                return val / 1000.0
            if unit == "cm":
                return val / 100.0
            return val
    return None


def get_tolerance(site_name: str, use_cached: bool = True) -> Tuple[float, float]:
    """
    Returns:
        (h_tol, v_tol): 수평/수직 허용오차 (미터)
    """
    gsd_m = None

    # 1) 캐시된 GSD 사용
    if use_cached and site_name in SITE_GSD:
        gsd_m = SITE_GSD[site_name]

    # 2) report.xml 파싱
    if gsd_m is None:
        report_path = C.SITE_REPORT_PATHS.get(site_name)
        gsd_m = _parse_gsd_from_report(report_path)

    # 3) 실패 시 기본값
    if gsd_m is None:
        gsd_m = C.GSD_H

    return (1.0 * gsd_m, 3.0 * gsd_m)
