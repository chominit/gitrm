# -*- coding: utf-8 -*-
"""part3_complete_pipeline_optimized.py

대용량 사이트(Site A/B/C 처럼 1e8~1e9 포인트)를 위한 Forward 전용 최적화 버전.
- coords 전체를 GPU에 올리지 않고,
- Scanline + Row chunk + footprint 기반 band 필터 + GPU 부분 배치.
기존 part3_complete_pipeline.py를 보존하고, 이 파일은 별도 엔트리포인트로 사용한다.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
from tqdm import tqdm

import constants as C
from camera_io import load_camera_db
from camera_calibration import get_camera_matrix, compute_camera_footprint_bbox
from geometry import r_from_opk
from gsd_parser import get_tolerance
from scanline_engine import process_image_chunked
from las_utils import load_point_cloud, save_las


try:
    import cupy as cp  # type: ignore
    GPU_AVAILABLE = True
    print("[INFO] CuPy detected. GPU acceleration enabled.")
except Exception:
    GPU_AVAILABLE = False
    print("[ERROR] CuPy is required for this optimized pipeline.")
    sys.exit(1)


def _get_iop(site_name: str):
    """Return intrinsic parameters dict for the site (fx, fy, width, height)."""
    try:
        from camera_calibration import SITE_CALIBRATION, ZENMUSE_IOP
        if site_name in SITE_CALIBRATION:
            return SITE_CALIBRATION[site_name]
        return ZENMUSE_IOP
    except Exception:
        # Fallback to Zenmuse P1 35mm-like defaults
        return {"fx": 12000.0, "fy": 12000.0, "width": 8192, "height": 5460}


def _camera_forward(cam_info: dict) -> np.ndarray:
    """카메라 광축(세계 좌표) 계산. z>0이면 뒤집어 지면(-z) 보도록 맞춤."""
    R_wc = r_from_opk(cam_info['omega'], cam_info['phi'], cam_info['kappa'])
    R_c2w = R_wc.T
    fwd = R_c2w @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if fwd[2] > 0:
        fwd *= -1.0
    n = np.linalg.norm(fwd)
    if n > 0:
        fwd /= n
    return fwd


def filter_images_by_fp_and_view(cam_db: dict,
                                 site_name: str,
                                 fp_coords: np.ndarray,
                                 fov_margin_deg: float = 0.0,
                                 view_cos_min: float = 0.0) -> list[str]:
    """Footprint bbox 겹침 + FoV 기반 시야각 필터.

    - fov_margin_deg: FoV 여유각(도) 수평/수직 모두 적용. 0이면 센서 FoV 그대로.
    - view_cos_min: 추가로 광축·footprint 중심 cos 조건을 줄 때 사용(0이면 생략).
    """
    if len(fp_coords) == 0:
        return list(cam_db['images'].keys())

    x_min, x_max = fp_coords[:, 0].min(), fp_coords[:, 0].max()
    y_min, y_max = fp_coords[:, 1].min(), fp_coords[:, 1].max()
    fp_center = np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5, fp_coords[:, 2].mean()])

    iop = _get_iop(site_name)
    fx, fy = float(iop['fx']), float(iop['fy'])
    width, height = float(iop['width']), float(iop['height'])
    hfov = 2 * np.arctan(width / (2 * fx))  # radians
    vfov = 2 * np.arctan(height / (2 * fy))
    margin_rad = np.deg2rad(fov_margin_deg)
    hfov_half = hfov * 0.5 + margin_rad
    vfov_half = vfov * 0.5 + margin_rad
    tan_h = np.tan(hfov_half)
    tan_v = np.tan(vfov_half)

    kept: list[str] = []
    for name, cam in cam_db['images'].items():
        # 1) 지상 footprint bbox 겹침 여부
        bx_min, bx_max, by_min, by_max = compute_camera_footprint_bbox(
            np.array(cam['C']), site_name, margin=1.0
        )
        overlap = not (bx_max < x_min or bx_min > x_max or by_max < y_min or by_min > y_max)
        if not overlap:
            continue

        # 2) FoV 체크: footprint 중심이 카메라 FoV 안인지
        R_wc = r_from_opk(cam['omega'], cam['phi'], cam['kappa'])
        to_fp_world = fp_center - np.array(cam['C'], dtype=np.float64)
        v_cam = R_wc @ to_fp_world  # world -> camera frame
        if v_cam[2] <= 0:
            continue  # 뒤쪽이면 스킵
        tan_x = abs(v_cam[0] / v_cam[2])
        tan_y = abs(v_cam[1] / v_cam[2])
        if tan_x > tan_h or tan_y > tan_v:
            continue

        # 3) 추가 cos 조건 (옵션)
        if view_cos_min > 0:
            fwd = _camera_forward(cam)
            to_fp = to_fp_world / np.linalg.norm(to_fp_world)
            cos_ang = float(np.dot(fwd, to_fp))
            if cos_ang < view_cos_min:
                continue

        kept.append(name)

    return kept


def run_forward_pass(site_name: str,
                     coords: np.ndarray,
                     colors: np.ndarray,
                     device_id: int = 0) -> np.ndarray:
    """Forward Pass 실행 (Scanline + Chunked, footprint 기반)."""
    print(f"\n[Forward Pass] {site_name}")

    det_dir = C.DETECTION_DIRS.get(site_name)
    if not det_dir or not det_dir.exists():
        print("   [SKIP] Detection directory not found.")
        return np.zeros(len(coords), dtype=np.int32)

    cam_db = load_camera_db(site_name)
    K = get_camera_matrix(site_name)
    h_tol, v_tol = get_tolerance(site_name)

    # Ground Z 근사 (카메라 평균 고도 - 사이트 비행고도)
    cam_zs = [cam_info['C'][2] for cam_info in cam_db['images'].values()]
    Z_avg = float(np.mean(cam_zs))
    H_site = C.FLIGHT_ALT_BY_SITE.get(site_name, 80.0)
    ground_Z = Z_avg - H_site
    print(f"   Z_avg: {Z_avg:.2f}m, H_site: {H_site:.1f}m, ground_Z ≈ {ground_Z:.2f}m")


    # 1차 Footprint 필터 (카메라 XY 범위 ± 여유 마진)
    cams_xy = np.array([[c['C'][0], c['C'][1]] for c in cam_db['images'].values()], dtype=np.float64)
    margin_xy = 100.0  # meters
    fp_x_min = cams_xy[:, 0].min() - margin_xy
    fp_x_max = cams_xy[:, 0].max() + margin_xy
    fp_y_min = cams_xy[:, 1].min() - margin_xy
    fp_y_max = cams_xy[:, 1].max() + margin_xy

    print("   Filtering footprint points (CPU)...", flush=True)
    mask_fp = (
        (coords[:, 0] >= fp_x_min) & (coords[:, 0] <= fp_x_max) &
        (coords[:, 1] >= fp_y_min) & (coords[:, 1] <= fp_y_max)
    )
    fp_indices = np.where(mask_fp)[0]
    fp_coords = coords[fp_indices]

    print(f"   Footprint points: {len(fp_coords):,}")
    if len(fp_coords) == 0:
        print("   [WARN] No points in footprint. Forward pass skipped.")
        return np.zeros(len(coords), dtype=np.int32)

    print(f"   Point Z range in footprint: [{fp_coords[:, 2].min():.2f}, {fp_coords[:, 2].max():.2f}]" )

    # Vote 배열 (원본 coords 크기 기준)
    votes = np.zeros(len(coords), dtype=np.int32)

    view_cos_min = float(os.environ.get("VIEW_COS_MIN", "0.0"))
    fov_margin_deg = float(os.environ.get("FOV_MARGIN_DEG", "0.0"))

    images_all = list(cam_db['images'].keys())
    images = filter_images_by_fp_and_view(
        cam_db, site_name, fp_coords,
        fov_margin_deg=fov_margin_deg,
        view_cos_min=view_cos_min,
    )
    print(
        f"   Images filtered: {len(images)}/{len(images_all)} "
        f"(FoV margin {fov_margin_deg} deg, cos>={view_cos_min})"
    )

    total_rays = 0

    for img_name in tqdm(images, desc="Processing Images"):
        cam_info = cam_db['images'][img_name]
        rays_count = process_image_chunked(
            img_name=img_name,
            cam_info=cam_info,
            fp_coords=fp_coords,
            fp_indices=fp_indices,
            det_dir=det_dir,
            ground_Z=ground_Z,
            K=K,
            h_tol=h_tol,
            v_tol=v_tol,
            votes=votes,
            chunk_size=32,
            device_id=device_id,
        )
        total_rays += rays_count

    print(f"   Total rays processed: {total_rays:,}")
    print(f"   Total votes: {votes.sum():,}")
    return votes


def process_site(site_name: str,
                 device_id: int = 0) -> None:
    """단일 사이트 처리 (Forward only + LAS 저장)."""
    print(f"\n==============================")
    print(f"Processing site: {site_name}")
    print(f"==============================")


    las_dir = C.PART2_DIR / site_name
    output_dir = C.PART3_DIR / site_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 포인트 클라우드 로드 (coords, colors)
    try:
        coords, colors = load_point_cloud(las_dir)
    except Exception as e:
        print(f"[ERROR] Failed to load point cloud for {site_name}: {e}")
        return

    print(f"   Points: {len(coords):,}")

    # Forward pass
    t0 = time.time()
    forward_votes = run_forward_pass(site_name, coords, colors, device_id=device_id)
    t1 = time.time()
    print(f"[TIME] Forward pass: {t1 - t0:.1f}s")


    # Forward 결과 저장
    np.save(output_dir / "forward_votes_opt.npy", forward_votes)

    # Threshold 7 기준으로 LAS 저장
    thresh = 7
    valid_mask = forward_votes >= thresh
    if not np.any(valid_mask):
        print(f"   [WARN] No points above threshold {thresh}.")
        return

    coords_sel = coords[valid_mask]
    colors_sel = colors[valid_mask]

    las_path = output_dir / f"vote_opt_{thresh}.las"
    save_las(coords_sel, colors_sel, las_path)
    print(f"   Saved LAS: {las_path}")


def main() -> None:
    if len(sys.argv) > 1:
        site = sys.argv[1]
        process_site(site)
    else:
        # 기본 테스트용
        default_site = "Zenmuse_AI_Site_A"
        process_site(default_site)


if __name__ == "__main__":
    main()
