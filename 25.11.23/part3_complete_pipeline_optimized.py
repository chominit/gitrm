# -*- coding: utf-8 -*-
"""part3_complete_pipeline_optimized.py

대용량 사이트(Site A/B/C 처럼 1e8~1e9 포인트)를 위한 Forward 전용 최적화 버전.
- coords 전체를 GPU에 올리지 않고,
- Scanline + Row chunk + footprint 기반 band 필터 + GPU 부분 배치.
기존 part3_complete_pipeline.py를 보존하고, 이 파일은 별도 엔트리포인트로 사용한다.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
from tqdm import tqdm

import constants as C
from camera_io import load_camera_db
from camera_calibration import get_camera_matrix
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

    images = list(cam_db['images'].keys())
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
