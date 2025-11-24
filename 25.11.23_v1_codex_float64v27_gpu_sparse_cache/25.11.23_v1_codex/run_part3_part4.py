# -*- coding: utf-8 -*-
"""
run_part3_part4.py
part2 산출물(NPY)을 이용해 Forward(Part3) + Backward(Part4) 수행
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import cupy as cp
from tqdm import tqdm
import laspy

import constants as C
from camera_io import load_camera_db
from camera_calibration import get_camera_matrix, get_image_size, compute_camera_footprint_bbox
from img_mask import img_mask_diff
from gsd_parser import get_tolerance
from geometry import r_from_opk


# ----------------------------
# Part2 로드/저장 유틸
# ----------------------------
def load_part2(site: str) -> tuple[np.ndarray, np.ndarray]:
    """part2에서 생성된 NPY 로드"""
    d = C.PART2_DIR / site
    coords = np.load(d / "coords_float64.npy")
    colors = np.load(d / "colors_uint8.npy")
    return coords, colors


def save_las(coords, colors, votes, threshold, path: Path):
    """LAS 저장"""
    mask = votes >= threshold
    if not mask.any():
        return
    c = coords[mask]
    col = colors[mask]
    v = votes[mask]

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(c, axis=0)
    header.scales = [0.001, 0.001, 0.001]

    las = laspy.LasData(header)
    las.x, las.y, las.z = c[:, 0], c[:, 1], c[:, 2]
    las.red = (col[:, 0] * 257).astype(np.uint16)
    las.green = (col[:, 1] * 257).astype(np.uint16)
    las.blue = (col[:, 2] * 257).astype(np.uint16)
    las.point_source_id = np.clip(v, 0, 65535).astype(np.uint16)
    las.write(str(path))


# ----------------------------
# Ray Casting (단일 hit)
# ----------------------------
def process_rays_gpu(rays_gpu, C_cam_gpu, pts_gpu, h_tol, v_tol):
    """ray_batch × point_batch 이중 배치, ray당 최소 hit 1개"""
    N = len(rays_gpu)
    M = len(pts_gpu)
    if M == 0:
        return cp.full(N, -1, dtype=cp.int32)

    MAX_RAY_BATCH = 50000
    MAX_POINT_BATCH = int(os.environ.get("MAX_POINTS_PER_BATCH", "2000000"))

    best_idx = cp.full(N, -1, dtype=cp.int32)
    best_d = cp.full(N, cp.inf, dtype=cp.float64)

    for rs in range(0, N, MAX_RAY_BATCH):
        re = min(rs + MAX_RAY_BATCH, N)
        rb = rays_gpu[rs:re]

        local_idx = cp.full(len(rb), -1, dtype=cp.int32)
        local_d = cp.full(len(rb), cp.inf, dtype=cp.float64)

        for ps in range(0, M, MAX_POINT_BATCH):
            pe = min(ps + MAX_POINT_BATCH, M)
            p_batch = pts_gpu[ps:pe]

            P = p_batch - C_cam_gpu
            proj = cp.dot(rb, P.T)
            valid = proj > 0.1

            proj_pts = C_cam_gpu + rb[:, None, :] * proj[:, :, None]
            res = p_batch[None, :, :] - proj_pts
            h = cp.linalg.norm(res[:, :, :2], axis=2)
            v = cp.abs(res[:, :, 2])
            hit = valid & (h <= h_tol) & (v <= v_tol)

            for i in range(len(rb)):
                idx = cp.where(hit[i])[0]
                if len(idx) == 0:
                    continue
                d = proj[i, idx]
                m = cp.argmin(d)
                if d[m] < local_d[i]:
                    local_d[i] = d[m]
                    local_idx[i] = ps + idx[m]

        best_d[rs:re] = local_d
        best_idx[rs:re] = local_idx

    return best_idx


# ----------------------------
# Forward
# ----------------------------
def process_image_sparse(mask, K_inv, R_c2w_gpu, C_cam_gpu,
                         pts_gpu, h_tol, v_tol, fp_idx_global, sub_idx, votes):
    """희박 마스크: 마스크 좌표만 한번에 레이 캐스팅"""
    y_idx, x_idx = np.nonzero(mask)
    if len(x_idx) == 0:
        return 0
    pixels = np.stack([x_idx, y_idx, np.ones(len(x_idx))], axis=1).astype(np.float64)
    rays_cam = (K_inv @ pixels.T).T
    rays_world = (R_c2w_gpu @ rays_cam.T).T
    rays_world /= np.linalg.norm(rays_world, axis=1, keepdims=True)
    up = rays_world[:, 2] > 0
    if np.any(up):
        rays_world[up] *= -1

    max_ray_batch = 50000
    total = 0
    for rs in range(0, len(rays_world), max_ray_batch):
        re = min(rs + max_ray_batch, len(rays_world))
        rays_gpu = cp.asarray(rays_world[rs:re], dtype=cp.float64)
        hits = process_rays_gpu(rays_gpu, C_cam_gpu, pts_gpu, h_tol, v_tol)
        hits_cpu = cp.asnumpy(hits)
        valid = hits_cpu >= 0
        if np.any(valid):
            global_idx = fp_idx_global[sub_idx[hits_cpu[valid]]]
            np.add.at(votes, global_idx, 1)
        total += (re - rs)
    return total


def forward_site(site):
    """사이트별 Forward"""
    coords, colors = load_part2(site)
    cam_db = load_camera_db(site)
    K = get_camera_matrix(site)
    K_inv = np.linalg.inv(K)
    h_tol, v_tol = get_tolerance(site, use_cached=True)
    IMG_W, IMG_H = get_image_size(site)
    det_dir = C.DETECTION_DIRS[site]
    out_dir = C.PART3_DIR / site
    cache_dir = out_dir / "cache_pts"
    cache_dir.mkdir(parents=True, exist_ok=True)

    votes = np.zeros(len(coords), dtype=np.int32)
    images = cam_db["images"]
    margin = float(os.environ.get("FP_MARGIN_M", "0"))
    sparse_thresh = float(os.environ.get("SPARSE_THRESH", "0.10"))

    # 사이트 전체 footprint로 1차 필터
    all_cams = np.array([c["C"] for c in images.values()])
    fp_mask = (
        (coords[:, 0] >= all_cams[:, 0].min() - margin) &
        (coords[:, 0] <= all_cams[:, 0].max() + margin) &
        (coords[:, 1] >= all_cams[:, 1].min() - margin) &
        (coords[:, 1] <= all_cams[:, 1].max() + margin)
    )
    fp_idx_global = np.where(fp_mask)[0]
    fp_coords = coords[fp_idx_global]

    for img_name in tqdm(images.keys(), desc=f"[FORWARD] {site}"):
        mask = img_mask_diff(str(Path(det_dir) / img_name))
        if mask is None or not mask.any():
            continue

        C_cam = np.array(images[img_name]['C'], dtype=np.float64)
        R_wc = r_from_opk(images[img_name]['omega'], images[img_name]['phi'], images[img_name]['kappa'])
        R_c2w = R_wc.T

        # 이미지별 포인트 subset 캐시
        cache_file = cache_dir / f"{img_name}_idx.npy"
        if cache_file.exists():
            sub_idx = np.load(cache_file)
        else:
            x_min, x_max, y_min, y_max = compute_camera_footprint_bbox(C_cam, site, margin=C.FOOTPRINT_MARGIN)
            mask_xy = (
                (fp_coords[:, 0] >= x_min) & (fp_coords[:, 0] <= x_max) &
                (fp_coords[:, 1] >= y_min) & (fp_coords[:, 1] <= y_max)
            )
            sub_idx = np.where(mask_xy)[0].astype(np.int32)
            np.save(cache_file, sub_idx)
        if len(sub_idx) == 0:
            continue

        pts_gpu = cp.asarray(fp_coords[sub_idx], dtype=cp.float64)
        C_cam_gpu = cp.asarray(C_cam, dtype=cp.float64)
        R_c2w_gpu = cp.asarray(R_c2w, dtype=cp.float64)

        density = mask.mean()
        if density <= sparse_thresh:
            # sparse 모드
            _ = process_image_sparse(mask, K_inv, R_c2w_gpu, C_cam_gpu,
                                     pts_gpu, h_tol, v_tol,
                                     fp_idx_global, sub_idx, votes)
        else:
            # row 모드
            row_stride = int(os.environ.get("ROW_STRIDE", "4"))
            show_row_progress = os.environ.get("TQDM_ROWS", "0") != "0"
            row_iter = range(0, IMG_H, row_stride)
            if show_row_progress:
                row_iter = tqdm(row_iter, desc=f"      Rows {img_name}", leave=False)

            for row in row_iter:
                row_mask = mask[row, :]
                if not row_mask.any():
                    continue

                x_idx = np.where(row_mask)[0]
                pixels = np.stack([x_idx, np.full(len(x_idx), row), np.ones(len(x_idx))], axis=1)
                rays_cam = (K_inv @ pixels.T).T
                rays_world = (R_c2w @ rays_cam.T).T
                rays_world = rays_world / np.linalg.norm(rays_world, axis=1, keepdims=True)
                up = rays_world[:, 2] > 0
                if np.any(up):
                    rays_world[up] *= -1

                rays_gpu = cp.asarray(rays_world, dtype=cp.float64)
                hits = process_rays_gpu(rays_gpu, C_cam_gpu, pts_gpu, h_tol, v_tol)
                hits_cpu = cp.asnumpy(hits)
                valid = hits_cpu >= 0
                if np.any(valid):
                    global_idx = fp_idx_global[sub_idx[hits_cpu[valid]]]
                    np.add.at(votes, global_idx, 1)

    return votes


# ----------------------------
# Backward
# ----------------------------
def backward_site(site, coords, colors, forward_votes, threshold=7):
    cam_db = load_camera_db(site)
    det_dir = C.DETECTION_DIRS[site]
    K = get_camera_matrix(site)
    images = cam_db["images"]

    mask_forward = forward_votes >= threshold
    valid_idx = np.where(mask_forward)[0]
    if len(valid_idx) == 0:
        return np.zeros(len(coords), dtype=np.int32)

    valid_coords = coords[valid_idx]
    K_gpu = cp.asarray(K, dtype=cp.float64)
    backward_votes = np.zeros(len(coords), dtype=np.int32)

    for img_name in tqdm(list(images.keys())[:100], desc=f"[BACKWARD] {site}"):
        mask = img_mask_diff(str(Path(det_dir) / img_name))
        if mask is None or not mask.any():
            continue

        C_cam = np.array(images[img_name]['C'], dtype=np.float64)
        R_wc = r_from_opk(images[img_name]['omega'], images[img_name]['phi'], images[img_name]['kappa'])

        C_cam_gpu = cp.asarray(C_cam, dtype=cp.float64)
        R_wc_gpu = cp.asarray(R_wc, dtype=cp.float64)

        P_world = cp.asarray(valid_coords, dtype=cp.float64) - C_cam_gpu
        P_cam = (R_wc_gpu @ P_world.T).T

        front = P_cam[:, 2] > 0.1
        if not front.any():
            continue

        P_cam = P_cam[front]
        idx_front = valid_idx[front]

        P_img = (K_gpu @ P_cam.T).T
        P_img[:, 0] /= P_img[:, 2]
        P_img[:, 1] /= P_img[:, 2]

        H, W = mask.shape
        x = P_img[:, 0]
        y = P_img[:, 1]
        in_img = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if not in_img.any():
            continue

        x_i = x[in_img].astype(np.int32)
        y_i = y[in_img].astype(np.int32)
        idx_i = idx_front[in_img]

        for xi, yi, gi in zip(x_i, y_i, idx_i):
            if mask[yi, xi]:
                backward_votes[gi] += 1

    return backward_votes


# ----------------------------
# 메인
# ----------------------------
def run_site(site):
    coords, colors = load_part2(site)

    # Forward
    fwd = forward_site(site)
    out_dir = C.PART3_DIR / site
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "forward_votes.npy", fwd)

    # Backward
    bwd = backward_site(site, coords, colors, fwd, threshold=C.DEFAULT_VOTE_THRESHOLD)
    np.save(out_dir / "backward_votes.npy", bwd)

    # Final
    final = fwd + bwd
    np.save(out_dir / "final_votes.npy", final)
    for th in C.VOTE_THRESHOLDS:
        save_las(coords, colors, final, th, out_dir / f"vote_{th}.las")


def main():
    sites = C.SITE_WHITELIST
    for s in sites:
        run_site(s)


if __name__ == "__main__":
    main()
