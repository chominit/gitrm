"""
Step 2 : RANSAC 후처리 전용 스크립트
 - Original Geometry   → RANSAC
 - Detected Color      → RANSAC
 - Detected Color+Geo  → RANSAC
author : 2024‑06‑23
"""

import os, sys
import numpy as np
import laspy
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ──────────────────────────────────────
# ① 경로 설정  (필요 시 수정)
# ──────────────────────────────────────
orig_geo_dir      = r"G:\UAV_RANSAC\1.Original_RANSAC\1.Geo"          # Original Geometry 결과
det_color_dir     = r"G:\UAV_RANSAC\2.Detected_RANSAC\1.Color"        # Detected Color 결과
det_geo_dir       = r"G:\UAV_RANSAC\2.Detected_RANSAC\2.Geo"          # Detected Color+Geometry 결과

out_orig_ransac        = r"G:\UAV_RANSAC\1.Original_RANSAC\2.RANSAC_Finish"
out_det_color_ransac   = r"G:\UAV_RANSAC\2.Detected_RANSAC\3.RANSAC_Finish_Color"
out_det_geo_ransac     = r"G:\UAV_RANSAC\2.Detected_RANSAC\4.RANSAC_Finish_Geo"

for d in [out_orig_ransac, out_det_color_ransac, out_det_geo_ransac]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────
# ② 공통 유틸
# ──────────────────────────────────────
MIN_FILE_SIZE = 100 * 1024          # 100 kB 미만은 ‘빈 파일’ 간주
THREADS       = os.cpu_count() or 32

log_entries: list[tuple] = []
log_lock = threading.Lock()

def log_step(fname, pipe, step, t0, t1):
    dur = (t1 - t0).total_seconds()
    with log_lock:
        log_entries.append(
            (fname, pipe, step,
             t0.strftime("%Y-%m-%d %H:%M:%S.%f UTC"),
             t1.strftime("%Y-%m-%d %H:%M:%S.%f UTC"),
             f"{dur:.6f}")
        )

def delete_small_files(directory: str, min_bytes: int = MIN_FILE_SIZE):
    for f in os.listdir(directory):
        if f.lower().endswith(".las"):
            p = os.path.join(directory, f)
            try:
                if os.path.getsize(p) < min_bytes:
                    os.remove(p)
                    print(f"[Info] 작은 불완전 파일 삭제: {f}")
            except OSError:
                pass

# ──────────────────────────────────────
# ③ RANSAC  &  LAS 저장 함수
# ──────────────────────────────────────
def extract_powerlines(coords: np.ndarray,
                       threshold=0.05,
                       min_inliers=30,
                       max_iter=1000) -> np.ndarray:
    """단순 선 RANSAC – 여러 라인을 반복 검출"""
    if coords.shape[0] < 2:
        return np.empty((0, 3))

    remain = coords.copy()
    out = []
    rng = np.random.default_rng()

    while remain.shape[0] >= min_inliers:
        best_idx, best_n = None, 0
        for _ in range(max_iter):
            i1, i2 = rng.choice(remain.shape[0], 2, replace=False)
            p1, p2 = remain[i1], remain[i2]
            v = p2 - p1
            nrm = np.linalg.norm(v)
            if nrm < 1e-6:
                continue
            d = np.linalg.norm(np.cross((remain - p1), v / nrm), axis=1)
            idx = np.where(d <= threshold)[0]
            if idx.size > best_n:
                best_idx, best_n = idx, idx.size
        if best_idx is None or best_n < min_inliers:
            break
        out.append(remain[best_idx])
        remain = np.delete(remain, best_idx, axis=0)

    return np.vstack(out) if out else np.empty((0, 3))

def save_las_points(template: laspy.LasData,
                    coords: np.ndarray,
                    colors: np.ndarray | None,
                    out_path: str):
    las_out = laspy.LasData(template.header)

    if coords.size:
        las_out.x, las_out.y, las_out.z = coords.T
        if colors is not None and {'red','green','blue'} <= set(template.point_format.dimension_names):
            col = colors.astype(np.float64)
            if col.max() <= 255:                        # 8‑bit  →  16‑bit
                col *= 65535.0 / 255.0
            col = np.clip(col, 0, 65535).astype(np.uint16)
            las_out.red, las_out.green, las_out.blue = col.T

        las_out.header.min = coords.min(axis=0)
        las_out.header.max = coords.max(axis=0)
        las_out.header.point_count = coords.shape[0]

    las_out.write(out_path)

# ──────────────────────────────────────
# ④   처리 루틴 (Original / Detected)
# ──────────────────────────────────────
def process_original_ransac(fname: str):
    src = os.path.join(orig_geo_dir, fname)
    dst = os.path.join(out_orig_ransac, fname)
    print(f"\n[Original‑RANSAC] {fname}")

    if os.path.exists(dst) and os.path.getsize(dst) >= MIN_FILE_SIZE:
        print("  - 완료 파일 존재, 건너뜀.")
        return

    try:
        las = laspy.read(src)
        xyz = np.vstack((las.x, las.y, las.z)).T
        rgb = None
        try:
            rgb = np.vstack((las.red, las.green, las.blue)).T
        except AttributeError:
            pass

        t0 = datetime.now(timezone.utc)
        inliers = extract_powerlines(xyz)
        t1 = datetime.now(timezone.utc)
        log_step(fname, "Original", "RANSAC", t0, t1)

        if inliers.size and rgb is not None:
            sel = np.isin(xyz.view([('',xyz.dtype)]*3),
                          inliers.view([('',inliers.dtype)]*3))
            colors = rgb[sel]
        else:
            colors = None

        save_las_points(las, inliers, colors, dst)
        if os.path.getsize(dst) < MIN_FILE_SIZE:
            os.remove(dst);  print("  → 선 검출 0, 결과 삭제.")
        else:
            print("  · 완료")

    except Exception as e:
        print(f"[오류] Original‑RANSAC 실패: {e}")

def _detect_ransac_common(fname: str,
                          in_dir: str,
                          out_dir: str,
                          tag: str):
    src = os.path.join(in_dir, fname)
    dst = os.path.join(out_dir, fname)
    print(f"\n[{tag}] {fname}")

    if os.path.exists(dst) and os.path.getsize(dst) >= MIN_FILE_SIZE:
        print("  - 완료 파일 존재, 건너뜀.")
        return

    try:
        las = laspy.read(src)
        xyz = np.vstack((las.x, las.y, las.z)).T
        rgb = None
        try:
            rgb = np.vstack((las.red, las.green, las.blue)).T
        except AttributeError:
            pass

        t0 = datetime.now(timezone.utc)
        inliers = extract_powerlines(xyz)
        t1 = datetime.now(timezone.utc)
        log_step(fname, tag, "RANSAC", t0, t1)

        if inliers.size and rgb is not None:
            sel = np.isin(xyz.view([('',xyz.dtype)]*3),
                          inliers.view([('',inliers.dtype)]*3))
            colors = rgb[sel]
        else:
            colors = None

        save_las_points(las, inliers, colors, dst)
        if os.path.getsize(dst) < MIN_FILE_SIZE:
            os.remove(dst);  print("  → 선 검출 0, 결과 삭제.")
        else:
            print("  · 완료")

    except Exception as e:
        print(f"[오류] {tag} 실패: {e}")

def process_detect_color_ransac(fname:str):
    _detect_ransac_common(fname, det_color_dir, out_det_color_ransac, "Detect‑Color")

def process_detect_geo_ransac(fname:str):
    _detect_ransac_common(fname, det_geo_dir,   out_det_geo_ransac,   "Detect‑Geo")

# ──────────────────────────────────────
# ⑤   실행부
# ──────────────────────────────────────
if __name__ == "__main__":
    print("=== Step 2 RANSAC 후처리 시작 ===")
    for d in [orig_geo_dir, det_color_dir, det_geo_dir]:
        delete_small_files(d)

    orig_files      = sorted(f for f in os.listdir(orig_geo_dir)    if f.lower().endswith(".las"))
    det_color_files = sorted(f for f in os.listdir(det_color_dir)   if f.lower().endswith(".las"))
    det_geo_files   = sorted(f for f in os.listdir(det_geo_dir)     if f.lower().endswith(".las"))

    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        fut = []
        fut += [ex.submit(process_original_ransac, f)      for f in orig_files]
        fut += [ex.submit(process_detect_color_ransac, f)  for f in det_color_files]
        fut += [ex.submit(process_detect_geo_ransac, f)    for f in det_geo_files]

        for f in as_completed(fut):
            f.result()

    # 로그 CSV
    try:
        import csv
        with open("postprocessing_log.csv", "w", newline='', encoding="utf-8") as fp:
            wr = csv.writer(fp)
            wr.writerow(["File","Pipeline","Step","StartUTC","EndUTC","DurationSec"])
            for row in sorted(log_entries, key=lambda x: x[0]):
                wr.writerow(row)
        print("[Info] 로그 저장 완료: postprocessing_log.csv")
    except Exception as e:
        print(f"[경고] 로그 저장 실패: {e}")

    print("=== Step 2 완료 ===")
