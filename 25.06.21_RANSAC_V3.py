import os, sys
import numpy as np
import laspy
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Input directories
original_dir = r"C:\Users\jscool\datasets\Pix4d\Zenmuse Site C Original"
detect_dir   = r"C:\Users\jscool\datasets\Pix4d\Zenmuse Site C Detect\2_densification\point_cloud"

# Output directories
out_orig_geo    = r"G:\UAV_RANSAC\1.Original_RANSAC\1.Geo"
out_orig_ransac = r"G:\UAV_RANSAC\1.Original_RANSAC\2.RANSAC_Finish"
out_det_rgb     = r"G:\UAV_RANSAC\2.Detected_RANSAC\1.RGB"
out_det_geo     = r"G:\UAV_RANSAC\2.Detected_RANSAC\2.Geo"
out_det_ransac  = r"G:\UAV_RANSAC\2.Detected_RANSAC\3.RANSAC_Finish"

# Ensure output directories exist
for d in [out_orig_geo, out_orig_ransac, out_det_rgb, out_det_geo, out_det_ransac]:
    os.makedirs(d, exist_ok=True)

# Global list for log entries and a lock for thread-safety
log_entries = []
log_lock = threading.Lock()

def log_step(file_name, pipeline, step, t_start, t_end):
    """Record a log entry for a processing step with UTC timestamps and duration."""
    duration = (t_end - t_start).total_seconds()
    start_str = t_start.strftime("%Y-%m-%d %H:%M:%S.%f UTC")
    end_str   = t_end.strftime("%Y-%m-%d %H:%M:%S.%f UTC")
    entry = (file_name, pipeline, step, start_str, end_str, f"{duration:.6f}")
    with log_lock:
        log_entries.append(entry)

def delete_small_files(directory, min_bytes=102400):
    """Delete .las files smaller than min_bytes in the given directory (e.g., incomplete files)."""
    for fname in os.listdir(directory):
        if fname.lower().endswith(".las"):
            fpath = os.path.join(directory, fname)
            try:
                size = os.path.getsize(fpath)
            except OSError:
                continue
            if size < min_bytes:
                try:
                    os.remove(fpath)
                    print(f"[Info] Removed small incomplete file: {fname} ({size} bytes)")
                except OSError as e:
                    print(f"[Warning] Could not remove {fname}: {e}")

def filter_by_ground(coords: np.ndarray) -> np.ndarray:
    N = coords.shape[0]
    mask = np.zeros(N, dtype=bool)

    if N == 0:
        return mask  # 빈 배열인 경우 즉시 반환

    cell_size = 0.15
    height_threshold = 1.0

    gx = np.floor(coords[:, 0] / cell_size).astype(int)
    gy = np.floor(coords[:, 1] / cell_size).astype(int)

    ground_z = {}
    for x_idx, y_idx, z_val in zip(gx, gy, coords[:,2]):
        key = (x_idx, y_idx)
        ground_z[key] = min(ground_z.get(key, z_val), z_val)

    for idx in range(N):
        x_idx, y_idx, z_val = gx[idx], gy[idx], coords[idx, 2]
        neighbor_min_z = np.inf
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_key = (x_idx + dx, y_idx + dy)
                neighbor_z = ground_z.get(neighbor_key, np.inf)
                neighbor_min_z = min(neighbor_min_z, neighbor_z)
        if z_val >= neighbor_min_z + height_threshold:
            mask[idx] = True

    return mask

def filter_by_color(colors: np.ndarray):
    """Color filter: keep points whose color is within ±10% of (255,0,191) after 16→8 bit normalization:contentReference[oaicite:11]{index=11}:contentReference[oaicite:12]{index=12}."""
    if colors is None:
        return None  # no color data, skip filtering
    col = colors.astype(np.float64)
    # Normalize 16-bit color to 0-255 if needed
    if col.max() > 255.0:
        col = col * 255.0 / 65535.0
    R, G, B = col[:,0], col[:,1], col[:,2]
    mask = np.ones(R.shape, dtype=bool)
    # Apply 90%/10%/±10% thresholds for R, G, B respectively
    mask &= (R >= 229.5)        # R >= 230 (~90% of 255)
    mask &= (G <= 25.5)         # G <= 25 (~10% of 255)
    mask &= (B >= 172.0) & (B <= 210.0)  # B within [172, 210] (~±10% of 191)
    return mask

def extract_powerlines(coords: np.ndarray) -> np.ndarray:
    """Iterative RANSAC to extract linear features (lines) with 5cm threshold and min 30 points:contentReference[oaicite:13]{index=13}:contentReference[oaicite:14]{index=14}."""
    if coords.shape[0] < 2:
        return np.empty((0,3))
    pts = coords.copy()
    line_inliers = []
    threshold = 0.05  # 5 cm
    min_inliers = 30
    max_iterations = 1000
    rng = np.random.default_rng()
    # Iteratively find lines until not enough points remain
    while pts.shape[0] >= min_inliers:
        best_inliers_idx = None
        best_count = 0
        # RANSAC loop to find one line model
        for _ in range(max_iterations):
            # Randomly pick 2 distinct points to define a line
            idx_samples = rng.choice(pts.shape[0], size=2, replace=False)
            p1, p2 = pts[idx_samples[0]], pts[idx_samples[1]]
            if np.allclose(p1, p2):
                continue
            line_vec = p2 - p1
            norm = np.linalg.norm(line_vec)
            if norm < 1e-6:
                continue
            line_dir = line_vec / norm
            # Distance of all points to the infinite line
            vecs = pts - p1
            cross_prod = np.cross(line_dir, vecs)        # cross product
            dist = np.linalg.norm(cross_prod, axis=1)    # perpendicular distances
            inlier_idx = np.where(dist <= threshold)[0]
            count = inlier_idx.size
            if count > best_count:
                best_count = count
                best_inliers_idx = inlier_idx
            # (Optional early break if a very large inlier set is found)
        if best_inliers_idx is None or best_count < min_inliers:
            break  # no sufficient line found
        # Store inlier points for this line and remove them from consideration
        line_inliers.append(pts[best_inliers_idx])
        mask = np.ones(pts.shape[0], dtype=bool)
        mask[best_inliers_idx] = False
        pts = pts[mask]
    if not line_inliers:
        return np.empty((0,3))
    # Combine all line inliers from all detected lines
    return np.vstack(line_inliers)

def save_las_points(template_las, coords: np.ndarray, colors: np.ndarray, out_path: str):
    """Save an array of points (coords and optional colors) to a LAS file, using the template LAS header:contentReference[oaicite:15]{index=15}:contentReference[oaicite:16]{index=16}."""
    new_las = laspy.LasData(template_las.header)
    if coords.shape[0] == 0:
        # No points to save
        new_las.points = np.array([], dtype=template_las.points.dtype)
    else:
        # If color data is provided and template has RGB fields
        if colors is not None and 'red' in template_las.point_format.dimension_names:
            col = colors.astype(np.float64)
            # If color is in 0-255 range, scale up to 0-65535 for LAS storage
            if col.max() <= 255.0:
                col = col * 65535.0 / 255.0
            col = np.clip(col, 0, 65535).astype(np.uint16)
            # Assign coordinates and color to new LAS structure
            new_las.x = coords[:,0]
            new_las.y = coords[:,1]
            new_las.z = coords[:,2]
            new_las.red   = col[:,0]
            new_las.green = col[:,1]
            new_las.blue  = col[:,2]
        else:
            # No color case or no color fields in template
            new_las.x = coords[:,0]
            new_las.y = coords[:,1]
            new_las.z = coords[:,2]
    # Update header bounds and point count
    if coords.shape[0] > 0:
        new_las.header.min = np.min(coords, axis=0)
        new_las.header.max = np.max(coords, axis=0)
    new_las.header.point_count = coords.shape[0]
    # Write LAS (or LAZ) file to output path
    new_las.write(out_path)

def process_original_file(file_path: str):
    """Process one Original file: Geometry filter -> RANSAC (skipping done stages as needed)."""
    fname = os.path.basename(file_path)
    print(f"[Original] Processing file: {fname}")
    # Skip entire file if final RANSAC output already exists (unless it's an empty placeholder, then redo)
    final_path = os.path.join(out_orig_ransac, fname)
    if os.path.exists(final_path):
        if os.path.getsize(final_path) >= 100 * 1024:
            print(f"  - Skipping {fname}: final output already exists.")
            return
        else:
            # Remove small final file and proceed with processing
            try:
                os.remove(final_path)
                print(f"  - Removed incomplete final output for {fname}, will recompute RANSAC.")
            except OSError:
                pass
    # Determine if intermediate (geometry) output exists
    geo_path = os.path.join(out_orig_geo, fname)
    geo_exists = os.path.exists(geo_path)
    if geo_exists and os.path.getsize(geo_path) < 100 * 1024:
        # Remove empty or small geometry file and treat as not existing
        try:
            os.remove(geo_path)
            print(f"  - Removed incomplete geometry output for {fname}, will recompute geometry.")
        except OSError:
            pass
        geo_exists = False

    try:
        if geo_exists:
            # If geometry stage was done before, load the filtered points from file
            las_geom = laspy.read(geo_path)
            coords_geo = np.vstack((las_geom.x, las_geom.y, las_geom.z)).T
            # Load color from intermediate if present
            try:
                r, g, b = las_geom.red, las_geom.green, las_geom.blue
                colors_geo = np.vstack((r, g, b)).T  # (may be 16-bit)
            except AttributeError:
                colors_geo = None
            total_points = coords_geo.shape[0]
            print(f"  - Geometry stage skipped (loaded {total_points} points from previous output).")
        else:
            # Otherwise, load the original LAS and perform geometry filtering
            t0 = datetime.now(timezone.utc)
            las = laspy.read(file_path)
            coords = np.vstack((las.x.copy(), las.y.copy(), las.z.copy())).T
            total_points = coords.shape[0]
            print(f"  - Loaded {total_points} points from {fname}")
            # Geometry filter
            t1 = datetime.now(timezone.utc)
            mask_geo = filter_by_ground(coords)
            t2 = datetime.now(timezone.utc)
            coords_geo = coords[mask_geo]
            # Preserve corresponding color data (if any)
            try:
                r, g, b = las.red, las.green, las.blue
                color_arr = np.vstack((r, g, b)).T
                # Normalize color for filtering logic (not strictly needed for saving)
                if color_arr.max() > 255:
                    color_arr = (color_arr * 255.0 / 65535.0).astype(np.float64)
                colors_geo = color_arr[mask_geo] if mask_geo.any() else np.empty((0,3))
            except AttributeError:
                colors_geo = None
            t3 = datetime.now(timezone.utc)
            log_step(fname, "Original", "GeometryFilter", t1, t2)
            print(f"  - Geometry filter: {coords_geo.shape[0]} / {total_points} points remain after ground removal")
            # Save geometry-filtered output for possible reuse
            save_las_points(las, coords_geo, colors_geo, geo_path)
            # Remove the LAS object to free memory
            del las
        # Now perform RANSAC on the geometry-filtered points (if any)
        if coords_geo.shape[0] == 0:
            line_coords = np.empty((0,3))
        else:
            t4 = datetime.now(timezone.utc)
            line_coords = extract_powerlines(coords_geo)
            t5 = datetime.now(timezone.utc)
            log_step(fname, "Original", "RANSAC", t4, t5)
        line_count = line_coords.shape[0]
        print(f"  - RANSAC: found {line_count} line points out of {coords_geo.shape[0]} input points")
        # Save RANSAC result (only if any line points found, otherwise create an empty file)
        line_colors = None
        if line_count > 0:
            # If original had color data, find colors for the line points
            if colors_geo is not None and line_coords.shape[0] > 0:
                coord_set = {tuple(pt) for pt in line_coords}
                mask_line = np.array([tuple(pt) in coord_set for pt in coords_geo])
                line_colors = colors_geo[mask_line]
            else:
                line_colors = None
        # Always write out the RANSAC result (even if empty, for logging completeness)
        save_las_points(las_geom if geo_exists else laspy.LasData(las_geom.header) if 'las_geom' in locals() else laspy.read(file_path), 
                        line_coords, line_colors, final_path)
        # Clean up small output files if they contain no meaningful data
        for path in [geo_path, final_path]:
            if os.path.exists(path) and os.path.getsize(path) < 100 * 1024:
                try:
                    os.remove(path)
                    print(f"    -> Removed empty output file {os.path.basename(path)}")
                except OSError:
                    pass
    except Exception as e:
        print(f"[오류] Original pipeline failed for {fname}: {e}")
    finally:
        # Free any large arrays from memory
        try:
            del coords, coords_geo, colors_geo, line_coords, line_colors
        except NameError:
            pass

def process_detect_file(file_path: str):
    """Process one Detect file: RGB color filter -> Geometry filter -> RANSAC (skipping done stages accordingly)."""
    fname = os.path.basename(file_path)
    print(f"[Detect] Processing file: {fname}")
    # Skip entire file if final output exists (unless it's an incomplete small file)
    final_path = os.path.join(out_det_ransac, fname)
    if os.path.exists(final_path):
        if os.path.getsize(final_path) >= 100 * 1024:
            print(f"  - Skipping {fname}: final output already exists.")
            return
        else:
            try:
                os.remove(final_path)
                print(f"  - Removed incomplete final output for {fname}, will recompute RANSAC.")
            except OSError:
                pass
    # Check existence of stage outputs
    rgb_path = os.path.join(out_det_rgb, fname)
    geo_path = os.path.join(out_det_geo, fname)
    rgb_exists = os.path.exists(rgb_path)
    geo_exists = os.path.exists(geo_path)
    # Remove small incomplete files if any
    if rgb_exists and os.path.getsize(rgb_path) < 100 * 1024:
        try:
            os.remove(rgb_path)
            print(f"  - Removed incomplete RGB output for {fname}, will recompute color filter.")
        except OSError:
            pass
        rgb_exists = False
    if geo_exists and os.path.getsize(geo_path) < 100 * 1024:
        try:
            os.remove(geo_path)
            print(f"  - Removed incomplete Geo output for {fname}, will recompute geometry filter.")
        except OSError:
            pass
        geo_exists = False

    try:
        if not rgb_exists:
            # Stage 1: Load original file and apply RGB color filter
            t0 = datetime.now(timezone.utc)
            las = laspy.read(file_path)
            coords = np.vstack((las.x, las.y, las.z)).T
            total_points = coords.shape[0]
            print(f"  - Loaded {total_points} points from {fname}")
            # Apply color filter if color data exists
            try:
                # Get color arrays
                r, g, b = las.red, las.green, las.blue
                colors = np.vstack((r, g, b)).T
            except AttributeError:
                colors = None
            t1 = datetime.now(timezone.utc)
            mask_color = filter_by_color(colors)
            t2 = datetime.now(timezone.utc)
            if mask_color is None:
                # No color filtering applied (keep all points)
                coords_color = coords
                colors_color = colors
            else:
                coords_color = coords[mask_color]
                colors_color = colors[mask_color] if colors is not None else None
            t3 = datetime.now(timezone.utc)
            log_step(fname, "Detect", "ColorFilter", t1, t2)
            print(f"  - Color filter: {coords_color.shape[0]} / {total_points} points remain after color filtering")
            # Save Stage 1 (RGB filter) output
            save_las_points(las, coords_color, colors_color, rgb_path)
        else:
            # If RGB stage already done, load its result
            las_color = laspy.read(rgb_path)
            coords_color = np.vstack((las_color.x, las_color.y, las_color.z)).T
            try:
                r, g, b = las_color.red, las_color.green, las_color.blue
                colors_color = np.vstack((r, g, b)).T
            except AttributeError:
                colors_color = None
            total_points = coords_color.shape[0]
            print(f"  - Color stage skipped (loaded {total_points} pre-filtered points).")
        # Now apply geometry filter on color-filtered points (Stage 2)
        if not geo_exists:
            t4 = datetime.now(timezone.utc)
            mask_geo = filter_by_ground(coords_color)
            t5 = datetime.now(timezone.utc)
            coords_geo = coords_color[mask_geo]
            colors_geo = colors_color[mask_geo] if (colors_color is not None and mask_geo is not None) else (None if colors_color is None else colors_color[mask_geo])
            t6 = datetime.now(timezone.utc)
            log_step(fname, "Detect", "GeometryFilter", t4, t5)
            print(f"  - Geometry filter: {coords_geo.shape[0]} / {coords_color.shape[0]} points remain after ground removal")
            # Save Stage 2 (Geo filter) output
            # Use the original LAS header (or las_color's header) as template for saving
            template_las = las_color if 'las_color' in locals() else laspy.read(rgb_path)
            save_las_points(template_las, coords_geo, colors_geo, geo_path)
        else:
            # If geometry stage already done, load its output
            las_geo = laspy.read(geo_path)
            coords_geo = np.vstack((las_geo.x, las_geo.y, las_geo.z)).T
            try:
                r, g, b = las_geo.red, las_geo.green, las_geo.blue
                colors_geo = np.vstack((r, g, b)).T
            except AttributeError:
                colors_geo = None
            print(f"  - Geometry stage skipped (loaded {coords_geo.shape[0]} points after previous Geo filter).")
        # Finally, apply RANSAC on the geometry-filtered points (Stage 3)
        if coords_geo.shape[0] == 0:
            line_coords = np.empty((0,3))
        else:
            t7 = datetime.now(timezone.utc)
            line_coords = extract_powerlines(coords_geo)
            t8 = datetime.now(timezone.utc)
            log_step(fname, "Detect", "RANSAC", t7, t8)
        line_count = line_coords.shape[0]
        print(f"  - RANSAC: found {line_count} line points out of {coords_geo.shape[0]} input points")
        # Determine colors for line points if available
        line_colors = None
        if line_count > 0:
            if colors_geo is not None:
                coord_set = {tuple(pt) for pt in line_coords}
                mask_line = np.array([tuple(pt) in coord_set for pt in coords_geo])
                line_colors = colors_geo[mask_line]
        # Save final RANSAC output (Stage 3)
        # Use geometry stage LAS header as template if available
        template_las = las_geo if 'las_geo' in locals() else (las_color if 'las_color' in locals() else laspy.read(geo_path))
        save_las_points(template_las, line_coords, line_colors, final_path)
        # Remove empty output files (cleanup)
        for path in [rgb_path, geo_path, final_path]:
            if os.path.exists(path) and os.path.getsize(path) < 100 * 1024:
                try:
                    os.remove(path)
                    print(f"    -> Removed empty output file {os.path.basename(path)}")
                except OSError:
                    pass
    except Exception as e:
        print(f"[오류] Detect pipeline failed for {fname}: {e}")
    finally:
        # Free large data arrays
        try:
            del coords, coords_color, coords_geo, colors, colors_color, colors_geo, line_coords, line_colors
        except NameError:
            pass

if __name__ == "__main__":

if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # 오류가 발생한 파일 목록 (반드시 여기에서 관리)
    error_files = [
        "Zenmuse P4R Model Site C_group1_densified_point_cloud_part_12_Original.las",
        "Zenmuse P4R Model Site C_group1_densified_point_cloud_part_13_Original.las",
        "Zenmuse P4R Model Site C_group1_densified_point_cloud_part_3_Original.las",
        "Zenmuse P4R Model Site C_group1_densified_point_cloud_part_21_Original.las",
        "Zenmuse P4R Model Site C_group1_densified_point_cloud_part_52_Original.las",
        "Zenmuse P4R Model Site C_group1_densified_point_cloud_part_51_Original.las"
    ]

    # 전체 파일 경로 변환
    orig_error_files = [os.path.join(original_dir, fname) for fname in error_files]

    # 오류 파일들의 이전 결과 삭제 (Geo, RANSAC 모두 제거하여 처음부터 처리)
    for fname in error_files:
        geo_path = os.path.join(out_orig_geo, fname)
        ransac_path = os.path.join(out_orig_ransac, fname)

        for path in [geo_path, ransac_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"[Info] Removed previous result: {path}")
                except OSError as e:
                    print(f"[Warning] Could not remove {path}: {e}")

    # 오류 파일만 ThreadPoolExecutor로 멀티스레딩 재처리
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = [executor.submit(process_original_file, fpath) for fpath in orig_error_files]
        for future in as_completed(futures):
            _ = future.result()

    print("[Info] 오류가 발생한 파일들에 대한 처음부터 재처리 완료.")



    # Clean up any tiny input LAS files that might be corrupt or empty
    delete_small_files(original_dir)
    delete_small_files(detect_dir)
    # Collect list of .las files to process
    orig_files = [os.path.join(original_dir, f) for f in os.listdir(original_dir) if f.lower().endswith(".las")]
    det_files  = [os.path.join(detect_dir, f) for f in os.listdir(detect_dir) if f.lower().endswith(".las")]
    orig_files.sort()
    det_files.sort()
    # Process files with multi-threading (one thread per file)
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = []
        for f in orig_files:
            futures.append(executor.submit(process_original_file, f))
        for f in det_files:
            futures.append(executor.submit(process_detect_file, f))
        # Wait for all tasks to complete (and catch exceptions if any)
        for future in as_completed(futures):
            _ = future.result()
    # Write processing log to CSV
    log_path = r"G:\UAV_RANSAC\processing_log.csv"
    try:
        import csv
        with open(log_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["File", "Pipeline", "Step", "StartUTC", "EndUTC", "DurationSec"])
            # Sort log entries by file for neatness
            for entry in sorted(log_entries, key=lambda x: x[0]):
                writer.writerow(entry)
        print(f"[Info] Processing log saved to {log_path}")
    except Exception as e:
        print(f"[Warning] Failed to write log CSV: {e}")
