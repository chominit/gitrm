import os
import numpy as np
import laspy
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Directories for input files
original_dir = r"C:\Users\jscool\datasets\Pix4d\Zenmuse Site C Original"
detect_dir = r"C:\Users\jscool\datasets\Pix4d\Zenmuse Site C Detect\2_densification\point_cloud"

# Directories for output files by stage
out_orig_geo = r"G:\UAV_RANSAC\1.Original_RANSAC\1.Geo"
out_orig_ransac = r"G:\UAV_RANSAC\1.Original_RANSAC\2.RANSAC_Finish"
out_det_color = r"G:\UAV_RANSAC\2.Detected_RANSAC\1.RGB"
out_det_geo = r"G:\UAV_RANSAC\2.Detected_RANSAC\2.Geo"
out_det_ransac = r"G:\UAV_RANSAC\2.Detected_RANSAC\3.RANSAC_Finish"

# Ensure output directories exist
for d in [out_orig_geo, out_orig_ransac, out_det_geo, out_det_color, out_det_ransac]:
    os.makedirs(d, exist_ok=True)

# Logging setup
log_entries = []
log_lock = threading.Lock()

def log_step(file_name: str, pipeline: str, step: str, t_start: datetime, t_end: datetime):
    """Record a log entry for a processing step with UTC timestamps and duration."""
    duration = (t_end - t_start).total_seconds()
    start_str = t_start.strftime("%Y-%m-%d %H:%M:%S.%f UTC")
    end_str   = t_end.strftime("%Y-%m-%d %H:%M:%S.%f UTC")
    entry = (file_name, pipeline, step, start_str, end_str, f"{duration:.6f}")
    # Append to global log in a thread-safe manner
    with log_lock:
        log_entries.append(entry)

# Helper to delete any tiny LAS files (e.g., incomplete outputs)
def delete_small_files(directory: str, min_bytes: int = 102400):
    """Delete .las files smaller than min_bytes in the given directory."""
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
                    print(f"[Warning] Could not remove file {fname}: {e}")

def load_las_points(file_path: str):
    """Load a LAS file and return (las_data, coords_array, colors_array)."""
    las = laspy.read(file_path)
    coords = np.vstack((las.x, las.y, las.z)).T  # N x 3 array of coordinates
    # Try to extract color data (if available)
    try:
        r = las.red; g = las.green; b = las.blue
        colors = np.vstack((r, g, b)).T
    except AttributeError:
        colors = None
    return las, coords, colors

def save_las_points(template_las: laspy.LasData, coords: np.ndarray, colors: np.ndarray, out_path: str):
    """Save points (coords and optional colors) to a LAS file at out_path, using template_las for header."""
    new_las = laspy.create(point_format=template_las.header.point_format, file_version=template_las.header.version)

    # 포인트가 있을 때만 필드를 개별 할당
    num_points = coords.shape[0]
    if num_points > 0:
        new_las.x = coords[:, 0]
        new_las.y = coords[:, 1]
        new_las.z = coords[:, 2]

        # RGB 필드가 존재하면 함께 처리
        if colors is not None and {'red', 'green', 'blue'}.issubset(template_las.point_format.dimension_names):
            col = colors.astype(np.float64)
            if col.max() <= 255.0:
                col = col * (65535.0 / 255.0)
            col = np.clip(col, 0, 65535).astype(np.uint16)
            new_las.red = col[:, 0]
            new_las.green = col[:, 1]
            new_las.blue = col[:, 2]

    # 헤더 업데이트 (점 수 및 좌표 범위)
    new_las.header.point_count = num_points
    if num_points > 0:
        new_las.header.min = np.min(coords, axis=0)
        new_las.header.max = np.max(coords, axis=0)

    # 파일 저장
    new_las.write(out_path)

def filter_by_ground(coords: np.ndarray) -> np.ndarray:
    N = coords.shape[0]
    mask = np.zeros(N, dtype=bool)
    if N == 0:
        return mask
    cell_size = 0.15
    height_threshold = 1.0

    # 최소 좌표를 기준으로 보정된 셀 인덱스를 계산
    min_x, min_y = np.min(coords[:, 0]), np.min(coords[:, 1])
    gx = np.floor((coords[:, 0] - min_x) / cell_size).astype(int)
    gy = np.floor((coords[:, 1] - min_y) / cell_size).astype(int)

    ground_z = {}
    for (x_idx, y_idx, z_val) in zip(gx, gy, coords[:, 2]):
        key = (x_idx, y_idx)
        if key in ground_z:
            if z_val < ground_z[key]:
                ground_z[key] = z_val
        else:
            ground_z[key] = z_val

    for i in range(N):
        x_idx, y_idx, z_val = gx[i], gy[i], coords[i, 2]
        neighbor_min_z = float('inf')
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor_key = (x_idx + dx, y_idx + dy)
                if neighbor_key in ground_z:
                    nz = ground_z[neighbor_key]
                    if nz < neighbor_min_z:
                        neighbor_min_z = nz
        if z_val >= neighbor_min_z + height_threshold:
            mask[i] = True
    return mask

def filter_by_color(colors: np.ndarray):
    """Color filter: return mask of points close to target color (255,0,191) with 10% tolerance."""
    if colors is None:
        return None  # no color data, skip filtering
    # Ensure type is float for computation
    col = colors.astype(np.float64)
    # Normalize 16-bit color values to 0-255 range if needed
    if col.max() > 255.0:
        col = col * (255.0 / 65535.0)
    R, G, B = col[:, 0], col[:, 1], col[:, 2]
    mask = np.ones(R.shape, dtype=bool)
    # ±10% tolerance around (255,0,191) in 0-255 scale
    mask &= (R >= 229.5)        # R >= 90% of 255
    mask &= (G <= 25.5)         # G <= 10% of 255
    mask &= ((B >= 172.0) & (B <= 210.0))  # B within ~±10% of 191
    return mask

def extract_powerlines(coords: np.ndarray) -> np.ndarray:
    """Extract power line points using iterative RANSAC line-fitting (5cm threshold, min 30 points per line)."""
    if coords.shape[0] < 2:
        return np.empty((0, 3))
    pts = coords.copy()
    line_points = []
    threshold = 0.05  # 5 cm distance threshold for inliers
    min_inliers = 30
    max_iterations = 1000
    rng = np.random.default_rng()
    # Find lines iteratively
    while pts.shape[0] >= min_inliers:
        best_inliers_idx = None
        best_count = 0
        # RANSAC loop to find one line
        for _ in range(max_iterations):
            # sample 2 random points to define a line
            i1, i2 = rng.choice(pts.shape[0], size=2, replace=False)
            p1, p2 = pts[i1], pts[i2]
            if np.allclose(p1, p2):
                continue  # almost identical points, skip
            line_vec = p2 - p1
            norm = np.linalg.norm(line_vec)
            if norm < 1e-6:
                continue  # degenerate line
            line_dir = line_vec / norm
            # distance of all points to infinite line
            vecs = pts - p1
            cross_prod = np.cross(line_dir, vecs)
            dist = np.linalg.norm(cross_prod, axis=1)
            inlier_idx = np.where(dist <= threshold)[0]
            count = inlier_idx.size
            if count > best_count:
                best_count = count
                best_inliers_idx = inlier_idx
            # (optional: could break early if a very large inlier set is found)
        if best_inliers_idx is None or best_count < min_inliers:
            break  # no sufficient line found
        # Save inlier points for this line
        line_points.append(pts[best_inliers_idx])
        # Remove these inliers from consideration for next lines
        mask = np.ones(pts.shape[0], dtype=bool)
        mask[best_inliers_idx] = False
        pts = pts[mask]
    if len(line_points) == 0:
        return np.empty((0, 3))
    return np.vstack(line_points)

# Globals to track failures
fail_lock = threading.Lock()
failed_files = []  # will store tuples (file_path, pipeline, failed_stage)

def process_original_file(file_path: str):
    """Process a file from the Original directory: Geometry filter -> RANSAC."""
    fname = os.path.basename(file_path)
    pipeline = "Original"
    try:
        # Stage 1: GeometryFilter
        geo_path = os.path.join(out_orig_geo, fname)
        geo_exists = os.path.exists(geo_path)
        # If geometry output exists, check size
        if geo_exists:
            if os.path.getsize(geo_path) < 102400:
                # Incomplete output, remove it and recompute geometry
                try:
                    os.remove(geo_path)
                    print(f"[Info] Removed incomplete Geo output for {fname}, will recompute GeometryFilter.")
                except OSError:
                    pass
                geo_exists = False
            else:
                print(f"[Info] Skipping GeometryFilter for {fname} (already done).")
        # Load input data if we need to run geometry filter, or load existing geometry output if skipping
        if not geo_exists:
            # Load original LAS file
            las, coords, colors = load_las_points(file_path)
            total_points = coords.shape[0]
            # Apply geometry filter (remove ground points)
            t_start = datetime.now(timezone.utc)
            mask_geo = filter_by_ground(coords)
            t_end = datetime.now(timezone.utc)
            coords_geo = coords[mask_geo]
            colors_geo = colors[mask_geo] if (colors is not None and mask_geo is not None) else (None if colors is None else colors[mask_geo])
            log_step(fname, pipeline, "GeometryFilter", t_start, t_end)
            print(f"[Original] GeometryFilter: {coords_geo.shape[0]} / {total_points} points remaining after ground removal.")
            # Save geometry-filtered points
            save_las_points(las, coords_geo, colors_geo, geo_path)
        else:
            # Geometry already done, load the geo-filtered data for next stage
            las_geo, coords_geo, colors_geo = load_las_points(geo_path)
            print(f"[Original] GeometryFilter skipped (loaded {coords_geo.shape[0]} points from existing output).")
        # Stage 2: ColorFilter (Optional for Original - skip if color filter is not needed)
        # Original pipeline typically doesn't use color filtering; we will skip this stage.
        # If needed, one could insert color filtering here similar to Detect pipeline.
        # For now, we assume no color filter on Original.
        # Stage 3: RANSAC
        final_path = os.path.join(out_orig_ransac, fname)
        # If final output exists, check if it's complete
        if os.path.exists(final_path):
            if os.path.getsize(final_path) < 102400:
                try:
                    os.remove(final_path)
                    print(f"[Info] Removed incomplete final output for {fname}, will recompute RANSAC.")
                except OSError:
                    pass
            else:
                print(f"[Info] Skipping RANSAC for {fname} (final output already exists).")
                return  # entire file is already processed successfully
        # Run RANSAC on geometry-filtered points
        if coords_geo.shape[0] == 0:
            # No points to process (all filtered out); create an empty result
            line_coords = np.empty((0, 3))
        else:
            t_start = datetime.now(timezone.utc)
            line_coords = extract_powerlines(coords_geo)
            t_end = datetime.now(timezone.utc)
            log_step(fname, pipeline, "RANSAC", t_start, t_end)
        line_count = line_coords.shape[0]
        print(f"[Original] RANSAC: found {line_count} line points out of {coords_geo.shape[0]} points.")
        # Save RANSAC result (even if empty, an empty LAS will be created)
        # If original had color data, try to get corresponding colors for line points
        line_colors = None
        if line_count > 0 and colors_geo is not None:
            # Find colors of the line points by matching coordinates (exact match since they come from coords_geo)
            coord_set = {tuple(pt) for pt in line_coords}
            mask_line = np.array([tuple(pt) in coord_set for pt in coords_geo])
            line_colors = colors_geo[mask_line]
        # Use geometry stage LAS header as template for saving (las_geo is available from above)
        template_las = las_geo if 'las_geo' in locals() else las  # fall back to original LAS if needed
        save_las_points(template_las, line_coords, line_colors, final_path)
        # If final output is empty (small file), treat as failure
        if os.path.getsize(final_path) < 102400:
            # Remove the empty file and raise an error to trigger retry
            try:
                os.remove(final_path)
            except OSError:
                pass
            raise RuntimeError("RANSAC output empty (no line points detected)")
    except Exception as e:
        # Log and record failure
        print(f"[오류] {pipeline} pipeline failed for {fname}: {e}")
        with fail_lock:
            failed_files.append((file_path, pipeline))
    # (No return value needed; log and fail list capture outcome)

def process_detect_file(file_path: str):
    """Process a file from the Detect directory: Geometry filter -> Color filter -> RANSAC."""
    fname = os.path.basename(file_path)
    pipeline = "Detect"
    try:
        # Stage 1: GeometryFilter
        geo_path = os.path.join(out_det_geo, fname)
        geo_exists = os.path.exists(geo_path)
        if geo_exists:
            if os.path.getsize(geo_path) < 102400:
                try:
                    os.remove(geo_path)
                    print(f"[Info] Removed incomplete Geo output for {fname}, will recompute GeometryFilter.")
                except OSError:
                    pass
                geo_exists = False
            else:
                print(f"[Info] Skipping GeometryFilter for {fname} (already done).")
        # We need data for either doing geometry filter or for skipping directly to color filter.
        # If geometry not done, load original file first.
        if not geo_exists:
            las, coords, colors = load_las_points(file_path)
            total_points = coords.shape[0]
            # Apply geometry filter first (assuming we remove ground before color filtering)
            t_start = datetime.now(timezone.utc)
            mask_geo = filter_by_ground(coords)
            t_end = datetime.now(timezone.utc)
            coords_geo = coords[mask_geo]
            colors_geo = colors[mask_geo] if (colors is not None and mask_geo is not None) else (None if colors is None else colors[mask_geo])
            log_step(fname, pipeline, "GeometryFilter", t_start, t_end)
            print(f"[Detect] GeometryFilter: {coords_geo.shape[0]} / {total_points} points remain after ground removal.")
            # Save geometry-filtered output for this Detect file
            save_las_points(las, coords_geo, colors_geo, geo_path)
        else:
            # Geometry already done previously, load those filtered points
            las_geo, coords_geo, colors_geo = load_las_points(geo_path)
            total_points = coords_geo.shape[0]
            print(f"[Detect] GeometryFilter skipped (loaded {total_points} points from existing output).")
        # Stage 2: ColorFilter
        color_path = os.path.join(out_det_color, fname)
        color_exists = os.path.exists(color_path)
        if color_exists:
            if os.path.getsize(color_path) < 102400:
                try:
                    os.remove(color_path)
                    print(f"[Info] Removed incomplete Color output for {fname}, will recompute ColorFilter.")
                except OSError:
                    pass
                color_exists = False
            else:
                print(f"[Info] Skipping ColorFilter for {fname} (already done).")
        if not color_exists:
            # If we reached here, we have coords_geo (output of geometry stage) ready
            # Apply color filter on those points
            t_start = datetime.now(timezone.utc)
            mask_color = filter_by_color(colors_geo)
            t_end = datetime.now(timezone.utc)
            if mask_color is None:
                # No color data or no filtering needed
                coords_color = coords_geo
                colors_color = colors_geo
            else:
                coords_color = coords_geo[mask_color]
                colors_color = colors_geo[mask_color] if colors_geo is not None else None
            log_step(fname, pipeline, "ColorFilter", t_start, t_end)
            print(f"[Detect] ColorFilter: {coords_color.shape[0]} / {coords_geo.shape[0]} points remain after color filtering.")
            # Save color-filtered output
            # Use the geometry stage LAS (las_geo) as template for saving
            template_las = las_geo  # las_geo holds the header from original input (post-geometry)
            save_las_points(template_las, coords_color, colors_color, color_path)
        else:
            # Color already done, load color-filtered result
            las_color, coords_color, colors_color = load_las_points(color_path)
            print(f"[Detect] ColorFilter skipped (loaded {coords_color.shape[0]} points from existing output).")
        # Stage 3: RANSAC
        final_path = os.path.join(out_det_ransac, fname)
        if os.path.exists(final_path):
            if os.path.getsize(final_path) < 102400:
                try:
                    os.remove(final_path)
                    print(f"[Info] Removed incomplete final output for {fname}, will recompute RANSAC.")
                except OSError:
                    pass
            else:
                print(f"[Info] Skipping RANSAC for {fname} (final output already exists).")
                return
        # Perform RANSAC on the color-filtered points
        if coords_color.shape[0] == 0:
            line_coords = np.empty((0, 3))
        else:
            t_start = datetime.now(timezone.utc)
            line_coords = extract_powerlines(coords_color)
            t_end = datetime.now(timezone.utc)
            log_step(fname, pipeline, "RANSAC", t_start, t_end)
        line_count = line_coords.shape[0]
        print(f"[Detect] RANSAC: found {line_count} line points out of {coords_color.shape[0]} points.")
        # Determine colors for line points (if any and if color data available)
        line_colors = None
        if line_count > 0 and colors_color is not None:
            coord_set = {tuple(pt) for pt in line_coords}
            mask_line = np.array([tuple(pt) in coord_set for pt in coords_color])
            line_colors = colors_color[mask_line]
        # Use color stage LAS header as template (las_color or las_geo as available)
        template_las = locals().get('las_color', None)
        if template_las is None:
            template_las = las_geo  # if color stage was skipped or not present, use geometry stage header
        save_las_points(template_las, line_coords, line_colors, final_path)
        # Validate final output size
        if os.path.getsize(final_path) < 102400:
            try:
                os.remove(final_path)
            except OSError:
                pass
            raise RuntimeError("RANSAC output empty (no line points detected)")
    except Exception as e:
        print(f"[오류] {pipeline} pipeline failed for {fname}: {e}")
        with fail_lock:
            failed_files.append((file_path, pipeline))
    # end of process_detect_file

# ---- Main execution block ----
if __name__ == "__main__":
    # Step 1: Clean up any small/corrupt input files in advance (optional safety measure)
    delete_small_files(original_dir)
    delete_small_files(detect_dir)

    # Gather list of files to process
    orig_files = [os.path.join(original_dir, f) for f in os.listdir(original_dir) if f.lower().endswith(".las")]
    det_files  = [os.path.join(detect_dir, f) for f in os.listdir(detect_dir) if f.lower().endswith(".las")]
    orig_files.sort()
    det_files.sort()

    # Process all files concurrently (multithreading)
    max_workers = os.cpu_count() or 32
    print(f"[Info] Starting processing of {len(orig_files)} original and {len(det_files)} detect files with {max_workers} threads...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for f in orig_files:
            futures.append(executor.submit(process_original_file, f))
        for f in det_files:
            futures.append(executor.submit(process_detect_file, f))
        # Wait for all tasks to complete
        for future in as_completed(futures):
            _ = future.result()  # we already handle exceptions inside the functions

    # If any files failed, attempt a one-time re-run for those starting at their failed stage
    if failed_files:
        # Copy the list of failed files (to avoid modification issues)
        to_retry = failed_files.copy()
        failed_files.clear()  # reset for next attempt
        print(f"[Info] Retrying {len(to_retry)} failed files...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for (fpath, pipeline) in to_retry:
                if pipeline == "Original":
                    futures.append(executor.submit(process_original_file, fpath))
                else:
                    futures.append(executor.submit(process_detect_file, fpath))
            for future in as_completed(futures):
                _ = future.result()
        # After retry, check if any still failed
        if failed_files:
            # These files failed again; we will not attempt further in this run
            for (fpath, pipe) in failed_files:
                fname = os.path.basename(fpath)
                print(f"[Error] File {fname} in {pipe} pipeline failed again. Manual investigation may be required.")
        else:
            print("[Info] All failed files have been reprocessed successfully.")
    else:
        print("[Info] All files processed successfully on first attempt.")

    # Write log entries to CSV
    log_path = "processing_log.csv"
    try:
        import csv
        # Sort log entries by file name for clarity
        log_entries_sorted = sorted(log_entries, key=lambda x: x[0])
        with open(log_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["File", "Pipeline", "Step", "StartUTC", "EndUTC", "DurationSec"])
            for entry in log_entries_sorted:
                writer.writerow(entry)
        print(f"[Info] Processing log saved to {log_path}")
    except Exception as e:
        print(f"[Warning] Failed to write log CSV: {e}")
