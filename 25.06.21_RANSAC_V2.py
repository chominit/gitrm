import os, sys
import numpy as np
import laspy
from datetime import datetime, timezone
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor

# Directories
original_dir = r"C:\Users\jscool\datasets\Pix4d\Zenmuse Site C Original"
detect_dir   = r"C:\Users\jscool\datasets\Pix4d\Zenmuse Site C Detect\2_densification\point_cloud"

# Output directories
out_orig_geo   = r"G:\UAV_RANSAC\1.Original_RANSAC\1.Geo"
out_orig_ransac= r"G:\UAV_RANSAC\1.Original_RANSAC\2.RANSAC_Finish"
out_det_rgb    = r"G:\UAV_RANSAC\2.Detected_RANSAC\1.RGB"
out_det_geo    = r"G:\UAV_RANSAC\2.Detected_RANSAC\2.Geo"
out_det_ransac = r"G:\UAV_RANSAC\2.Detected_RANSAC\3.RANSAC_Finish"

# Ensure output directories exist
for d in [out_orig_geo, out_orig_ransac, out_det_rgb, out_det_geo, out_det_ransac]:
    os.makedirs(d, exist_ok=True)

# Logging setup
log_entries = []  # will collect tuples to write to CSV at end

def log_step(file_name, pipeline, step, t_start, t_end):
    """Record a log entry for a processing step (with UTC timestamps)."""
    duration = (t_end - t_start).total_seconds()
    start_str = t_start.strftime("%Y-%m-%d %H:%M:%S.%f UTC")
    end_str   = t_end.strftime("%Y-%m-%d %H:%M:%S.%f UTC")
    log_entries.append((file_name, pipeline, step, start_str, end_str, f"{duration:.6f}"))

def delete_small_files(directory, min_bytes=102400):
    """Delete LAS files smaller than min_bytes in the given directory."""
    for fname in os.listdir(directory):
        if fname.lower().endswith(".las"):
            fpath = os.path.join(directory, fname)
            try:
                size = os.path.getsize(fpath)
            except OSError:
                continue
            if size < min_bytes:
                print(f"Removing small file: {fname} ({size} bytes)")
                try:
                    os.remove(fpath)
                except OSError as e:
                    print(f"Warning: failed to remove {fname}: {e}")

def load_las_points(file_path):
    """Load LAS file and return (las, coords_array, color_array)."""
    las = laspy.read(file_path)
    coords = np.vstack((las.x, las.y, las.z)).T  # shape (N,3)
    colors = None
    # If color dimensions present in point format, extract them
    try:
        r = las.red; g = las.green; b = las.blue
        # Normalize 16-bit color to 0-255 float
        color_arr = np.vstack((r, g, b)).T.astype(np.float64)
        # Many LAS store color in 0-65535 range. Convert to 0-255.
        if color_arr.max() > 255:
            color_arr = (color_arr * 255.0 / 65535.0)
        colors = color_arr
    except AttributeError:
        # No color data
        colors = None
    return las, coords, colors

def filter_by_ground(coords):
    """Geometry filter: keep points that are >1m above any neighbor within 15cm XY."""
    if coords.size == 0:
        return np.array([], dtype=bool)
    # Define grid cell size and neighbor height threshold
    cell = 0.15  # 15 cm grid
    z_thresh = 1.0  # 1m above neighbor
    # Compute 2D grid indices for each point
    gx = np.floor(coords[:,0] / cell).astype(int)
    gy = np.floor(coords[:,1] / cell).astype(int)
    # Map each grid cell to lowest Z (ground height in that cell)
    ground_z = {}
    for i, (cx, cy) in enumerate(zip(gx, gy)):
        z = coords[i, 2]
        key = (cx, cy)
        # track minimum z in each cell
        if key in ground_z:
            if z < ground_z[key]:
                ground_z[key] = z
        else:
            ground_z[key] = z
    # Prepare mask array
    N = coords.shape[0]
    keep_mask = np.zeros(N, dtype=bool)
    # For each point, find min ground in its 3x3 neighbor cells (including its own cell)
    for i, (cx, cy) in enumerate(zip(gx, gy)):
        min_neighbor_z = None
        for nx in (cx-1, cx, cx+1):
            for ny in (cy-1, cy, cy+1):
                nz = ground_z.get((nx, ny))
                if nz is None:
                    continue
                if min_neighbor_z is None or nz < min_neighbor_z:
                    min_neighbor_z = nz
        if min_neighbor_z is None:
            # no neighbor points found (should not happen if point itself counted)
            min_neighbor_z = coords[i,2]
        # Check height difference
        if coords[i, 2] > min_neighbor_z + z_thresh:
            keep_mask[i] = True
    return keep_mask

def filter_by_color(colors):
    """Color filter: keep points within ±10% of (255,0,191) in 8-bit normalized color."""
    if colors is None:
        # No color data, keep all
        return None  # indicate no filtering needed
    # Ensure colors are in 0-255 range (float or int)
    col = colors
    if col.dtype != np.float64:
        col = col.astype(np.float64)
    # If values seem above 255 (like up to 65535), scale them
    if col.max() > 255.0:
        col = (col * 255.0 / 65535.0)
    R = col[:,0]; G = col[:,1]; B = col[:,2]
    # Define thresholds (10% tolerance)
    mask = np.ones(R.shape, dtype=bool)
    # R > 229.5 -> R >= 230
    mask &= (R >= 229.5)
    # G < 25.5 -> G <= 25
    mask &= (G <= 25.5)
    # B in [172, 210]
    mask &= (B >= 172.0) & (B <= 210.0)
    return mask

def extract_powerlines(coords):
    """Extract line points using multi-line RANSAC (threshold=0.05m, min_inliers=30)."""
    if coords.shape[0] == 0:
        return np.empty((0,3))
    pts = coords.copy()
    line_inliers = []  # will collect inlier point sets
    # Parameters
    threshold = 0.05  # 5 cm distance threshold
    min_inliers = 30
    max_iterations = 1000  # RANSAC trials per line model
    rng = np.random.default_rng()  # random generator
    # While enough points for a line and model found
    while pts.shape[0] >= min_inliers:
        best_inliers_idx = None
        best_inlier_count = 0
        # RANSAC iterations to find one line
        for _ in range(max_iterations):
            # Randomly sample 2 distinct points
            idx_samples = rng.choice(pts.shape[0], size=2, replace=False)
            p1 = pts[idx_samples[0]]
            p2 = pts[idx_samples[1]]
            # Skip if points are identical or too close
            if np.allclose(p1, p2):
                continue
            # Define line vector and unit direction
            line_vec = p2 - p1
            norm = np.linalg.norm(line_vec)
            if norm < 1e-6:
                continue
            line_dir = line_vec / norm
            # Compute distances of all points to the infinite line defined by p1->p2
            # Vector from p1 to all points
            vecs = pts - p1  # shape (M,3)
            # Cross product magnitude gives area of parallelogram = |line_dir x vec| * |line_dir| 
            # Here |line_dir|=1, so distance = |cross| 
            cross_prod = np.cross(line_dir, vecs)  # shape (M,3)
            dist = np.linalg.norm(cross_prod, axis=1)  # perpendicular distances
            inlier_idx = np.where(dist <= threshold)[0]
            count = inlier_idx.size
            if count > best_inlier_count:
                best_inlier_count = count
                best_inliers_idx = inlier_idx
            # Early break if found a line with a very large inlier count (optional)
            # if best_inlier_count > 1000: break  (for example, but not strictly necessary)
        # After iterations, check best model
        if best_inliers_idx is None or best_inlier_count < min_inliers:
            break  # no good line found
        # Save the inlier points for this line
        line_inliers.append(pts[best_inliers_idx])
        # Remove inliers from point set for next iteration
        mask = np.ones(pts.shape[0], dtype=bool)
        mask[best_inliers_idx] = False
        pts = pts[mask]
    if not line_inliers:
        return np.empty((0,3))
    # Combine all inlier points from all detected lines
    line_points = np.vstack(line_inliers)
    return line_points

def save_las_points(template_las, coords, colors, out_path):
    """Save given points (coords and optional colors) to a new LAS file using template header."""
    # Create new LasData with template header
    new_las = laspy.LasData(template_las.header)
    if coords.shape[0] == 0:
        # No points to save, write empty LAS
        new_las.points = np.array([], dtype=template_las.points.dtype)
    else:
        # If color is available and template has color fields
        if colors is not None and 'red' in template_las.point_format.dimension_names:
            # Ensure we assign the color in correct format (integers likely for LAS)
            # Convert normalized 0-255 floats back to 0-65535 scale integers if needed
            col = colors
            if col.max() <= 255.0:
                # assume it was normalized, scale up to 0-65535
                col = (col * 65535.0 / 255.0)
            col = np.clip(col, 0, 65535).astype(np.uint16)
            # Build points with X, Y, Z, and color dimensions
            # laspy expects scaled integers for coordinates (X, Y, Z) in LasData.points,
            # but we have real coords in template (las.x etc. are already scaled to real).
            # We can directly set new_las.x, new_las.y, new_las.z and new_las.red,... instead.
            new_las.x = coords[:,0]
            new_las.y = coords[:,1]
            new_las.z = coords[:,2]
            new_las.red   = col[:,0]
            new_las.green = col[:,1]
            new_las.blue  = col[:,2]
        else:
            # No color case
            new_las.x = coords[:,0]
            new_las.y = coords[:,1]
            new_las.z = coords[:,2]
    # Optionally update header bounds and point count for safety
    if coords.shape[0] > 0:
        new_las.header.min = np.min(coords, axis=0)
        new_las.header.max = np.max(coords, axis=0)
    new_las.header.point_count = coords.shape[0]
    # Write to file (LAS or LAZ depending on extension of out_path)
    new_las.write(out_path)

def process_detect_file(file_path):
    """Process a file from the Detect directory with both Pipeline A and B."""
    filename = os.path.basename(file_path)
    print(f"\n=== [Detect] Processing file: {filename} ===")
    t0 = datetime.now(timezone.utc)
    try:
        las, coords, colors = load_las_points(file_path)
        total_points = coords.shape[0]
        t1 = datetime.now(timezone.utc)
        log_step(filename, "Detect", "Load", t0, t1)
    except Exception as e:
        print(f"[Error] Failed to load {filename}: {e}")
        return  # skip this file
    # Pipeline A: Geometry -> RGB -> RANSAC
    try:
        # Geometry filter (remove ground)
        tA1 = datetime.now(timezone.utc)
        mask_geo = filter_by_ground(coords)
        coords_geo = coords[mask_geo]
        colors_geo = colors[mask_geo] if (colors is not None and mask_geo is not None) else (None if colors is None else colors[mask_geo])
        tA2 = datetime.now(timezone.utc)
        log_step(filename, "Detect-A", "GeometryFilter", tA1, tA2)
        print(f"[Geo Filter A] Points after ground removal: {coords_geo.shape[0]} / {total_points}")
        # RGB filter
        tA3 = datetime.now(timezone.utc)
        mask_color = filter_by_color(colors_geo)
        if mask_color is None:
            # No color data, skip color filtering (keep all)
            coords_geo_rgb = coords_geo
            colors_geo_rgb = colors_geo
        else:
            coords_geo_rgb = coords_geo[mask_color]
            colors_geo_rgb = colors_geo[mask_color] if colors_geo is not None else None
        tA4 = datetime.now(timezone.utc)
        log_step(filename, "Detect-A", "ColorFilter", tA3, tA4)
        print(f"[Color Filter A] Points after color filter: {coords_geo_rgb.shape[0]} / {coords_geo.shape[0]}")
        # RANSAC line extraction
        tA5 = datetime.now(timezone.utc)
        line_coords_A = extract_powerlines(coords_geo_rgb)
        tA6 = datetime.now(timezone.utc)
        log_step(filename, "Detect-A", "RANSAC", tA5, tA6)
        print(f"[RANSAC A] Line points found: {line_coords_A.shape[0]} / {coords_geo_rgb.shape[0]}")
        # Save outputs for Pipeline A
        # (We don't have separate intermediate output dirs for A in spec, only final lines)
        if line_coords_A.shape[0] > 0:
            # Save final RANSAC result for pipeline A (Geo->RGB) with suffix
            out_path = os.path.join(out_det_ransac, filename.replace('.las', '_geoRGB.las'))
            save_las_points(las, line_coords_A, colors_geo_rgb, out_path)
    except Exception as e:
        print(f"[Error] Pipeline A failed for {filename}: {e}")
    # Pipeline B: RGB -> Geometry -> RANSAC
    try:
        # RGB filter first
        tB1 = datetime.now(timezone.utc)
        mask_color2 = filter_by_color(colors)
        if mask_color2 is None:
            coords_color = coords
            colors_color = colors
        else:
            coords_color = coords[mask_color2]
            colors_color = colors[mask_color2] if colors is not None else None
        tB2 = datetime.now(timezone.utc)
        log_step(filename, "Detect-B", "ColorFilter", tB1, tB2)
        print(f"[Color Filter B] Points after color filter: {coords_color.shape[0]} / {total_points}")
        # Save Stage 1 (RGB filter result) for pipeline B
        out_path_stage1 = os.path.join(out_det_rgb, filename)
        save_las_points(las, coords_color, colors_color, out_path_stage1)
        # Geometry filter second
        tB3 = datetime.now(timezone.utc)
        mask_geo2 = filter_by_ground(coords_color)
        coords_rgb_geo = coords_color[mask_geo2]
        colors_rgb_geo = colors_color[mask_geo2] if (colors_color is not None and mask_geo2 is not None) else (None if colors_color is None else colors_color[mask_geo2])
        tB4 = datetime.now(timezone.utc)
        log_step(filename, "Detect-B", "GeometryFilter", tB3, tB4)
        print(f"[Geo Filter B] Points after ground removal: {coords_rgb_geo.shape[0]} / {coords_color.shape[0]}")
        # Save Stage 2 (Geo filter result) for pipeline B
        out_path_stage2 = os.path.join(out_det_geo, filename)
        save_las_points(las, coords_rgb_geo, colors_rgb_geo, out_path_stage2)
        # RANSAC line extraction
        tB5 = datetime.now(timezone.utc)
        line_coords_B = extract_powerlines(coords_rgb_geo)
        tB6 = datetime.now(timezone.utc)
        log_step(filename, "Detect-B", "RANSAC", tB5, tB6)
        print(f"[RANSAC B] Line points found: {line_coords_B.shape[0]} / {coords_rgb_geo.shape[0]}")
        # Save final RANSAC result for Pipeline B
        if line_coords_B.shape[0] > 0:
            out_path3 = os.path.join(out_det_ransac, filename.replace('.las', '_rgbGeo.las'))
            # Find colors for inlier points: they are subset of coords_rgb_geo
            if colors_rgb_geo is not None and line_coords_B.shape[0] > 0:
                # We need to get the color of line_coords_B points.
                # Easiest is to match by coordinates (assuming exact matches).
                coord_set = {tuple(pt) for pt in line_coords_B}
                mask_line = np.array([tuple(pt) in coord_set for pt in coords_rgb_geo])
                line_colors_B = colors_rgb_geo[mask_line]
            else:
                line_colors_B = colors_rgb_geo  # could be None
            save_las_points(las, line_coords_B, line_colors_B, out_path3)
    except Exception as e:
        print(f"[Error] Pipeline B failed for {filename}: {e}")
    print(f"=== [Detect] Completed file: {filename} ===")

def process_original_file(file_path):
    """Process a file from the Original directory with geometry filter then RANSAC."""
    filename = os.path.basename(file_path)
    print(f"\n=== [Original] Processing file: {filename} ===")
    t0 = datetime.now(timezone.utc)
    try:
        las, coords, colors = load_las_points(file_path)
        total_points = coords.shape[0]
        t1 = datetime.now(timezone.utc)
        log_step(filename, "Original", "Load", t0, t1)
    except Exception as e:
        print(f"[Error] Failed to load {filename}: {e}")
        return
    try:
        # Geometry filter (ground removal)
        t1g = datetime.now(timezone.utc)
        mask_geo = filter_by_ground(coords)
        coords_geo = coords[mask_geo]
        colors_geo = colors[mask_geo] if (colors is not None and mask_geo is not None) else (None if colors is None else colors[mask_geo])
        t2g = datetime.now(timezone.utc)
        log_step(filename, "Original", "GeometryFilter", t1g, t2g)
        print(f"[Geo Filter] Points after ground removal: {coords_geo.shape[0]} / {total_points}")
        # Save intermediate Geo-filtered points (Original)
        out_path1 = os.path.join(out_orig_geo, filename)
        save_las_points(las, coords_geo, colors_geo, out_path1)
        # RANSAC line extraction
        t1r = datetime.now(timezone.utc)
        line_coords = extract_powerlines(coords_geo)
        t2r = datetime.now(timezone.utc)
        log_step(filename, "Original", "RANSAC", t1r, t2r)
        print(f"[RANSAC] Line points found: {line_coords.shape[0]} / {coords_geo.shape[0]}")
        # Save final RANSAC result (Original)
        if line_coords.shape[0] > 0:
            # If color info exists, extract color of inlier points similarly
            if colors_geo is not None and line_coords.shape[0] > 0:
                coord_set = {tuple(pt) for pt in line_coords}
                mask_line = np.array([tuple(pt) in coord_set for pt in coords_geo])
                line_colors = colors_geo[mask_line]
            else:
                line_colors = colors_geo
            out_path2 = os.path.join(out_orig_ransac, filename)
            save_las_points(las, line_coords, line_colors, out_path2)
    except Exception as e:
        print(f"[Error] Original pipeline failed for {filename}: {e}")
    print(f"=== [Original] Completed file: {filename} ===")

if __name__ == "__main__":
    delete_small_files(original_dir)
    delete_small_files(detect_dir)
    
    orig_files = [os.path.join(original_dir, f) for f in os.listdir(original_dir) if f.lower().endswith(".las")]
    det_files  = [os.path.join(detect_dir, f) for f in os.listdir(detect_dir) if f.lower().endswith(".las")]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        executor.map(process_original_file, orig_files)
        executor.map(process_detect_file, det_files)

    # Write log to CSV
    log_path = r"G:\UAV_RANSAC\processing_log.csv"
    try:
        import csv
        with open(log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["File", "Pipeline", "Step", "StartUTC", "EndUTC", "DurationSec"])
            for entry in log_entries:
                writer.writerow(entry)
        print(f"Processing log saved to {log_path}")
    except Exception as e:
        print(f"Failed to write log CSV: {e}")
