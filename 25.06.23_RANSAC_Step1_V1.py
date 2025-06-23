import os, sys
import numpy as np
import laspy
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ─────────────────────────────────────────
# 0) 경로 설정
# ─────────────────────────────────────────
original_dir = r"C:\Users\jscool\datasets\Pix4d\Zenmuse Site C Original"
detect_dir   = r"C:\Users\jscool\datasets\Pix4d\Zenmuse Site C Detect\2_densification\point_cloud"

out_orig_geo    = r"G:\UAV_RANSAC\1.Original_RANSAC\1.Geo"
out_orig_ransac = r"G:\UAV_RANSAC\1.Original_RANSAC\2.RANSAC_Finish"
out_det_color   = r"G:\UAV_RANSAC\2.Detected_RANSAC\1.RGB"
out_det_geo     = r"G:\UAV_RANSAC\2.Detected_RANSAC\2.Geo"
out_det_ransac  = r"G:\UAV_RANSAC\2.Detected_RANSAC\3.RANSAC_Finish"

for d in [out_orig_geo, out_det_color, out_det_geo]:   # ← 변수 통일
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────
# 1) Original Geometry 스킵 기준
# ─────────────────────────────────────────
EXPECTED_FILE_COUNT = 37   # 스킵하려면 .las 파일이 정확히 37개
SIZE_THRESHOLD_GB   = 20   # 용량 합계가 20GB 이상





# 로그 저장용 리스트와 락
log_entries = []
log_lock = threading.Lock()

def log_step(file_name, pipeline, step, t_start, t_end):
    """UTC 기준 시작/종료 시간과 소요 시간을 기록하는 로그 항목 추가 함수."""
    duration = (t_end - t_start).total_seconds()
    start_str = t_start.strftime("%Y-%m-%d %H:%M:%S.%f UTC")
    end_str   = t_end.strftime("%Y-%m-%d %H:%M:%S.%f UTC")
    entry = (file_name, pipeline, step, start_str, end_str, f"{duration:.6f}")
    with log_lock:
        log_entries.append(entry)

def delete_small_files(directory, min_bytes=102400):
    """지정된 디렉터리에서 크기가 min_bytes 미만인 .las 파일 삭제 (불완전 파일 정리)."""
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
                    print(f"[Info] 작은 불완전 파일 삭제: {fname} ({size} bytes)")
                except OSError as e:
                    print(f"[경고] 파일 삭제 실패: {fname} - {e}")

def filter_by_ground(coords: np.ndarray) -> np.ndarray:
    """Geometry 필터: 주변 격자셀의 최저 지면보다 height_threshold 이상 높은 점만 유지 (지면 제거)."""
    N = coords.shape[0]
    mask = np.zeros(N, dtype=bool)
    if N == 0:
        return mask  # 포인트 없음
    cell_size = 0.15        # 격자 셀 크기 (예: 15cm)
    height_threshold = 1.0  # 높이 임계값 (1m)
    # 각 포인트의 격자 좌표 계산
    gx = np.floor(coords[:, 0] / cell_size).astype(int)
    gy = np.floor(coords[:, 1] / cell_size).astype(int)
    # 각 셀의 최소 z값 (지면 고도) 계산
    ground_z = {}
    for x_idx, y_idx, z_val in zip(gx, gy, coords[:, 2]):
        key = (x_idx, y_idx)
        ground_z[key] = min(ground_z.get(key, z_val), z_val)
    # 인접 3x3 셀 내 최저 지면 높이와 비교하여 height_threshold 이상 높은 점 식별
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
    """Color 필터: 대상 색상 (255,0,191)의 ±10% 범위 내에 속하는 포인트만 유지."""
    if colors is None:
        return None  # 색상 데이터 없음
    # 색상 데이터를 float로 변환하여 범위 계산
    col = colors.astype(np.float64)
    # 16-bit 컬러값이면 0-255로 정규화
    if col.max() > 255.0:
        col = col * 255.0 / 65535.0
    R, G, B = col[:, 0], col[:, 1], col[:, 2]
    mask = np.ones(R.shape, dtype=bool)
    # R >= 230 (255의 90%), G <= 25 (255의 10%), B는 191 ±10% (약 172~210) 조건
    mask &= (R >= 229.5)
    mask &= (G <= 25.5)
    mask &= (B >= 172.0) & (B <= 210.0)
    return mask

def save_las_points(template_las, coords: np.ndarray, colors: np.ndarray, out_path: str):
    """주어진 템플릿 LAS 헤더로 좌표 및 색상 배열을 LAS 파일로 저장."""
    new_las = laspy.LasData(template_las.header)
    if coords.shape[0] == 0:
        # 포인트가 없으면 빈 구조로 초기화
        new_las.points = np.array([], dtype=template_las.points.dtype)
    else:
        if colors is not None and 'red' in template_las.point_format.dimension_names:
            # RGB 필드가 있고 색상 데이터가 있을 경우
            col = colors.astype(np.float64)
            if col.max() <= 255.0:
                # 0-255 범위 색상을 LAS 표준 0-65535로 변환
                col = col * 65535.0 / 255.0
            col = np.clip(col, 0, 65535).astype(np.uint16)
            new_las.x = coords[:, 0];  new_las.y = coords[:, 1];  new_las.z = coords[:, 2]
            new_las.red   = col[:, 0]; new_las.green = col[:, 1]; new_las.blue  = col[:, 2]
        else:
            # 색상 데이터 없음 (또는 LAS에 RGB 필드 없음)
            new_las.x = coords[:, 0]; new_las.y = coords[:, 1]; new_las.z = coords[:, 2]
    # 헤더 정보 업데이트 (경계와 포인트 수)
    if coords.shape[0] > 0:
        new_las.header.min = np.min(coords, axis=0)
        new_las.header.max = np.max(coords, axis=0)
    new_las.header.point_count = coords.shape[0]
    # LAS 파일 저장
    new_las.write(out_path)

def process_original_file(file_path: str):
    """Original 폴더의 단일 LAS 파일에 Geometry 필터 적용."""
    fname = os.path.basename(file_path)
    print(f"\n=== [Original] 파일 처리 시작: {fname} ===")
    out_path = os.path.join(out_orig_geo, fname)
    # 이미 처리된 파일은 건너뛰기 (완전한 결과 존재 시)
    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        if size >= 100 * 1024:
            print(f"  - {fname} 기존 결과가 있어 건너뜀.")
            return
        else:
            # 불완전한 출력 파일 삭제 후 진행
            try:
                os.remove(out_path)
                print(f"  - 불완전 출력 파일 삭제: {fname}, 재처리 진행")
            except OSError:
                pass
    try:
        # LAS 파일 로드
        las = laspy.read(file_path)
        coords = np.vstack((las.x, las.y, las.z)).T
        total_points = coords.shape[0]
        print(f"  - 로드 완료: {total_points} points (파일: {fname})")
    except Exception as e:
        print(f"[오류] 파일 로드 실패: {fname} - {e}")
        return
    try:
        # Geometry 필터 적용 (지면 제거)
        t_start = datetime.now(timezone.utc)
        mask_geo = filter_by_ground(coords)
        t_end = datetime.now(timezone.utc)
        coords_geo = coords[mask_geo]
        # 원본 색상 데이터 유지 (있을 경우)
        try:
            r, g, b = las.red, las.green, las.blue
            color_arr = np.vstack((r, g, b)).T
            if color_arr.max() > 255:
                color_arr = (color_arr * 255.0 / 65535.0).astype(np.float64)
            colors_geo = color_arr[mask_geo] if mask_geo.any() else np.empty((0, 3))
        except AttributeError:
            colors_geo = None
        # 로그 기록 및 결과 출력
        log_step(fname, "Original", "GeometryFilter", t_start, t_end)
        print(f"  - Geometry 필터: {coords_geo.shape[0]} / {total_points} 점 유지 (지면 위 객체)")
        # 결과 LAS 저장
        save_las_points(las, coords_geo, colors_geo, out_path)
    except Exception as e:
        print(f"[오류] Geometry 필터 실패: {fname} - {e}")
    finally:
        # 메모리 해제
        try:
            las.close()
        except:
            pass
        try:
            del las, coords, coords_geo, colors_geo, color_arr
        except NameError:
            pass
    # 출력 파일이 너무 작으면 (포인트 없음 혹은 극소수) 삭제
    if os.path.exists(out_path) and os.path.getsize(out_path) < 100 * 1024:
        try:
            os.remove(out_path)
            print(f"    -> 결과 포인트 없음, 출력 파일 삭제: {fname}")
        except OSError:
            pass
    print(f"=== [Original] 파일 처리 완료: {fname} ===")

def process_detect_file(file_path: str):
    """Detect 폴더의 단일 LAS 파일에 Color 필터 + Geometry 필터 연속 적용."""
    fname = os.path.basename(file_path)
    print(f"\n=== [Detect] 파일 처리 시작: {fname} ===")
    rgb_path = os.path.join(out_det_rgb, fname)
    geo_path = os.path.join(out_det_geo, fname)
    # 기존 결과 존재 여부 확인 및 불완전 파일 정리
    rgb_exists = os.path.exists(rgb_path)
    geo_exists = os.path.exists(geo_path)
    if rgb_exists and os.path.getsize(rgb_path) < 100 * 1024:
        try:
            os.remove(rgb_path)
            print(f"  - 불완전 Color 출력 삭제: {fname}")
        except OSError:
            pass
        rgb_exists = False
    if geo_exists and os.path.getsize(geo_path) < 100 * 1024:
        try:
            os.remove(geo_path)
            print(f"  - 불완전 Geometry 출력 삭제: {fname}")
        except OSError:
            pass
        geo_exists = False
    try:
        # 1단계: Color 필터링
        if not rgb_exists:
            las = laspy.read(file_path)
            coords = np.vstack((las.x, las.y, las.z)).T
            total_points = coords.shape[0]
            print(f"  - 로드 완료: {total_points} points (파일: {fname})")
            # 색상 배열 추출
            try:
                r, g, b = las.red, las.green, las.blue
                colors = np.vstack((r, g, b)).T
            except AttributeError:
                colors = None
            # 색상 필터 적용
            t_start = datetime.now(timezone.utc)
            mask_color = filter_by_color(colors)
            t_end = datetime.now(timezone.utc)
            if mask_color is None:
                coords_color = coords
                colors_color = colors
            else:
                coords_color = coords[mask_color]
                colors_color = colors[mask_color] if colors is not None else None
            # 로그 및 출력
            log_step(fname, "Detect", "ColorFilter", t_start, t_end)
            print(f"  - Color 필터: {coords_color.shape[0]} / {total_points} 점 유지 (지정 색상 범위)")
            save_las_points(las, coords_color, colors_color, rgb_path)
        else:
            # 이전 Color 필터 결과 로드
            las_color = laspy.read(rgb_path)
            coords_color = np.vstack((las_color.x, las_color.y, las_color.z)).T
            try:
                r, g, b = las_color.red, las_color.green, las_color.blue
                colors_color = np.vstack((r, g, b)).T
            except AttributeError:
                colors_color = None
            total_points = coords_color.shape[0]
            print(f"  - Color 단계 결과 로드: {total_points} points (이미 필터 완료)")
        # 2단계: Geometry 필터링 (Color 결과에 적용)
        if not geo_exists:
            t_start = datetime.now(timezone.utc)
            mask_geo = filter_by_ground(coords_color)
            t_end = datetime.now(timezone.utc)
            coords_geo = coords_color[mask_geo]
            colors_geo = colors_color[mask_geo] if colors_color is not None else None
            log_step(fname, "Detect", "GeometryFilter", t_start, t_end)
            print(f"  - Geometry 필터: {coords_geo.shape[0]} / {coords_color.shape[0]} 점 유지 (지면 위 객체)")
            # 결과 저장 (템플릿 LAS는 가능하면 기존 객체 활용)
            template_las = las if 'las' in locals() else las_color
            save_las_points(template_las, coords_geo, colors_geo, geo_path)
        else:
            # 이전 Geometry 필터 결과 로드
            las_geo = laspy.read(geo_path)
            coords_geo = np.vstack((las_geo.x, las_geo.y, las_geo.z)).T
            try:
                r, g, b = las_geo.red, las_geo.green, las_geo.blue
                colors_geo = np.vstack((r, g, b)).T
            except AttributeError:
                colors_geo = None
            print(f"  - Geometry 단계 결과 로드: {coords_geo.shape[0]} points (이미 필터 완료)")
    except Exception as e:
        print(f"[오류] Detect 필터 처리 실패: {fname} - {e}")
    finally:
        # 메모리 해제
        try:
            if 'las' in locals():
                las.close()
            if 'las_color' in locals():
                las_color.close()
        except:
            pass
        try:
            del las, las_color, coords, coords_color, coords_geo, colors, colors_color, colors_geo
        except NameError:
            pass
    # 빈 출력 파일 정리 (포인트가 없으면 파일 삭제)
    for path in [rgb_path, geo_path]:
        if os.path.exists(path) and os.path.getsize(path) < 100 * 1024:
            try:
                os.remove(path)
                print(f"    -> 결과 포인트 없음, 출력 파일 삭제: {os.path.basename(path)}")
            except OSError:
                pass
    print(f"=== [Detect] 파일 처리 완료: {fname} ===")



# 2) 메인 블록
# ─────────────────────────────────────────
if __name__ == "__main__":
    delete_small_files(original_dir)
    delete_small_files(detect_dir)

    orig_files = [os.path.join(original_dir, f)
                  for f in os.listdir(original_dir) if f.lower().endswith(".las")]
    det_files  = [os.path.join(detect_dir,  f)
                  for f in os.listdir(detect_dir)  if f.lower().endswith(".las")]
    orig_files.sort(); det_files.sort()

    # ───────────────── Geometry 스킵 여부 확인 ─────────────────
    existing_geo = [os.path.join(out_orig_geo, f)
                    for f in os.listdir(out_orig_geo)
                    if f.lower().endswith(".las")]
    total_geo_bytes = sum(os.path.getsize(p) for p in existing_geo)

    if (len(existing_geo) == EXPECTED_FILE_COUNT and
        total_geo_bytes >= SIZE_THRESHOLD_GB * 1024**3):
        print(f"[Info] Original Geometry 폴더에 "
              f"{len(existing_geo)}개 / {total_geo_bytes/1024**3:.1f} GB "
              f"이미 존재 → Geometry 단계 스킵")
        orig_files = []       # Geometry 처리 submit 생략
    else:
        print(f"[Info] Original Geometry 단계 실행 "
              f"(현재 {len(existing_geo)}개 / {total_geo_bytes/1024**3:.1f} GB)")

    # ───────────────── 쓰레드 풀 실행 ─────────────────
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        fut = [ex.submit(process_original_file, f) for f in orig_files] + \
              [ex.submit(process_detect_file,   f) for f in det_files]
        for f in as_completed(fut):
            f.result()


    # 처리 로그 CSV 저장
    log_path = "25.06.23_Step2_V1_log.csv"
    try:
        import csv
        with open(log_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["File", "Pipeline", "Step", "StartUTC", "EndUTC", "DurationSec"])
            for entry in sorted(log_entries, key=lambda x: x[0]):
                writer.writerow(entry)
        print(f"[Info] 전처리 로그 저장 완료: {log_path}")
    except Exception as e:
        print(f"[경고] 로그 저장 실패: {e}")