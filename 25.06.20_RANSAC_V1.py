import traceback

def process_detect_directory(input_dir, output_dir_geo_rgb, output_dir_rgb_geo):
    """Detect 디렉터리 처리: (Geo→RGB)와 (RGB→Geo) 두 가지 파이프라인 실행."""
    os.makedirs(output_dir_geo_rgb, exist_ok=True)
    os.makedirs(output_dir_rgb_geo, exist_ok=True)
    # 작은 파일 삭제
    delete_small_files(input_dir)
    # 처리 대상 파일 목록
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".las")]
    for filename in files:
        file_path = os.path.join(input_dir, filename)
        print(f"\n=== [Detect] 파일 처리 시작: {filename} ===")
        try:
            las, coords, colors = load_las_points(file_path)
            total_points = coords.shape[0]
            print(f"로드 완료: {total_points} points (파일: {filename})")
        except Exception as e:
            print(f"[오류] 파일 로드 실패: {filename} - {e}")
            traceback.print_exc()
            continue

        # 1) Geometry -> RGB 순서
        try:
            # 지면 제거
            mask_geo = filter_by_ground(coords)
            coords_geo = coords[mask_geo]
            colors_geo = colors[mask_geo] if colors is not None else None
            print(f"[Geo 필터] 지면 제거 후 남은 점: {coords_geo.shape[0]} / {total_points}")
            # 색상 필터
            mask_color = filter_by_color(coords_geo, colors_geo)
            coords_geo_rgb = coords_geo[mask_color]
            colors_geo_rgb = colors_geo[mask_color] if colors_geo is not None else None
            print(f"[RGB 필터] 색상 필터 후 남은 점: {coords_geo_rgb.shape[0]} / {coords_geo.shape[0]}")
            # RANSAC 선 추출
            powerline_coords = extract_powerlines(coords_geo_rgb)
            # 색상 정보도 동일 인덱스로 추출 (인라이어만 저장)
            if colors_geo_rgb is not None and powerline_coords.shape[0] > 0:
                # Boolean mask를 사용할 수 없으므로 좌표 기준으로 일치하는 index 추출
                # (전선 좌표 집합이 원래 순서와 다를 수 있어 matching이 필요함)
                # 여기서는 단순히 동일한 순서로 간주하여 자를 수도 있지만, 안전하게 set 이용
                coord_set = {tuple(pt) for pt in powerline_coords}
                mask_power = np.array([tuple(pt) in coord_set for pt in coords_geo_rgb])
                powerline_colors = colors_geo_rgb[mask_power]
            else:
                powerline_colors = None
            output_path = os.path.join(output_dir_geo_rgb, filename)
            save_las_points(las, powerline_coords, (powerline_colors if powerline_colors is not None else None), output_path)
        except Exception as e:
            print(f"[오류] 파일 처리 실패 (Geo->RGB): {filename} - {e}")
            traceback.print_exc()

        # 2) RGB -> Geometry 순서
        try:
            # 색상 필터 먼저
            mask_color2 = filter_by_color(coords, colors)
            coords_color = coords[mask_color2]
            colors_color = colors[mask_color2] if colors is not None else None
            print(f"[RGB 필터] 색상 필터 후 남은 점: {coords_color.shape[0]} / {total_points}")
            # 지면 제거
            mask_geo2 = filter_by_ground(coords_color)
            coords_rgb_geo = coords_color[mask_geo2]
            colors_rgb_geo = colors_color[mask_geo2] if colors_color is not None else None
            print(f"[Geo 필터] 지면 제거 후 남은 점: {coords_rgb_geo.shape[0]} / {coords_color.shape[0]}")
            # RANSAC 선 추출
            powerline_coords2 = extract_powerlines(coords_rgb_geo)
            if colors_rgb_geo is not None and powerline_coords2.shape[0] > 0:
                coord_set2 = {tuple(pt) for pt in powerline_coords2}
                mask_power2 = np.array([tuple(pt) in coord_set2 for pt in coords_rgb_geo])
                powerline_colors2 = colors_rgb_geo[mask_power2]
            else:
                powerline_colors2 = None
            output_path2 = os.path.join(output_dir_rgb_geo, filename)
            save_las_points(las, powerline_coords2, (powerline_colors2 if powerline_colors2 is not None else None), output_path2)
        except Exception as e:
            print(f"[오류] 파일 처리 실패 (RGB->Geo): {filename} - {e}")
            traceback.print_exc()
        print(f"=== [Detect] 파일 처리 완료: {filename} ===")

def process_original_directory(input_dir, output_dir_geo):
    """Original 디렉터리 처리: Geometry 필터링 후 RANSAC."""
    os.makedirs(output_dir_geo, exist_ok=True)
    delete_small_files(input_dir)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".las")]
    for filename in files:
        file_path = os.path.join(input_dir, filename)
        print(f"\n=== [Original] 파일 처리 시작: {filename} ===")
        try:
            las, coords, colors = load_las_points(file_path)
            total_points = coords.shape[0]
            print(f"로드 완료: {total_points} points (파일: {filename})")
        except Exception as e:
            print(f"[오류] 파일 로드 실패: {filename} - {e}")
            traceback.print_exc()
            continue
        try:
            # 지면 제거 (Original 데이터는 RGB 필터 없음)
            mask_geo = filter_by_ground(coords)
            coords_geo = coords[mask_geo]
            colors_geo = colors[mask_geo] if colors is not None else None
            print(f"[Geo 필터] 지면 제거 후 남은 점: {coords_geo.shape[0]} / {total_points}")
            # RANSAC 선 추출
            powerline_coords = extract_powerlines(coords_geo)
            # colors_geo에서 해당 전선점 색상 추출 (원본 데이터의 색상은 분석에 사용 안 하지만 저장은 가능)
            if colors_geo is not None and powerline_coords.shape[0] > 0:
                coord_set = {tuple(pt) for pt in powerline_coords}
                mask_power = np.array([tuple(pt) in coord_set for pt in coords_geo])
                powerline_colors = colors_geo[mask_power]
            else:
                powerline_colors = None
            output_path = os.path.join(output_dir_geo, filename)
            save_las_points(las, powerline_coords, (powerline_colors if powerline_colors is not None else None), output_path)
        except Exception as e:
            print(f"[오류] 파일 처리 실패 (Original): {filename} - {e}")
            traceback.print_exc()
        print(f"=== [Original] 파일 처리 완료: {filename} ===")
