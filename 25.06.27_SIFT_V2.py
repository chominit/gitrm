# -*- coding: utf-8 -*-
# 필요한 패키지 임포트
import os
import cv2
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import csv
import sys

# 원하는 버퍼 크기 설정 (약 10GB) - tiepoints.csv 작성 시 메모리 사용을 제어하기 위한 버퍼 크기
BUFFER_SIZE_BYTES = 10 * 1024**3  # 10GB

# tiepoint 버퍼를 파일에 기록하는 보조 함수 (버퍼 비우기)
def flush_buffer(writer, buffer):
    """tiepoint_buffer에 쌓인 데이터를 CSV 파일에 기록하고 버퍼를 비우는 함수."""
    writer.writerows(buffer)
    buffer.clear()

# ----- 경로 설정 -----
# 입력 데이터셋 경로 (Original B 및 Original C 폴더들의 상위 폴더)
INPUT_ROOT = Path(r"C:\Users\jscool\datasets\Compared_Color_Result_PNG")
# 출력 루트 경로 (결과 CSV 및 이미지가 저장될 경로)
OUTPUT_ROOT = INPUT_ROOT / "Results"
# 처리 대상 이미지 파일 확장자
IMG_EXT = ".png"

# SIFT 검출기 생성 함수 
# (주의: cv2.SIFT_create는 멀티스레드 환경에서 thread-safe하지 않으므로 각 프로세스 내에서 생성해야 함)
def create_sift_detector():
    """SIFT 특징 검출기를 초기화하여 반환 (각 프로세스마다 독립적으로 생성)."""
    # 기본 설정으로 SIFT 생성 (nfeatures=0은 특징점 개수 제한 없음)
    return cv2.SIFT_create(
        nfeatures=0, nOctaveLayers=3,
        contrastThreshold=0.04, edgeThreshold=10, sigma=1.6
    )

# Color mask 설정 (method1, method2에서 사용) - 목표 RGB 색상=(255, 0, 191) ± tolerance=10
COLOR_TARGET = (255, 0, 191)  # (R, G, B) 목표 색상
COLOR_TOL = 10                # 허용 오차 범위
# OpenCV는 BGR 순서를 사용하므로, BGR 색 공간에서도 동일한 범위를 지정
# (R,G,B)=(255,0,191)에 대응하는 BGR은 (191,0,255)
BGR_LOWER = np.array([
    max(0, 191 - COLOR_TOL),
    max(0,   0 - COLOR_TOL),
    max(0, 255 - COLOR_TOL)
], dtype=np.uint8)
BGR_UPPER = np.array([
    min(255, 191 + COLOR_TOL),
    min(255,   0 + COLOR_TOL),
    min(255, 255 + COLOR_TOL)
], dtype=np.uint8)

def process_image(method, img_path_str):
    """
    단일 이미지를 처리하여 SIFT 특징점을 추출하는 함수.
    method: "baseline", "method1", "method2", "method3" 중 하나를 지정.
    img_path_str: 처리할 이미지 파일 경로 (문자열로 전달).
    """
    img_path = Path(img_path_str)
    # 이미지를 컬러(BGR)와 그레이스케일로 읽기
    color_img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    gray_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if color_img is None or gray_img is None:
        return None  # 이미지 로드 실패 시 None 반환하여 스킵

    # SIFT 검출기 생성 (각 프로세스마다 자체적으로 생성해서 사용)
    sift = create_sift_detector()

    # 키포인트와 디스크립터 검출
    start_time = time.time()  # 처리 시작 시각
    kp = []
    des = None

    if method == "baseline":
        # 방법 ①: 전체 이미지에 대해 SIFT 특징점 검출 (마스크 없음, 그레이스케일 이미지 사용)
        kp, des = sift.detectAndCompute(gray_img, mask=None)

    elif method == "method1":
        # 방법 ②: 지정된 RGB 색상 마스크 영역에 대해서만 SIFT 검출
        # 컬러 이미지에서 목표 색상 영역 마스크 생성
        mask_region = cv2.inRange(color_img, BGR_LOWER, BGR_UPPER)
        # 생성된 마스크를 사용하여 해당 영역에서만 특징점 검출
        kp, des = sift.detectAndCompute(gray_img, mask=mask_region)

    elif method == "method2":
        # 방법 ③: ROI (관심 영역) 이미지 크롭하여 SIFT 검출
        # 1) method1과 동일하게 색상 마스크 생성
        mask_region = cv2.inRange(color_img, BGR_LOWER, BGR_UPPER)
        # 2) 마스크로부터 관심 영역의 bounding box 계산
        ys, xs = np.nonzero(mask_region)
        if len(xs) == 0 or len(ys) == 0:
            # 마스크 영역이 없으면 전체 이미지에서 검출 (fallback 처리)
            kp, des = sift.detectAndCompute(gray_img, mask=None)
        else:
            x_min, x_max = np.min(xs), np.max(xs)
            y_min, y_max = np.min(ys), np.max(ys)
            # ROI 영역으로 그레이스케일 이미지 잘라내기
            roi_gray = gray_img[y_min:y_max+1, x_min:x_max+1]
            kp, des = sift.detectAndCompute(roi_gray, mask=None)
            # 검출된 키포인트 좌표를 원본 이미지 기준으로 변환 (ROI 위치만큼 offset 적용)
            for i in range(len(kp)):
                x, y = kp[i].pt  # ROI 내에서의 좌표 (float형)
                kp[i].pt = (x + x_min, y + y_min)  # 원본 이미지 좌표로 변환

    elif method == "method3":
        # 방법 ④: HSV 색상공간에서 주요 색상 영역에 대해서만 SIFT 검출
        # BGR 이미지를 HSV 색상공간으로 변환
        hsv_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv_img)
        # 포화도(S)와 명도(V)가 일정 임계값 이상인 픽셀만 유효한 색상 후보로 간주
        sat_thr = 50  # Saturation 임계값
        val_thr = 50  # Value 임계값
        valid_mask = (s_channel >= sat_thr) & (v_channel >= val_thr)
        if np.any(valid_mask):
            # 유효 픽셀에 대해 Hue 채널의 히스토그램 계산하여 가장 많은 색상(hue) 추출
            hist = cv2.calcHist([h_channel], [0], valid_mask.astype(np.uint8), [180], [0, 180])
            dominant_hue = int(np.argmax(hist))  # 히스토그램에서 빈도수가 가장 높은 Hue 값
        else:
            # 유효한 색상 픽셀이 거의 없는 경우 전체 픽셀 기준으로 Hue 히스토그램 계산
            hist = cv2.calcHist([h_channel], [0], None, [180], [0, 180])
            dominant_hue = int(np.argmax(hist))
        # dominant_hue를 중심으로 ±10 범위의 hue 값을 관심 범위로 설정 (HSV 특성상 0과 179 사이 랩어라운드 처리)
        hue_tol = 10
        h_lower = dominant_hue - hue_tol
        h_upper = dominant_hue + hue_tol
        if h_lower < 0:
            # 범위가 음수인 경우 (예: dominant_hue=5, h_lower=-5) -> [0, h_upper]와 [180+h_lower, 179] 두 구간
            h_range_mask = ((h_channel >= (h_lower + 180)) | (h_channel <= h_upper))
        elif h_upper > 179:
            # 범위가 179를 초과하는 경우 (예: dominant_hue=175, h_upper=185) -> [h_lower, 179]와 [0, h_upper-180] 두 구간
            h_range_mask = ((h_channel >= h_lower) | (h_channel <= (h_upper - 180)))
        else:
            # 일반적인 경우: 단일 연속 구간 [h_lower, h_upper] 사용
            h_range_mask = ((h_channel >= h_lower) & (h_channel <= h_upper))
        # 최종 관심 영역 마스크: 색상 범위 필터(h_range_mask) AND (포화도/명도 임계조건)
        dom_mask = (h_range_mask & (s_channel >= sat_thr) & (v_channel >= val_thr)).astype(np.uint8) * 255
        if cv2.countNonZero(dom_mask) == 0:
            # 만약 해당 범위에 유효한 픽셀이 없다면 전체 이미지에서 검출
            kp, des = sift.detectAndCompute(gray_img, mask=None)
        else:
            # 생성된 마스크 영역에서만 검출 수행
            kp, des = sift.detectAndCompute(gray_img, mask=dom_mask)
    else:
        return None  # 정의되지 않은 method 인자가 들어온 경우 안전하게 None 반환

    elapsed = time.time() - start_time  # 처리 시간 (초)
    # 특징점 시각화 이미지를 생성 (키포인트를 원본 그레이스케일 이미지에 그림)
    vis_img = cv2.drawKeypoints(
        gray_img, kp, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    # 결과 저장을 위한 경로 설정 (e.g., "Original B/filename.png")
    rel_path = img_path.parent.name + "/" + img_path.name
    stem = img_path.stem
    out_dir = OUTPUT_ROOT / img_path.parent.name / method
    out_dir.mkdir(parents=True, exist_ok=True)  # 출력 디렉토리 생성 (이미 존재하면 넘어감)
    # 출력 파일 경로들 설정
    vis_path = out_dir / f"{stem}_kp.png"
    desc_path = out_dir / f"{stem}_des.npy"
    kpt_path = out_dir / f"{stem}_kpts.npy"
    # 키포인트 시각화 이미지와 디스크립터, 키포인트 좌표 저장
    cv2.imwrite(str(vis_path), vis_img)
    np.save(str(desc_path), des)
    kpt_coords = np.array([kp_i.pt for kp_i in kp], dtype=np.float32)
    np.save(str(kpt_path), kpt_coords)
    # 처리 결과 정보 반환 (파일 경로, 키포인트 개수, 처리 시간)
    return {
        "file": rel_path,
        "num_kp": len(kp),
        "time": elapsed
    }

if __name__ == "__main__":
    mp.freeze_support()  # Windows 멀티프로세싱 실행 시의 안전 가드

    # OpenCV 및 NumPy 내부 연산에 대한 최적화 설정
    NUM_CORES = 24
    cv2.setUseOptimized(True)
    cv2.setNumThreads(NUM_CORES)
    os.environ["OMP_NUM_THREADS"] = str(NUM_CORES)
    os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_CORES)
    # (필요 시) MKL 등의 스레드 환경 변수도 설정 가능:
    # os.environ["MKL_NUM_THREADS"] = str(NUM_CORES)

    methods = ["baseline", "method1", "method2", "method3"]
    # Original B 와 Original C 두 폴더에 대해 순차적으로 처리
    for folder in ["Original B", "Original C"]:
        input_dir = INPUT_ROOT / folder
        if not input_dir.exists():
            print(f"Input folder not found: {input_dir}")
            continue  # 입력 폴더가 없으면 스킵
        # 대상 폴더의 모든 이미지 목록 가져오기
        img_list = sorted([str(p) for p in input_dir.glob(f"*{IMG_EXT}")])
        if not img_list:
            print(f"No {IMG_EXT} images in {input_dir}")
            continue  # 이미지가 없는 경우 스킵

        # 출력 폴더 구조 생성: 각 method별 하위 폴더 미리 생성
        for method in methods:
            (OUTPUT_ROOT / folder / method).mkdir(parents=True, exist_ok=True)

        # 각 처리 방법(method)에 대해 특징점 추출 및 매칭 수행
        for method in methods:
            # 이미 이전에 해당 폴더-방법으로 처리된 결과가 충분히 존재하면 건너뛰기
            summary_file = OUTPUT_ROOT / folder / method / "summary.csv"
            tiepoints_file = OUTPUT_ROOT / folder / method / "tiepoints.csv"
            if summary_file.exists() and summary_file.stat().st_size >= 30 * 1024 and \
               tiepoints_file.exists() and tiepoints_file.stat().st_size >= 1 * 1024**3:
                print(f"\n=== Skipping {folder} - {method} (already processed) ===")
                continue

            print(f"\n=== Processing {folder} - {method} ===")
            start_time = time.time()
            results = []
            # 이미지를 병렬 처리하여 SIFT 특징점 추출 (프로세스 풀 사용)
            with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
                futures = [executor.submit(process_image, method, img_path) for img_path in img_list]
                for f in tqdm(as_completed(futures), total=len(futures), desc=f"{method}"):
                    res = f.result()
                    if res:
                        results.append(res)
            # 이미지 파일명 기준으로 결과 정렬 (순서를 일정하게 유지)
            results.sort(key=lambda x: x["file"])

            # 각 이미지에 대한 디스크립터와 키포인트 좌표 로드
            descriptors = []
            keypoints_coords = []
            image_files = []
            for res in results:
                rel_path = res["file"]  # 예: "Original B/파일명.png"
                image_files.append(rel_path)
                folder_name = rel_path.split("/")[0]  # "Original B" or "Original C"
                file_name = Path(rel_path).stem      # 파일 이름 (확장자 제외)
                # 해당 이미지의 descripter 및 keypoint 파일 경로
                desc_path = OUTPUT_ROOT / folder_name / method / f"{file_name}_des.npy"
                kpt_path = OUTPUT_ROOT / folder_name / method / f"{file_name}_kpts.npy"
                # numpy 배열 로드 (allow_pickle=True: None 등 객체가 저장된 경우에도 불러오기 위해)
                des = np.load(str(desc_path), allow_pickle=True)
                kpts = np.load(str(kpt_path), allow_pickle=True)
                descriptors.append(des)
                keypoints_coords.append(kpts)

            # BFMatcher 초기화 (L2 거리, crossCheck=False로 KNN 매칭 사용)
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            num_images = len(image_files)
            matched_images_count = [0] * num_images  # 각 이미지별 매칭된 타 이미지 개수
            tiepoint_count = [0] * num_images        # 각 이미지별 총 tiepoint 개수

            # tiepoints 결과를 저장할 CSV 파일 오픈
            tiepoints_csv_path = OUTPUT_ROOT / folder / method / "tiepoints.csv"
            with open(tiepoints_csv_path, "w", encoding="utf-8", newline="") as tf:
                writer = csv.writer(tf)
                writer.writerow(["img1", "x1", "y1", "img2", "x2", "y2"])  # 헤더 작성
                tiepoint_buffer = []      # 메모리에 저장할 tiepoint 버퍼
                current_buffer_size = 0   # 현재 버퍼에 저장된 데이터의 메모리 크기 추적

                total_pairs = num_images * (num_images - 1) // 2  # 이미지 쌍(pair)의 총 개수
                pbar = tqdm(total=total_pairs, desc=f"{method} matching", leave=False)
                # 모든 이미지 쌍에 대해 매칭 수행
                for i in range(num_images):
                    for j in range(i + 1, num_images):
                        des1 = descriptors[i]
                        des2 = descriptors[j]
                        # 디스크립터가 None이거나 올바른 numpy 배열이 아닌 경우, 또는 비어있는 경우 매칭 생략
                        if des1 is None or not isinstance(des1, np.ndarray) or des1.dtype == object or des1.size == 0 or \
                           des2 is None or not isinstance(des2, np.ndarray) or des2.dtype == object or des2.size == 0:
                            pbar.update(1)
                            continue
                        # KNN 매칭 (k=2) 수행
                        matches = bf.knnMatch(des1, des2, k=2)
                        good_matches = []
                        # Lowe's ratio test 적용하여 좋은 매칭점 선별
                        for m, n in matches:
                            if m.distance < 0.75 * n.distance:
                                good_matches.append(m)
                        if good_matches:
                            # 유의미한 매칭이 존재하는 경우 각 이미지의 매칭 횟수와 tiepoint 개수 갱신
                            matched_images_count[i] += 1
                            matched_images_count[j] += 1
                            tiepoint_count[i] += len(good_matches)
                            tiepoint_count[j] += len(good_matches)
                            img1_name = image_files[i].split("/")[-1]
                            img2_name = image_files[j].split("/")[-1]
                            coords1 = keypoints_coords[i]
                            coords2 = keypoints_coords[j]
                            # good_matches에 포함된 각 매칭점 쌍의 좌표를 tiepoint 버퍼에 추가
                            for m in good_matches:
                                pt1 = coords1[m.queryIdx]
                                pt2 = coords2[m.trainIdx]
                                x1, y1 = float(pt1[0]), float(pt1[1])
                                x2, y2 = float(pt2[0]), float(pt2[1])
                                row = [img1_name, f"{x1:.2f}", f"{y1:.2f}", img2_name, f"{x2:.2f}", f"{y2:.2f}"]
                                tiepoint_buffer.append(row)
                                # 현재 row의 추정 메모리 크기를 누적 (버퍼 크기 관리용)
                                current_buffer_size += sys.getsizeof(row)
                            # 버퍼가 일정 크기 이상 커지면 파일에 기록(flush) 후 버퍼 초기화
                            if current_buffer_size >= BUFFER_SIZE_BYTES:
                                flush_buffer(writer, tiepoint_buffer)
                                current_buffer_size = 0
                        # 진행 상황 업데이트
                        pbar.update(1)
                pbar.close()
                # 루프 종료 후, 버퍼에 남아있는 tiepoint 데이터가 있으면 모두 기록
                if tiepoint_buffer:
                    flush_buffer(writer, tiepoint_buffer)
                    current_buffer_size = 0

            # summary.csv 파일 작성 (이미지별 특징점 및 매칭 요약)
            summary_csv_path = OUTPUT_ROOT / folder / method / "summary.csv"
            with open(summary_csv_path, "w", encoding="utf-8", newline="") as sf:
                sf.write("image,keypoints,matched_images,tiepoints,time(sec)\n")
                total_kp = 0
                total_tie = 0
                # 각 이미지별로 키포인트 개수, 매칭된 이미지 수, tiepoint 총합, 처리 시간을 작성
                for idx, res in enumerate(results):
                    fname = Path(res["file"]).name   # 이미지 파일 이름 (폴더 제외)
                    num_kp = res["num_kp"]           # 검출된 키포인트 수
                    mi_count = matched_images_count[idx]  # 해당 이미지와 매칭된 다른 이미지 개수
                    tp_count = tiepoint_count[idx]        # 해당 이미지의 총 tiepoint 수
                    time_sec = res["time"]           # 해당 이미지 처리에 걸린 시간
                    total_kp += num_kp
                    total_tie += tp_count
                    sf.write(f"{fname},{num_kp},{mi_count},{tp_count},{time_sec:.4f}\n")
                total_time = time.time() - start_time  # 해당 method 전체 처리에 걸린 시간
                # 총계 행 추가 (TOTAL 행: 총 키포인트 수, 총 tiepoint 수, 전체 소요 시간)
                sf.write(f"TOTAL,{total_kp},,{total_tie},{total_time:.2f}\n")
            # 현재 method 처리 완료 로그 출력
            print(f"Completed {folder}-{method}: {len(results)} images, time {total_time:.2f} sec")
