# -*- coding: utf-8 -*-
"""
part1_compute_diff_pixels.py
원본 이미지와 병합된(segmentation) 이미지의 차분 계산
변경된 픽셀 위치만 저장
멀티프로세싱으로 고속 처리
"""

from pathlib import Path
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

# 사이트별 경로 설정
SITE_PATHS = {
    "Site_A": {
        "raw": Path(r"F:\Images\Raw_Image_Data\Site_A_Original"),
        "merged": Path(r"F:\Images\병합된 이미지\Zenmuse_AI_Site_A"),
        "output": Path(r"F:\Images\차분된 이미지\Zenmuse_AI_Site_A")
    },
    "Site_B": {
        "raw": Path(r"F:\Images\Raw_Image_Data\Site_B_Original"),
        "merged": Path(r"F:\Images\병합된 이미지\Zenmuse_AI_Site_B"),
        "output": Path(r"F:\Images\차분된 이미지\Zenmuse_AI_Site_B")
    },
    "Site_C": {
        "raw": Path(r"F:\Images\Raw_Image_Data\Site_C_Original"),
        "merged": Path(r"F:\Images\병합된 이미지\Zenmuse_AI_Site_C"),
        "output": Path(r"F:\Images\차분된 이미지\Zenmuse_AI_Site_C")
    }
}

def compute_pixel_diff(raw_img_path: Path, merged_img_path: Path, threshold: int = 1) -> np.ndarray:
    """
    두 이미지의 픽셀 차분 계산

    Args:
        raw_img_path: 원본 이미지 경로
        merged_img_path: 병합된(segmentation) 이미지 경로
        threshold: RGB 차이 임계값 (각 채널당)

    Returns:
        changed_pixels: (N, 2) 변경된 픽셀 좌표 [x, y]
    """
    # 이미지 로드
    raw_img = Image.open(raw_img_path).convert('RGB')
    merged_img = Image.open(merged_img_path).convert('RGB')

    # numpy 배열로 변환
    raw_array = np.array(raw_img, dtype=np.int16)
    merged_array = np.array(merged_img, dtype=np.int16)

    # 크기 확인
    if raw_array.shape != merged_array.shape:
        print(f"[WARNING] 이미지 크기 불일치: {raw_img_path.name}")
        print(f"   Raw: {raw_array.shape}, Merged: {merged_array.shape}")
        return np.empty((0, 2), dtype=np.int32)

    # RGB 차이 계산
    diff = np.abs(raw_array - merged_array)

    # 각 픽셀에서 최대 채널 차이
    max_diff = np.max(diff, axis=2)

    # 임계값 이상인 픽셀 찾기
    changed_mask = max_diff > threshold

    # 변경된 픽셀 좌표 (y, x)
    y_coords, x_coords = np.where(changed_mask)

    # (x, y) 형식으로 변환
    changed_pixels = np.column_stack([x_coords, y_coords]).astype(np.int32)

    return changed_pixels


def process_single_image(args):
    """
    단일 이미지 처리 (멀티프로세싱용 헬퍼)

    Args:
        args: (merged_path, raw_dir, output_dir, threshold, overwrite)

    Returns:
        (success, num_pixels, num_changed, img_name)
    """
    merged_path, raw_dir, output_dir, threshold, overwrite = args

    img_name = merged_path.name
    raw_path = raw_dir / img_name
    output_file = output_dir / f"{merged_path.stem}_pixels.npy"

    # 이미 존재하고 덮어쓰기 않으면 스킵
    if output_file.exists() and not overwrite:
        return (False, 0, 0, img_name)

    # 원본 이미지 존재 확인
    if not raw_path.exists():
        print(f"\n[WARNING] 원본 이미지 없음: {img_name}")
        return (False, 0, 0, img_name)

    try:
        # 픽셀 차분 계산
        changed_pixels = compute_pixel_diff(raw_path, merged_path, threshold)

        # 이미지 크기 (통계용)
        img = Image.open(merged_path)
        width, height = img.size
        total_pixels = width * height

        # 변경된 픽셀 수
        num_changed = len(changed_pixels)

        # NPY 파일로 저장
        np.save(output_file, changed_pixels)

        return (True, total_pixels, num_changed, img_name)

    except Exception as e:
        print(f"\n[ERROR] {img_name} 처리 실패: {e}")
        return (False, 0, 0, img_name)


def process_site(site_name: str, threshold: int = 1, overwrite: bool = False, num_workers: int = None):
    """
    사이트별 모든 이미지의 픽셀 차분 계산 (멀티프로세싱)

    Args:
        site_name: "Site_A", "Site_B", "Site_C"
        threshold: RGB 차이 임계값
        overwrite: 기존 결과 덮어쓰기 여부
        num_workers: 워커 프로세스 수 (기본: CPU 코어 수)
    """
    if site_name not in SITE_PATHS:
        print(f"[ERROR] 알 수 없는 사이트: {site_name}")
        return

    paths = SITE_PATHS[site_name]
    raw_dir = paths["raw"]
    merged_dir = paths["merged"]
    output_dir = paths["output"]

    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    # 병합된 이미지 목록 (이게 처리 대상)
    merged_images = sorted(merged_dir.glob("*.JPG"))
    if not merged_images:
        merged_images = sorted(merged_dir.glob("*.jpg"))

    if not merged_images:
        print(f"[ERROR] 병합된 이미지 없음: {merged_dir}")
        return

    # 워커 수 결정
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # CPU 코어 - 1

    print(f"\n{'='*70}")
    print(f"[{site_name}] 픽셀 차분 계산 시작 (멀티프로세싱)")
    print(f"{'='*70}")
    print(f"원본 이미지: {raw_dir}")
    print(f"병합된 이미지: {merged_dir}")
    print(f"출력 경로: {output_dir}")
    print(f"이미지 수: {len(merged_images)}개")
    print(f"임계값: {threshold}")
    print(f"워커 수: {num_workers}개")
    print()

    # 멀티프로세싱 인자 준비
    process_args = [
        (merged_path, raw_dir, output_dir, threshold, overwrite)
        for merged_path in merged_images
    ]

    # 통계
    total_pixels = 0
    total_changed = 0
    processed = 0
    skipped = 0

    # 멀티프로세싱으로 처리
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_image, process_args),
            total=len(merged_images),
            desc=f"[{site_name}] Processing"
        ))

    # 결과 집계
    for success, num_pixels, num_changed, img_name in results:
        if success:
            total_pixels += num_pixels
            total_changed += num_changed
            processed += 1
        else:
            if num_pixels == 0 and num_changed == 0:
                skipped += 1

    # 결과 출력
    print(f"\n{'='*70}")
    print(f"[{site_name}] 처리 완료")
    print(f"{'='*70}")
    print(f"처리 완료: {processed}개")
    print(f"스킵: {skipped}개")
    print(f"전체 픽셀: {total_pixels:,}")
    print(f"변경된 픽셀: {total_changed:,} ({100*total_changed/total_pixels:.2f}%)")
    print(f"평균 변경 픽셀/이미지: {total_changed/processed:.0f}")
    print()

    # 메타데이터 저장
    meta_file = output_dir / "metadata.json"
    metadata = {
        "site_name": site_name,
        "raw_dir": str(raw_dir),
        "merged_dir": str(merged_dir),
        "threshold": threshold,
        "total_images": processed,
        "total_pixels": total_pixels,
        "total_changed_pixels": total_changed,
        "change_ratio": total_changed / total_pixels if total_pixels > 0 else 0,
        "avg_changed_per_image": total_changed / processed if processed > 0 else 0
    }

    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[INFO] 메타데이터 저장: {meta_file}")


def main():
    parser = argparse.ArgumentParser(description="원본/병합 이미지 픽셀 차분 계산 (멀티프로세싱)")
    parser.add_argument("--sites", nargs="+", choices=["Site_A", "Site_B", "Site_C", "all"],
                        default=["all"], help="처리할 사이트")
    parser.add_argument("--threshold", type=int, default=1,
                        help="RGB 차이 임계값 (기본: 1)")
    parser.add_argument("--overwrite", action="store_true",
                        help="기존 결과 덮어쓰기")
    parser.add_argument("--workers", type=int, default=None,
                        help="워커 프로세스 수 (기본: CPU 코어 - 1)")

    args = parser.parse_args()

    # 처리할 사이트 결정
    if "all" in args.sites:
        sites_to_process = ["Site_A", "Site_B", "Site_C"]
    else:
        sites_to_process = args.sites

    # 각 사이트 처리
    for site in sites_to_process:
        process_site(site, threshold=args.threshold, overwrite=args.overwrite, num_workers=args.workers)

    print("\n" + "="*70)
    print("전체 처리 완료!")
    print("="*70)


if __name__ == "__main__":
    main()
