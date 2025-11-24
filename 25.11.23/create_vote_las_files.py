# -*- coding: utf-8 -*-
"""
create_vote_las_files.py
forward_votes.npy를 LAS 파일로 변환
Vote threshold별로 LAS 파일 생성 (X,Y,Z, RGB 16bit, Point ID 포함)
"""

from pathlib import Path
import numpy as np
import laspy
import constants as C
from typing import List

def create_las_from_votes(
    coords: np.ndarray,
    colors: np.ndarray,
    votes: np.ndarray,
    threshold: int,
    output_path: Path,
    point_ids: np.ndarray = None
) -> None:
    """
    Vote threshold 이상인 포인트만 LAS 파일로 저장

    Args:
        coords: (N, 3) XYZ 좌표
        colors: (N, 3) RGB 색상 (0-255)
        votes: (N,) vote counts
        threshold: vote 임계값
        output_path: 출력 LAS 파일 경로
        point_ids: (N,) 포인트 ID (optional)
    """
    # Vote 마스크
    mask = votes >= threshold
    num_points = mask.sum()

    if num_points == 0:
        print(f"   [SKIP] vote_{threshold}: 0 points")
        return

    # 필터링
    filtered_coords = coords[mask]
    filtered_colors = colors[mask]
    filtered_votes = votes[mask]

    if point_ids is not None:
        filtered_ids = point_ids[mask]
    else:
        # Point ID가 없으면 원본 인덱스 사용
        filtered_ids = np.where(mask)[0].astype(np.uint32)

    # LAS 1.2 헤더 생성
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(filtered_coords, axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])  # 1mm 정밀도

    # LAS 파일 생성
    las = laspy.LasData(header)

    # 좌표 설정
    las.x = filtered_coords[:, 0]
    las.y = filtered_coords[:, 1]
    las.z = filtered_coords[:, 2]

    # RGB 색상 설정 (16-bit, 0-65535 범위)
    # LAS spec: 8-bit RGB (0-255)를 16-bit (0-65535)로 변환
    # 변환 공식: RGB_16 = RGB_8 * 257 (또는 RGB_8 << 8 | RGB_8)
    las.red = (filtered_colors[:, 0].astype(np.uint16) * 257).astype(np.uint16)
    las.green = (filtered_colors[:, 1].astype(np.uint16) * 257).astype(np.uint16)
    las.blue = (filtered_colors[:, 2].astype(np.uint16) * 257).astype(np.uint16)

    # Point Source ID에 vote count 저장
    las.point_source_id = filtered_votes.astype(np.uint16)

    # User Data에 포인트 ID 저장 (32-bit를 8-bit로 분할)
    # LAS 1.2 format 3에는 user_data (8-bit) 필드가 있음
    # 전체 point_id는 별도 extra dimension으로 저장

    # Extra dimension 추가: original_point_id (32-bit unsigned int)
    las.add_extra_dim(laspy.ExtraBytesParams(
        name="point_id",
        type=np.uint32,
        description="Original point ID in full point cloud"
    ))
    las.point_id = filtered_ids

    # 파일 저장
    las.write(str(output_path))

    print(f"   [OK] vote_{threshold}.las: {num_points:,} points")
    print(f"        - X: [{filtered_coords[:,0].min():.2f}, {filtered_coords[:,0].max():.2f}]")
    print(f"        - Y: [{filtered_coords[:,1].min():.2f}, {filtered_coords[:,1].max():.2f}]")
    print(f"        - Z: [{filtered_coords[:,2].min():.2f}, {filtered_coords[:,2].max():.2f}]")
    print(f"        - Votes: [{filtered_votes.min()}, {filtered_votes.max()}]")


def process_site(site_name: str, thresholds: List[int] = None) -> None:
    """
    사이트별 forward_votes.npy를 LAS 파일로 변환

    Args:
        site_name: 사이트 이름
        thresholds: vote 임계값 리스트 (기본: [7, 15, 30])
    """
    if thresholds is None:
        thresholds = list(C.VOTE_THRESHOLDS)

    print(f"\n{'='*70}")
    print(f"Site: {site_name}")
    print(f"{'='*70}")

    # 경로 설정
    part2_dir = C.PART2_DIR / site_name
    part3_dir = C.PART3_DIR / site_name

    # 포인트 클라우드 로드 (part2)
    coords_path = part2_dir / "coords_float64.npy"
    colors_path = part2_dir / "colors_uint8.npy"

    if not coords_path.exists() or not colors_path.exists():
        print(f"[ERROR] Point cloud not found: {part2_dir}")
        return

    print(f"\n[Loading Point Cloud]")
    coords = np.load(coords_path)
    colors = np.load(colors_path)
    print(f"  Points: {len(coords):,}")

    # Votes 로드 (part3)
    votes_path = part3_dir / "forward_votes.npy"

    if not votes_path.exists():
        print(f"[ERROR] Votes file not found: {votes_path}")
        return

    print(f"\n[Loading Votes]")
    votes = np.load(votes_path)
    print(f"  Votes shape: {votes.shape}")
    print(f"  Votes min: {votes.min()}, max: {votes.max()}")

    # 통계
    unique_votes, counts = np.unique(votes, return_counts=True)
    print(f"\n[Vote Statistics]")
    for vote_val, count in zip(unique_votes[:10], counts[:10]):  # 처음 10개만
        print(f"  Vote {vote_val}: {count:,} points ({100*count/len(votes):.2f}%)")

    # Vote threshold별 LAS 생성
    print(f"\n[Creating LAS Files]")
    print(f"  Output dir: {part3_dir}")

    for threshold in thresholds:
        las_path = part3_dir / f"vote_{threshold}.las"
        create_las_from_votes(coords, colors, votes, threshold, las_path)

    print(f"\n[OK] {site_name} complete\n")


def main():
    """모든 사이트 처리"""
    print(f"\n{'='*70}")
    print(f"Forward Votes to LAS Conversion")
    print(f"{'='*70}")

    # 처리할 사이트 목록
    sites = [
        "Zenmuse_AI_Site_A",
        "Zenmuse_AI_Site_B",
        "Zenmuse_AI_Site_C",
        "P4R_Site_A_Solid",
        "P4R_Site_B_Solid_Merge_V2",
        "P4R_Site_C_Solid_Merge_V2",
        "P4R_Zenmuse_Joint_AI_Site_A",
        "P4R_Zenmuse_Joint_AI_Site_B",
        "P4R_Zenmuse_Joint_AI_Site_C",
    ]

    # Vote 임계값
    thresholds = [7, 15, 30]

    print(f"\nSites: {len(sites)}")
    print(f"Vote thresholds: {thresholds}")

    # 각 사이트 처리
    for site in sites:
        process_site(site, thresholds)

    print(f"\n{'='*70}")
    print(f"All sites processed")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
