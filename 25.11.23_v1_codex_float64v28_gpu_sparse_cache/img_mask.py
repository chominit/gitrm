# -*- coding: utf-8 -*-
"""
img_mask.py
이미지에서 color gate RGB 조건에 맞는 픽셀 마스크 생성
color_gate.py의 RGB 범위: R(204-255), G(0-51), B(172-210)
밝은 빨강-보라색 계열 (마젠타/핑크)
"""

import numpy as np
from PIL import Image
from pathlib import Path
try:
    import numexpr as ne  # 멀티스레드 벡터 연산 가속
except ImportError:
    ne = None

def img_mask(img_path: str) -> np.ndarray:
    """
    이미지에서 color gate RGB 조건에 맞는 픽셀만 검출

    Color gate 조건 (color_gate.py에서 확인):
    - R: 204 ~ 255  (밝은 빨강)
    - G: 0 ~ 51     (어두운 초록)
    - B: 172 ~ 210  (밝은 파랑)
    → 밝은 빨강-보라색 계열 (마젠타/핑크)

    Args:
        img_path: 이미지 경로

    Returns:
        mask: (H, W) boolean array - 조건 만족 픽셀은 True
    """
    try:
        # 이미지 열기
        img = Image.open(img_path)

        # RGB로 변환
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # numpy 배열로 변환
        img_array = np.array(img, dtype=np.uint8)
        height, width = img_array.shape[:2]

        # Color gate RGB 범위 (color_gate.py와 동일)
        # 밝은 빨강-보라색 계열 (마젠타/핑크)
        R_MIN, R_MAX = 204, 255
        G_MIN, G_MAX = 0, 51
        B_MIN, B_MAX = 172, 210

        if ne:
            r = img_array[:, :, 0]
            g = img_array[:, :, 1]
            b = img_array[:, :, 2]
            final_mask = ne.evaluate(
                "(r>=R_MIN) & (r<=R_MAX) & (g>=G_MIN) & (g<=G_MAX) & (b>=B_MIN) & (b<=B_MAX)",
                local_dict={"r": r, "g": g, "b": b, "R_MIN": R_MIN, "R_MAX": R_MAX,
                            "G_MIN": G_MIN, "G_MAX": G_MAX, "B_MIN": B_MIN, "B_MAX": B_MAX},
                global_dict={}
            )
        else:
            # RGB 각 채널이 범위 내에 있는지 확인
            r_mask = (img_array[:,:,0] >= R_MIN) & (img_array[:,:,0] <= R_MAX)
            g_mask = (img_array[:,:,1] >= G_MIN) & (img_array[:,:,1] <= G_MAX)
            b_mask = (img_array[:,:,2] >= B_MIN) & (img_array[:,:,2] <= B_MAX)
            final_mask = r_mask & g_mask & b_mask

        # 검출 통계
        detected_pixels = np.sum(final_mask)
        total_pixels = height * width
        detection_ratio = detected_pixels / total_pixels if total_pixels > 0 else 0

        # 디버깅 정보 (첫 몇 이미지만)
        img_name = Path(img_path).name
        img_idx = img_name.split('_')[-1].split('.')[0]  # 예: "0001"

        try:
            idx_num = int(img_idx)
            if idx_num <= 5:  # 처음 5개 이미지만 출력
                print(f"\n[IMG Mask] {img_name}:")
                print(f"   이미지: {width}×{height} = {total_pixels:,} pixels")
                print(f"   Color gate 조건: RGB({R_MIN}-{R_MAX}, {G_MIN}-{G_MAX}, {B_MIN}-{B_MAX})")
                print(f"   검출: {detected_pixels:,} pixels ({detection_ratio*100:.2f}%)")
        except:
            pass  # 인덱스 파싱 실패 시 무시

        return final_mask

    except Exception as e:
        print(f"[ERROR] 이미지 마스크 생성 실패 ({img_path}): {e}")
        return None


def img_mask_diff(img_path: str) -> np.ndarray:
    """
    차분 이미지용 마스크 생성
    검정색(0,0,0) 배경에서 비검정 픽셀 감지

    차분 이미지 특성:
    - 배경: 검정색 (RGB: 0, 0, 0)
    - 변경된 부분: segmentation 색상 (class별 다른 색상)
    - 임계값 25로 생성되어 JPG 압축 노이즈 제거됨

    Args:
        img_path: 차분 이미지 경로

    Returns:
        mask: (H, W) boolean array - 비검정 픽셀은 True
              (R > 0 or G > 0 or B > 0)
    """
    try:
        # 이미지 열기
        img = Image.open(img_path)

        # RGB로 변환
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # numpy 배열로 변환
        img_array = np.array(img, dtype=np.uint8)
        height, width = img_array.shape[:2]

        if ne:
            r = img_array[:, :, 0]
            g = img_array[:, :, 1]
            b = img_array[:, :, 2]
            mask = ne.evaluate("(r>0) | (g>0) | (b>0)", local_dict={"r": r, "g": g, "b": b}, global_dict={})
        else:
            # 검정색이 아닌 픽셀 = R > 0 or G > 0 or B > 0
            mask = (img_array[:,:,0] > 0) | (img_array[:,:,1] > 0) | (img_array[:,:,2] > 0)

        # 검출 통계
        detected_pixels = np.sum(mask)
        total_pixels = height * width
        detection_ratio = detected_pixels / total_pixels if total_pixels > 0 else 0

        # 디버깅 정보 (첫 몇 이미지만)
        img_name = Path(img_path).name
        img_idx = img_name.split('_')[-1].split('.')[0]  # 예: "0001"

        try:
            idx_num = int(img_idx)
            if idx_num <= 5:  # 처음 5개 이미지만 출력
                print(f"\n[IMG Mask Diff] {img_name}:")
                print(f"   이미지: {width}×{height} = {total_pixels:,} pixels")
                print(f"   조건: 비검정 픽셀 (R>0 or G>0 or B>0)")
                print(f"   검출: {detected_pixels:,} pixels ({detection_ratio*100:.2f}%)")
        except:
            pass  # 인덱스 파싱 실패 시 무시

        return mask

    except Exception as e:
        print(f"[ERROR] 차분 이미지 마스크 생성 실패 ({img_path}): {e}")
        return None
