#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
setup_files.py - 초기 파일 설정 스크립트
프로젝트 파일들을 적절한 위치로 복사
"""

import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def copy_report_files():
    """Report XML 파일들을 복사"""
    
    # 소스와 대상 매핑
    file_mappings = [
        # P4R Site Report 파일들
        (Path(r"/mnt/project/P4R_Site_A_report.xml"), 
         Path(r"C:\Users\jscool\uav_pipeline_outputs\P4R_Site_A_report.xml")),
        
        (Path(r"/mnt/project/P4R_Site_B_report.xml"),
         Path(r"C:\Users\jscool\uav_pipeline_outputs\P4R_Site_B_report.xml")),
         
        (Path(r"/mnt/project/P4R_Site_C_report.xml"),
         Path(r"C:\Users\jscool\uav_pipeline_outputs\P4R_Site_C_report.xml")),
         
        # Camera DB 파일들 (필요시)
        (Path(r"/mnt/project/P4R_Site_A_cam_db.json"),
         Path(r"C:\Users\jscool\uav_pipeline_outputs\P4R_Site_A_cam_db.json")),
         
        (Path(r"/mnt/project/P4R_Site_cam_db.json"),
         Path(r"C:\Users\jscool\uav_pipeline_outputs\P4R_Site_cam_db.json")),
         
        (Path(r"/mnt/project/Site_C_cam_db.json"),
         Path(r"C:\Users\jscool\uav_pipeline_outputs\Site_C_cam_db.json")),
         
        # Calibration 파일
        (Path(r"/mnt/project/Zenmuse_AI_Site_A_calibrated_camera_parameters.txt"),
         Path(r"C:\Users\jscool\uav_pipeline_outputs\Zenmuse_AI_Site_A_calibrated_camera_parameters.txt")),
         
        # EOP 파일들
        (Path(r"/mnt/project/Site_A_Images_EOPs.txt"),
         Path(r"C:\Users\jscool\uav_pipeline_outputs\Site_A_Images_EOPs.txt")),
         
        (Path(r"/mnt/project/Site_B_C_Images_EOP.txt"),
         Path(r"C:\Users\jscool\uav_pipeline_outputs\Site_B_C_Images_EOP.txt")),
    ]
    
    for src, dst in file_mappings:
        # Windows 경로 처리
        if str(dst).startswith("C:"):
            dst_str = str(dst).replace("\\", "/").replace("C:", "/mnt/c")
            dst = Path(dst_str)
        
        try:
            # 소스 파일이 존재하는지 확인
            if src.exists():
                # 대상 디렉터리 생성
                dst.parent.mkdir(parents=True, exist_ok=True)
                
                # 파일 복사
                shutil.copy2(src, dst)
                logger.info(f"복사 완료: {src.name} -> {dst}")
            else:
                logger.warning(f"소스 파일 없음: {src}")
                
        except Exception as e:
            logger.error(f"복사 실패: {src.name} - {e}")

def check_files():
    """필요한 파일들이 있는지 확인"""
    
    required_files = [
        Path(r"C:\Users\jscool\uav_pipeline_outputs\P4R Image.txt"),
        Path(r"C:\Users\jscool\uav_pipeline_outputs\Zenmuse P1 Image 젠뮤즈 P1 이미지 좌표.txt"),
        Path(r"C:\Users\jscool\uav_pipeline_outputs\P4R_Site_A_report.xml"),
        Path(r"C:\Users\jscool\uav_pipeline_outputs\P4R_Site_B_report.xml"),
        Path(r"C:\Users\jscool\uav_pipeline_outputs\P4R_Site_C_report.xml"),
    ]
    
    missing = []
    for file_path in required_files:
        # Windows 경로 처리
        if str(file_path).startswith("C:"):
            path_str = str(file_path).replace("\\", "/").replace("C:", "/mnt/c")
            file_path = Path(path_str)
        
        if not file_path.exists():
            missing.append(file_path.name)
    
    if missing:
        logger.warning("누락된 파일들:")
        for name in missing:
            logger.warning(f"  - {name}")
        return False
    else:
        logger.info("✅ 모든 필수 파일이 준비되었습니다.")
        return True

if __name__ == "__main__":
    print("=" * 60)
    print("UAV Pipeline 파일 설정")
    print("=" * 60)
    
    # Report 파일 복사
    copy_report_files()
    
    # 파일 확인
    print()
    print("파일 확인 중...")
    if check_files():
        print("\n✅ 설정 완료! 이제 파이프라인을 실행할 수 있습니다.")
        print("python run.py")
    else:
        print("\n⚠️  일부 파일이 누락되었습니다.")
        print("수동으로 파일을 복사하거나 경로를 확인하세요.")
