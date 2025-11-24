@echo off
REM Footprint-based Forward Projection 실행 스크립트
REM Site B 샘플 테스트용

echo ========================================
echo Footprint-based Forward Projection
echo ========================================
echo.

REM Conda 환경 활성화
call conda activate yolov11

REM Python 스크립트 실행
python run_all_sites.py --sites Zenmuse_AI_Site_B --k-max 1

echo.
echo ========================================
echo 실행 완료!
echo ========================================
pause
