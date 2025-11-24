@echo off
REM Footprint-based Forward Projection - 전체 사이트 실행
REM 9개 사이트 모두 처리

echo ========================================
echo Footprint-based Forward Projection
echo 전체 9개 사이트 처리
echo ========================================
echo.

REM Conda 환경 활성화
call conda activate yolov11

REM Python 스크립트 실행 (모든 사이트)
python run_all_sites.py --sites all --k-max 1

echo.
echo ========================================
echo 전체 실행 완료!
echo ========================================
pause
