@echo off
REM ====================================================
REM Part 1: 원본 이미지와 병합 이미지 픽셀 차분 계산
REM ====================================================

echo ====================================================
echo Part 1: Pixel Difference Computation
echo ====================================================
echo.

REM Conda 환경 활성화
call C:\Users\jscool\anaconda3\Scripts\activate.bat yolov11

REM 작업 디렉토리 이동
cd /d C:\Users\jscool\uav_pipeline_codes\25.11.21\footprint_based_forward

REM 모든 사이트 처리 (threshold=10, overwrite)
echo [INFO] 모든 사이트 처리 시작...
echo.

python part1_compute_diff_pixels.py --sites all --threshold 10 --overwrite

echo.
echo ====================================================
echo 처리 완료! 결과 확인:
echo C:\Users\jscool\uav_pipeline_outputs\part1_io\
echo ====================================================
echo.

pause
