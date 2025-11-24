@echo off
REM ====================================================
REM 전체 사이트 Forward + Backward + LAS 생성
REM ====================================================

echo ====================================================
echo 전체 파이프라인 실행 (9개 사이트)
echo ====================================================
echo.
echo [처리 사이트]
echo - Zenmuse_AI_Site_A/B/C
echo - P4R_Site_A_Solid, P4R_Site_B/C_Solid_Merge_V2
echo - P4R_Zenmuse_Joint_AI_Site_A/B/C
echo.
echo [처리 단계]
echo 1. Forward (Scanline ray casting)
echo 2. Backward (역투영 검증)
echo 3. LAS 파일 생성 (vote 7/15/30)
echo.

REM Conda 환경 활성화
call C:\Users\jscool\anaconda3\Scripts\activate.bat yolov11

REM 작업 디렉토리 이동
cd /d C:\Users\jscool\uav_pipeline_codes\25.11.21\footprint_based_forward

REM 전체 처리 실행
echo [시작] %date% %time%
python part3_complete_pipeline.py

echo.
echo ====================================================
echo 완료! 결과 확인:
echo C:\Users\jscool\uav_pipeline_outputs\part3_las\
echo ====================================================
echo [종료] %date% %time%
echo.

pause