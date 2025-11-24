@echo off
echo ======================================================================
echo Site B Forward Processing
echo ======================================================================

cd C:\Users\jscool\uav_pipeline_codes\25.11.21\footprint_based_forward

echo.
echo Clearing cache...
rd /s /q __pycache__ 2>nul

echo.
echo Activating conda environment...
call conda activate yolov11

echo.
echo Running Site B...
python run_all_sites.py --sites Zenmuse_AI_Site_B --k-max 1

echo.
echo ======================================================================
echo Complete!
echo ======================================================================
pause