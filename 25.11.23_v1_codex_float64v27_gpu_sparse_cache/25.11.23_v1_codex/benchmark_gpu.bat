@echo off
setlocal
cd /d "%~dp0"
set LOG=benchmark_gpu.log
echo ==== GPU RUN START %date% %time% ==== > "%LOG%"
powershell -Command "$s=Get-Date; python part3_complete_pipeline.py 2>&1 | Tee-Object -FilePath '%LOG%' -Append; $e=Get-Date; 'Elapsed_sec: ' + ($e-$s).TotalSeconds | Tee-Object -FilePath '%LOG%' -Append"
echo ==== GPU RUN END %date% %time% ==== >> "%LOG%"
type "%LOG%"
