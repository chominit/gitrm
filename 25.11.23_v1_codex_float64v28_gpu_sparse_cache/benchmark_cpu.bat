@echo off
setlocal
cd /d "%~dp0"
set LOG=benchmark_cpu.log
echo ==== CPU RUN START %date% %time% ==== > "%LOG%"
powershell -Command "$s=Get-Date; python run_cpu_mode.py 2>&1 | Tee-Object -FilePath '%LOG%' -Append; $e=Get-Date; 'Elapsed_sec: ' + ($e-$s).TotalSeconds | Tee-Object -FilePath '%LOG%' -Append"
echo ==== CPU RUN END %date% %time% ==== >> "%LOG%"
type "%LOG%"
