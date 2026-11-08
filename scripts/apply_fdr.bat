@echo off
setlocal
REM Usage: scripts\apply_fdr.bat runs_final_r10
set ROOT=%~1
if "%ROOT%"=="" set ROOT=runs_final_r10
md-emg fdr "%ROOT%"
if errorlevel 1 goto :fail
echo FDR correction completed.
pause
exit /b 0
:fail
echo FDR correction failed.
pause
exit /b 1
