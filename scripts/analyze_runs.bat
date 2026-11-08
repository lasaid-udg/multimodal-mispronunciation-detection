@echo off
setlocal
REM Usage: scripts\analyze_runs.bat runs_final_r10 1000 1000 8
set ROOT=%~1
set BOOT=%~2
set PERM=%~3
set JOBS=%~4
if "%ROOT%"=="" set ROOT=runs_final_r10
if "%BOOT%"=="" set BOOT=2000
if "%PERM%"=="" set PERM=2000
if "%JOBS%"=="" set JOBS=8

for /D %%D in ("%ROOT%\*") do (
  echo ============================================================
  echo ANALYZING %%D
  echo ============================================================
  md-emg analyze "%%D" --bootstrap %BOOT% --permutations %PERM% --n-jobs %JOBS%
  if errorlevel 1 goto :fail
)

echo.
echo ALL ANALYSES COMPLETED.
pause
exit /b 0

:fail
echo.
echo ANALYSIS STOPPED DUE TO ERROR.
pause
exit /b 1
