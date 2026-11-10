@echo off
setlocal

REM Usage:
REM   scripts\run_all_20_repeated.bat [REPEATS] [N_JOBS] [OUTPUT_ROOT]
REM Example:
REM   scripts\run_all_20_repeated.bat 10 8
REM   scripts\run_all_20_repeated.bat 10 8 runs_final_r10

set REPEATS=%~1
if "%REPEATS%"=="" set REPEATS=10
set NJOBS=%~2
if "%NJOBS%"=="" set NJOBS=8
set OUTROOT=%~3
if "%OUTROOT%"=="" set OUTROOT=runs_final_r%REPEATS%

if not exist "configs\paper.yaml" (
  echo ERROR: run this BAT from the project root. configs\paper.yaml was not found.
  pause
  exit /b 1
)

if not exist "%OUTROOT%" mkdir "%OUTROOT%"

echo ============================================================
echo Final repeated experiments: 20 scenarios
echo Repeats=%REPEATS%  n_jobs=%NJOBS%
echo Output root=%OUTROOT%
echo ============================================================

REM 4 general scenarios
for %%C in (ch1 ch2 ch3 ch4) do (
  call :run_one %%C all
  if errorlevel 1 goto :fail
)

REM 16 phoneme x channel scenarios
for %%C in (ch1 ch2 ch3 ch4) do (
  for %%P in (t d theta eth) do (
    call :run_one %%C %%P
    if errorlevel 1 goto :fail
  )
)

echo.
echo ============================================================
echo COMPLETED: all 20 repeated scenarios finished.
echo Results: %OUTROOT%
echo ============================================================
pause
exit /b 0

:run_one
echo.
echo ------------------------------------------------------------
echo channel=%~1  phoneme=%~2  repeats=%REPEATS%
echo ------------------------------------------------------------
md-emg run -c configs/paper.yaml --channel %~1 --phoneme %~2 --repeats %REPEATS% --n-jobs %NJOBS% --output-root "%OUTROOT%"
if errorlevel 1 exit /b 1
exit /b 0

:fail
echo.
echo ============================================================
echo STOPPED because one scenario failed. Check the last run above.
echo Completed runs remain saved and can be resumed individually.
echo ============================================================
pause
exit /b 1
