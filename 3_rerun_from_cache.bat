@echo off
chcp 65001 >nul
echo ============================================================
echo  MIMIC-IV BVQ11 -- Chay lai tu cache
echo  (Dung khi bi ngat giua chung hoac muon tao lai tensor)
echo ============================================================
echo.

call venv\Scripts\activate.bat

for /f "tokens=2 delims==" %%A in ('findstr /i "^DATA_DIR" config.txt') do set DATA_DIR=%%A
for /f "tokens=2 delims==" %%A in ('findstr /i "^OUT_DIR"  config.txt') do set OUT_DIR=%%A
for /f "tokens=2 delims==" %%A in ('findstr /i "^HOURS"    config.txt') do set HOURS=%%A

echo DATA_DIR = %DATA_DIR%
echo OUT_DIR  = %OUT_DIR%
echo.

REM Kiem tra tung cache file
set SKIP_ARGS=

if exist "%OUT_DIR%\chart_wide.parquet" (
    echo [OK] chart_wide.parquet -- dung cache
    set SKIP_ARGS=%SKIP_ARGS% --skip_chart
) else (
    echo [!!] chart_wide.parquet -- se xu ly lai ~3-4 gio
)

if exist "%OUT_DIR%\lab_wide.parquet" (
    echo [OK] lab_wide.parquet -- dung cache
    set SKIP_ARGS=%SKIP_ARGS% --skip_lab
) else (
    echo [!!] lab_wide.parquet -- se xu ly lai ~40 phut
)

if exist "%OUT_DIR%\vaso_wide.parquet" (
    echo [OK] vaso_wide.parquet -- dung cache
    set SKIP_ARGS=%SKIP_ARGS% --skip_vaso
) else (
    echo [!!] vaso_wide.parquet -- se xu ly lai ~5 phut
)

if exist "%OUT_DIR%\urine_crrt.parquet" (
    echo [OK] urine_crrt.parquet -- dung cache
    set SKIP_ARGS=%SKIP_ARGS% --skip_urine
) else (
    echo [!!] urine_crrt.parquet -- se xu ly lai ~5 phut
)

echo.
echo Chay voi cac flag: %SKIP_ARGS%
echo.

python mimic4_preprocessing.py ^
    --data_dir "%DATA_DIR%" ^
    --out_dir  "%OUT_DIR%" ^
    --hours    %HOURS% ^
    %SKIP_ARGS%

echo.
echo [XONG]
pause
