@echo off
chcp 65001 >nul
echo ============================================================
echo  MIMIC-IV BVQ11 -- Bat dau xu ly (5-6 gio)
echo ============================================================
echo.

call venv\Scripts\activate.bat
if errorlevel 1 (echo [LOI] Chua chay 1_setup.bat! & pause & exit /b 1)

for /f "tokens=2 delims==" %%A in ('findstr /i "^DATA_DIR" config.txt') do set DATA_DIR=%%A
for /f "tokens=2 delims==" %%A in ('findstr /i "^OUT_DIR"  config.txt') do set OUT_DIR=%%A
for /f "tokens=2 delims==" %%A in ('findstr /i "^HOURS"    config.txt') do set HOURS=%%A

echo Cau hinh:
echo   DATA_DIR = %DATA_DIR%
echo   OUT_DIR  = %OUT_DIR%
echo   HOURS    = %HOURS%
echo.

if not exist "%DATA_DIR%\hosp\patients.csv" (
    echo [LOI] Khong tim thay %DATA_DIR%\hosp\patients.csv
    echo Hay kiem tra lai DATA_DIR trong config.txt
    pause & exit /b 1
)
if not exist "%DATA_DIR%\icu\chartevents.csv" (
    echo [LOI] Khong tim thay %DATA_DIR%\icu\chartevents.csv
    pause & exit /b 1
)

echo [OK] Du lieu hop le. Bat dau xu ly...
echo [!] Qua trinh nay mat khoang 5-6 gio. Khong tat may!
echo.

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

python mimic4_preprocessing.py ^
    --data_dir "%DATA_DIR%" ^
    --out_dir  "%OUT_DIR%" ^
    --hours    %HOURS% ^
    --icd

if errorlevel 1 (
    echo.
    echo [LOI] Xu ly that bai. Xem thong bao loi phia tren.
    pause & exit /b 1
)

echo.
echo ============================================================
echo  [XONG] Xu ly thanh cong!
echo  Chay 4_verify.bat de kiem tra ket qua.
echo ============================================================
pause
