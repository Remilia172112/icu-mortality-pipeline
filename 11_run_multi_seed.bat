@echo off
REM ==========================================================
REM   run_multi_seed.bat
REM   Chay thi nghiem Multi-Seed cho KLTN
REM ==========================================================

setlocal

REM ── Cau hinh ────────────────────────────────────────────
for /f "tokens=2 delims==" %%A in ('findstr /i "^OUT_DIR" config.txt') do set DATA_DIR=%%A


REM Cau hinh huan luyen
set EPOCHS=50
set PATIENCE=15
set SEEDS=42 123 456 789 2026

REM ── Banner ──────────────────────────────────────────────
echo.
echo ============================================================
echo   MULTI-SEED EXPERIMENT - ICU Mortality Prediction
echo   5 models x 5 seeds = 25 runs
echo   Uoc tinh: 30-50 phut tren RTX 5060
echo ============================================================
echo   DATA_DIR:  %DATA_DIR%
echo   EPOCHS:    %EPOCHS%
echo   PATIENCE:  %PATIENCE%
echo   SEEDS:     %SEEDS%
echo ============================================================
echo.

call venv\Scripts\activate.bat

REM ── Kiem tra Python ─────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [LOI] Khong tim thay Python. Hay cai dat Python 3.10+
    pause
    exit /b 1
)

REM ── Kiem tra folder du lieu ─────────────────────────────
if not exist "%DATA_DIR%\X_train.pt" (
    echo [LOI] Khong tim thay file du lieu trong %DATA_DIR%
    echo       Hay chay pipeline tien xu ly truoc, hoac sua DATA_DIR
    pause
    exit /b 1
)

REM ── Kiem tra GPU ────────────────────────────────────────
python -c "import torch; print(' GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only (se cham)')"
echo.

REM ── Hoi xac nhan ────────────────────────────────────────
echo Bat dau chay multi-seed experiment? ^(Y/N^)
set /p CONFIRM=
if /i not "%CONFIRM%"=="Y" (
    echo Huy bo.
    exit /b 0
)

REM ── Chay script ─────────────────────────────────────────
echo.
echo [BAT DAU] %date% %time%
echo.

python multi_seed_runs.py ^
    --data-dir "%DATA_DIR%" ^
    --epochs %EPOCHS% ^
    --patience %PATIENCE% ^
    --seeds %SEEDS%

if errorlevel 1 (
    echo.
    echo [LOI] Script gap loi! Xem log o tren.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   HOAN THANH!
echo   Ket qua trong folder: multi_seed_results\
echo     - multi_seed_raw.csv      (25 hang du lieu thô)
echo     - multi_seed_summary.csv  (mean ^& std cho moi mo hinh)
echo     - multi_seed_ci95.csv     (khoang tin cay 95%%)
echo     - boxplot_auroc.png       (box plot AUROC va F1)
echo.
echo ============================================================
echo.
pause
endlocal