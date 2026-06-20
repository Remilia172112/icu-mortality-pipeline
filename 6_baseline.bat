@echo off
chcp 65001 >nul
echo ============================================================
echo  Baseline Models -- ICU Mortality Prediction
echo  Logistic Regression  ^|  XGBoost
echo ============================================================
echo.

call venv\Scripts\activate.bat
for /f "tokens=2 delims==" %%A in ('findstr /i "^OUT_DIR" config.txt') do set DATA_DIR=%%A

echo Data dir: %DATA_DIR%
echo.

REM Kiem tra / cai thu vien neu thieu
echo Kiem tra thu vien...
python -c "import xgboost" 2>nul || (
    echo  [!] Chua co xgboost -- dang cai...
    pip install xgboost --quiet
)
python -c "import matplotlib" 2>nul || (
    echo  [!] Chua co matplotlib -- dang cai...
    pip install matplotlib --quiet
)
echo  [OK] Thu vien san sang.
echo.

echo Bat dau chay baseline...
echo.

python train_baseline.py ^
    --data_dir "%DATA_DIR%" ^
    --out_dir  "results" ^
    --use_mask 1 ^
    --threshold 0.5

echo.
echo ============================================================
echo  Hoan thanh! Ket qua trong thu muc results\
echo    - baseline.log              <- log chi tiet
echo    - baseline_comparison.csv   <- bang so sanh 3 model
echo    - roc_curves.png            <- bieu do ROC
echo  Nhan phim bat ky de dong...
echo ============================================================
pause >nul
