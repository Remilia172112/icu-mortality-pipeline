@echo off
chcp 65001 >nul
echo ============================================================
echo  COMPARE ALL MODELS -- ICU Mortality Prediction
echo  LogReg ^| XGBoost ^| LSTM ^| Bi-LSTM ^| Transformer
echo ============================================================
echo.

call venv\Scripts\activate.bat
for /f "tokens=2 delims==" %%A in ('findstr /i "^OUT_DIR" config.txt') do set DATA_DIR=%%A

echo Data dir: %DATA_DIR%
echo.

REM Kiem tra thu vien
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

echo Bat dau so sanh 5 model...
echo.

python compare_all.py ^
    --data_dir "%DATA_DIR%" ^
    --out_dir  "results_compare" ^
    --threshold 0.5 ^
    --bilstm_ckpt      "checkpoints_v4/best_model.pt" ^
    --lstm_ckpt        "checkpoints_lstm/best_model.pt" ^
    --transformer_ckpt "checkpoints_transformer/best_model.pt"

echo.
echo ============================================================
echo  Hoan thanh! Ket qua trong thu muc results_compare\
echo    - comparison_table.csv    -- bang chi tiet
echo    - comparison_table.md     -- paste vao bao cao
echo    - roc_curves.png          -- 5 duong ROC
echo    - pr_curves.png           -- 5 duong PR
echo    - compare.log             -- log day du
echo  Nhan phim bat ky de dong...
echo ============================================================
pause >nul
