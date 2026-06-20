@echo off
chcp 65001 >nul
echo ============================================================
echo  STEP 1/2 -- Visualize Training Curves
echo ============================================================
echo.

call venv\Scripts\activate.bat
for /f "tokens=2 delims==" %%A in ('findstr /i "^OUT_DIR" config.txt') do set DATA_DIR=%%A

python plot_training_curves.py ^
    --bilstm_dir      "checkpoints_v4" ^
    --lstm_dir        "checkpoints_lstm" ^
    --transformer_dir "checkpoints_transformer" ^
    --out_dir         "results_compare"

echo.
echo ============================================================
echo  STEP 2/2 -- Ablation Study (Bi-LSTM)
echo  Du kien: 12 configs x ~30s = ~6-8 phut
echo ============================================================
echo.

python ablation_study.py ^
    --data_dir "%DATA_DIR%" ^
    --out_dir  "ablation" ^
    --bilstm_py "train_bilstm.py" ^
    --max_epochs 25 ^
    --patience 8 ^
    --batch_size 128

echo.
echo ============================================================
echo  Hoan thanh!
echo    Curves   : results_compare\plot_curves_*.png
echo    Ablation : ablation\ablation_*.png + .csv + .md
echo  Nhan phim bat ky de dong...
echo ============================================================
pause >nul
