@echo off
chcp 65001 >nul
echo ============================================================
echo  LSTM Training -- ICU Mortality Prediction
echo  (Unidirectional -- so sanh voi Bi-LSTM)
echo ============================================================
echo.

call venv\Scripts\activate.bat
for /f "tokens=2 delims==" %%A in ('findstr /i "^OUT_DIR" config.txt') do set DATA_DIR=%%A

echo Data dir: %DATA_DIR%
echo.
echo Hyperparameters:
echo   Hidden size: 128
echo   Layers: 2
echo   Dropout: 0.5
echo   LR: 0.0003
echo   Weight decay: 0.001
echo   Batch size: 128
echo   Max epochs: 50
echo   Early stopping: 15 epochs
echo   Missing mask: ON
echo   WeightedSampler: OFF
echo   pos_weight: 3.0
echo.

python train_lstm.py ^
    --data_dir  "%DATA_DIR%" ^
    --ckpt_dir  "checkpoints_lstm" ^
    --hidden_size 128 ^
    --n_layers  2 ^
    --dropout   0.5 ^
    --lr        0.0003 ^
    --weight_decay 0.001 ^
    --batch_size 128 ^
    --epochs    50 ^
    --patience  15 ^
    --use_mask  1 ^
    --use_sampler 0 ^
    --pos_weight 3.0

echo.
echo ============================================================
echo  Training xong! Ket qua trong thu muc checkpoints_lstm\
echo  Nhan phim bat ky de dong...
echo ============================================================
pause >nul
