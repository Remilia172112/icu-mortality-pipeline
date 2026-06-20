@echo off
chcp 65001 >nul
echo ============================================================
echo  Transformer Training -- ICU Mortality Prediction
echo  (Encoder-only + Positional Encoding)
echo ============================================================
echo.

call venv\Scripts\activate.bat
for /f "tokens=2 delims==" %%A in ('findstr /i "^OUT_DIR" config.txt') do set DATA_DIR=%%A

echo Data dir: %DATA_DIR%
echo.
echo Hyperparameters:
echo   d_model: 128
echo   Heads: 8
echo   Layers: 4
echo   Feedforward dim: 256
echo   Dropout: 0.3
echo   LR: 0.0001 (Transformer nhay hon LSTM)
echo   Warmup epochs: 3
echo   Weight decay: 0.001
echo   Batch size: 128
echo   Max epochs: 50
echo   Early stopping: 15 epochs
echo   Missing mask: ON
echo   WeightedSampler: OFF
echo   pos_weight: 3.0
echo.

python train_transformer.py ^
    --data_dir  "%DATA_DIR%" ^
    --ckpt_dir  "checkpoints_transformer" ^
    --d_model   128 ^
    --n_heads   8 ^
    --n_layers  4 ^
    --dim_ff    256 ^
    --dropout   0.3 ^
    --lr        0.0001 ^
    --warmup_epochs 3 ^
    --weight_decay  0.001 ^
    --batch_size 128 ^
    --epochs    50 ^
    --patience  15 ^
    --use_mask  1 ^
    --use_sampler 0 ^
    --pos_weight 3.0

echo.
echo ============================================================
echo  Training xong! Ket qua trong thu muc checkpoints_transformer\
echo  Nhan phim bat ky de dong...
echo ============================================================
pause >nul
