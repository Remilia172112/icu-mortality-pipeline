@echo off
chcp 65001 >nul
echo ============================================================
echo  MIMIC-IV BVQ11 -- Kiem tra ket qua xu ly
echo ============================================================
echo.

call venv\Scripts\activate.bat
for /f "tokens=2 delims==" %%A in ('findstr /i "^OUT_DIR" config.txt') do set OUT_DIR=%%A

python verify_output.py "%OUT_DIR%"

echo.
echo ============================================================
echo  Nhan phim bat ky de dong cua so nay...
echo ============================================================
pause >nul
