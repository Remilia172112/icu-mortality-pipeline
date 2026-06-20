@echo off
chcp 65001 >nul
echo ============================================================
echo  MIMIC-IV BVQ11 -- Cai dat moi truong Python
echo  (Chi can chay 1 lan)
echo ============================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [LOI] Chua cai Python!
    echo Tai tai: https://www.python.org/downloads/
    echo Khi cai: TICH chon "Add Python to PATH"
    pause
    exit /b 1
)
echo [OK] Da tim thay:
python --version
echo.

echo [1/4] Tao virtual environment...
if exist venv (echo      Da co, bo qua.) else (python -m venv venv && echo      [OK] Da tao)
echo.

echo [2/4] Kich hoat venv...
call venv\Scripts\activate.bat
echo      [OK]
echo.

echo [3/4] Nang cap pip...
python -m pip install --upgrade pip -q
echo      [OK]
echo.

echo [4/4] Cai cac thu vien can thiet...
pip install pyarrow pandas numpy scikit-learn joblib tqdm torch --index-url https://download.pytorch.org/whl/cpu -q
if errorlevel 1 (
    echo     Thu cach khac...
    pip install pyarrow pandas numpy scikit-learn joblib tqdm torch -q
)
echo      [OK]
echo.

echo ============================================================
echo  [XONG] Cai dat thanh cong!
echo  Buoc tiep theo: Mo config.txt chinh sua duong dan, roi chay 2_run.bat
echo ============================================================
pause
