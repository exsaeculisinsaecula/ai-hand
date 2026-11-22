@echo off
chcp 65001 >nul
rem AI-HAND Windows 설치 (처음 한 번)
cd /d "%~dp0"
where py >nul 2>nul && (py -3 -m venv .venv) || (python -m venv .venv)
if errorlevel 1 (echo [오류] Python 3.10 이상을 설치하세요: https://www.python.org/downloads/ & pause & exit /b 1)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install pyserial "numpy<2" "opencv-python-headless==4.10.0.84" ultralytics
if errorlevel 1 (echo [오류] 패키지 설치 실패 & pause & exit /b 1)
echo.
echo 설치 완료. run_windows.bat 를 실행하세요.
pause
