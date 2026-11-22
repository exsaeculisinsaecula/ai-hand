@echo off
chcp 65001 >nul
rem AI-HAND 실행: micro:bit(COM 포트) 자동 탐색, 대시보드 http://127.0.0.1:8090
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo 먼저 setup_windows.bat 를 실행하세요 & pause & exit /b 1)
.venv\Scripts\python.exe handover_app.py %*
pause
