@echo off
rem Starts the YOLO-World demo. Checks the GPU first and uses the CPU if the GPU is throttled.
cd /d "%~dp0"
set DEMO_DEVICE=
echo Checking the GPU (about 10 seconds)...
.venv\Scripts\python.exe scripts\gpu_check.py
if errorlevel 1 set DEMO_DEVICE=cpu
echo.
echo Starting the demo. Wait for "Ready in". The browser tab opens by itself at http://127.0.0.1:7860
echo Close this window to stop the demo.
echo.
.venv\Scripts\python.exe demo\app.py --sizes S
pause
