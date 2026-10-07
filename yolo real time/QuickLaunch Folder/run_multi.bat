@echo off
title "YOLOv26 Multi-Camera Streamer"
cd /d "%~dp0"

echo ===================================================
echo   YOLOv26 Multi-Camera Splitscreen Streamer
echo ===================================================
echo Available Cameras:
echo  1. 192.168.1.28
echo  2. 192.168.1.22
echo  3. 192.168.1.38
echo  4. 192.168.1.26
echo  5. 192.168.1.34
echo  6. 192.168.1.36
echo  7. 192.168.1.2
echo  8. 192.168.1.17
echo  9. 192.168.1.15
echo ===================================================
echo.

:prompt
set /p count="Enter the number of cameras to run (1-9): "

REM Simple check to ensure input is a number
echo %count%| findstr /r "^[0-9][0-9]*$" >nul
if errorlevel 1 (
    echo Invalid input. Please enter a number between 1 and 9.
    goto prompt
)

if %count% LSS 1 (
    echo Count too small. Defaulting to 1 camera.
    set count=1
)
if %count% GTR 9 (
    echo Count too large. Defaulting to 9 cameras.
    set count=9
)

echo.
echo Starting %count% cameras in a single splitscreen view.
echo Press 'q' inside the video window to quit, or 'f' to toggle Fullscreen.
echo.

"..\.venv\Scripts\python.exe" ..\multi_detect.py --count %count%

pause
