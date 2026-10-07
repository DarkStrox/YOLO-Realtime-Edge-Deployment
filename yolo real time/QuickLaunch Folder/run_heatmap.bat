@echo off
title "YOLOv26 Person Heatmap Generator"
cd /d "%~dp0"

echo ===================================================
echo   YOLOv26 Person Heatmap Generator
echo ===================================================
echo Select or Enter a Camera:
echo  1. Camera 1 (192.168.1.28)
echo  2. Camera 2 (192.168.1.22)
echo  3. Camera 3 (192.168.1.38)
echo  4. Camera 4 (192.168.1.26)
echo  5. Camera 5 (192.168.1.34)
echo  6. Camera 6 (192.168.1.36)
echo  7. Camera 7 (192.168.1.2)
echo  8. Camera 8 (192.168.1.17)
echo  9. Camera 9 (192.168.1.15)
echo 10. Camera 10 (192.168.1.37)
echo 11. Camera 11 (192.168.1.50)
echo  0. Built-in Webcam (Index 0)
echo ===================================================
echo.

set /p choice="Select camera option (0-11) or paste your own RTSP URL: "

set camera_url=%choice%

if "%choice%"=="1" set camera_url=rtsp://admin:sedra12301@192.168.1.28:554/Streaming/Channels/102
if "%choice%"=="2" set camera_url=rtsp://admin:sedra12301@192.168.1.22:554/Streaming/Channels/102
if "%choice%"=="3" set camera_url=rtsp://admin:sedra12301@192.168.1.38:554/Streaming/Channels/102
if "%choice%"=="4" set camera_url=rtsp://admin:sedra12301@192.168.1.26:554/Streaming/Channels/102
if "%choice%"=="5" set camera_url=rtsp://admin:sedra12301@192.168.1.34:554/Streaming/Channels/102
if "%choice%"=="6" set camera_url=rtsp://admin:sedra12301@192.168.1.36:554/Streaming/Channels/102
if "%choice%"=="7" set camera_url=rtsp://admin:sedra12301@192.168.1.2:554/Streaming/Channels/102
if "%choice%"=="8" set camera_url=rtsp://admin:sedra12301@192.168.1.17:554/Streaming/Channels/102
if "%choice%"=="9" set camera_url=rtsp://admin:sedra12301@192.168.1.15:554/Streaming/Channels/102
if "%choice%"=="10" set camera_url=rtsp://admin:sedra12301@192.168.1.37:554/Streaming/Channels/102
if "%choice%"=="11" set camera_url=rtsp://admin:sedra12301@192.168.1.50:554/Streaming/Channels/102
if "%choice%"=="0" set camera_url=0

echo.
echo Starting Heatmap for: %camera_url%
echo.
echo Controls inside window:
echo  - Press 'q' to quit.
echo  - Press 'r' to reset heatmap.
echo  - Press 'f' to toggle Fullscreen.
echo.

"..\.venv\Scripts\python.exe" ..\heatmap.py --camera "%camera_url%"

pause
