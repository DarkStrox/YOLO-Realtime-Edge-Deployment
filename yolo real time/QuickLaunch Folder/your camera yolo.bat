@echo off
title New YOLO (OpenVINO MULTI) - Laptop Camera
cd /d "%~dp0"

echo Starting New YOLO (OpenVINO MULTI) using Laptop Camera...
"..\.venv\Scripts\python.exe" ..\detect.py --camera 0

pause
