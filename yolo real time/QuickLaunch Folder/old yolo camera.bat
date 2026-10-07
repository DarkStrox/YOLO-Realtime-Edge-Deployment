@echo off
title Old YOLO (PyTorch CPU) - Laptop Camera
cd /d "%~dp0"

echo Starting Old YOLO (PyTorch CPU) using Laptop Camera...
"..\.venv\Scripts\python.exe" "..\..\yolov8 Aitrnix day 1 old\detect.py" --camera 0

pause
