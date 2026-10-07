@echo off
rem ------------------------------------------------------------
rem Mobile Camera Launch Script for YOLO Real-Time
rem ------------------------------------------------------------
rem Replace <MOBILE_IP> with the IP address of your phone running an
rem IP‑camera app (e.g., IP Webcam, DroidCam, or similar). The default
rem stream URL below works for most Android IP‑camera apps.

set "MOBILE_IP=192.168.1.6"
rem Adjust the port/path if your app uses a different one.
set "CAMERA_URL=http://%MOBILE_IP%:8080/video"

rem Activate the virtual‑environment and run detect.py with the mobile
rem camera URL as source.
"..\.venv\Scripts\python.exe" "..\detect.py" --camera "%CAMERA_URL%"

pause
