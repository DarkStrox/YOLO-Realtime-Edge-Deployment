@echo off
REM CamScense launcher - uses the existing YOLO project's virtual environment
SETLOCAL
SET ROOT=%~dp0..
SET VENV=%ROOT%\yolo real time\.venv\Scripts\python.exe

IF NOT EXIST "%VENV%" (
    echo Virtual environment not found at: %VENV%
    echo Please create it or edit this script to point at your Python interpreter.
    pause
    exit /b 1
)

"%VENV%" "%~dp0server.py"
ENDLOCAL
