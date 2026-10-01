@echo off
REM PixelOps launcher for Windows 11 / venv at .venv
REM Usage: run.bat            (full pipeline)
REM        run.bat --skip-dense   (force sparse-only, e.g. no GPU)

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Create it first:
    echo     python -m venv .venv
    echo     .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    exit /b 1
)

.\.venv\Scripts\python.exe src\main.py %*
