@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
if not exist logs mkdir logs
echo. >> "%~dp0logs\whisper-writer.log"
echo ===== %DATE% %TIME% start ===== >> "%~dp0logs\whisper-writer.log"
"%~dp0venv\Scripts\pythonw.exe" -u "%~dp0run.py" 1>>"%~dp0logs\whisper-writer.log" 2>&1
