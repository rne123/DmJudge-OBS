@echo off
rem Run with a console window to see error messages
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo DmJudge is not installed yet. Double-click INSTALL.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" main.py
pause
