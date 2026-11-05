@echo off
rem First-time setup: creates .venv and installs dependencies.
cd /d "%~dp0"
python -m venv .venv || goto :failed
.venv\Scripts\python.exe -m pip install -r requirements.txt || goto :failed
echo.
echo Done. Start the program with run.bat
pause
exit /b 0
:failed
echo.
echo Setup failed. Python 3.11 or newer must be installed and on PATH.
pause
exit /b 1
