@echo off
rem Builds dist\AIVoice\AIVoice.exe. Rebuilding replaces that folder, including settings.json inside it.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe call setup.bat
.venv\Scripts\python.exe -m pip install -r requirements-build.txt || goto :failed
.venv\Scripts\python.exe -m PyInstaller AIVoice.spec --noconfirm --clean || goto :failed
echo.
echo Done: dist\AIVoice\AIVoice.exe
exit /b 0
:failed
echo.
echo Build failed.
exit /b 1
