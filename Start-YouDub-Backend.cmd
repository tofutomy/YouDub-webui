@echo off
setlocal
chcp 65001 >nul
pushd "%~dp0" || exit /b 1
set "PYTHONUTF8=1"
if not exist ".venv\Scripts\python.exe" (
    echo [YouDub] Missing virtual environment: .venv\Scripts\python.exe
    echo [YouDub] Please create .venv and install dependencies first.
    pause
    popd
    exit /b 1
)
".venv\Scripts\python.exe" "scripts\webui_ctl.py" start --target backend %*
set "YOUDUB_EXIT=%ERRORLEVEL%"
if not "%YOUDUB_EXIT%"=="0" pause
popd
exit /b %YOUDUB_EXIT%
