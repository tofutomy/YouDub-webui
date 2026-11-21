@echo off
setlocal
chcp 65001 >nul
pushd "%~dp0" || exit /b 1
set "PYTHONUTF8=1"
if not exist ".venv\Scripts\python.exe" (
    echo [YouDub] Missing virtual environment: .venv\Scripts\python.exe
    pause
    popd
    exit /b 1
)
".venv\Scripts\python.exe" "scripts\webui_ctl.py" stop --target frontend %*
set "YOUDUB_EXIT=%ERRORLEVEL%"
if not "%YOUDUB_EXIT%"=="0" pause
popd
exit /b %YOUDUB_EXIT%
