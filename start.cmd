@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title lmstudio-finetune
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
    echo.
    echo [错误] 启动失败，错误码 %EC%
    pause
)
exit /b %EC%
