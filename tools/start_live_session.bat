@echo off
setlocal
REM Full PC→Pi live session: start receiver, SSH Pi with --kiosk, Chromium + optional tunnel.
REM Shared logic lives in start_live_session.ps1 (do not duplicate SSH/Chromium here).
cd /d "%~dp0.."
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_live_session.ps1" -Mode full %*
exit /b %ERRORLEVEL%
