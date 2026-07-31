@echo off
setlocal
REM Pi-only session: pass --session N (and optional --receiver-url), SSH + Chromium + tunnel.
REM Shared logic lives in start_live_session.ps1 (do not duplicate SSH/Chromium here).
cd /d "%~dp0.."
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_live_session.ps1" -Mode pi %*
exit /b %ERRORLEVEL%
