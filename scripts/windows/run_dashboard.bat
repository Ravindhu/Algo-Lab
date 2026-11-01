@echo off
REM Starts the AlgoLab FastAPI dashboard on this PC.
REM Settings (DASH_PASSWORD, DASH_ALLOW_WRITES, ...) are read from .env in the repo folder.
REM
REM Default: this PC only (http://127.0.0.1:8050).
REM Phone on the same Wi-Fi: set DASH_HOST=0.0.0.0 in .env. A password is required (DASH_PASSWORD).
REM Plain HTTP sends that password unencrypted on the network, so prefer a private VPN such as Tailscale
REM over opening the port on your Wi-Fi.

cd /d "%~dp0\..\.."
if "%DASH_HOST%"=="" set DASH_HOST=127.0.0.1
if "%DASH_PORT%"=="" set DASH_PORT=8050

where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% -m uvicorn algolab.dashboard.api:app --host %DASH_HOST% --port %DASH_PORT%
pause
