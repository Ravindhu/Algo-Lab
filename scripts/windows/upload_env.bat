@echo off
REM Uploads .env to the VPS with WinSCP. Install WinSCP first:
REM   winget install --id WinSCP.WinSCP -e --silent
set WINSCP="C:\Program Files (x86)\WinSCP\WinSCP.com"
if not exist %WINSCP% set WINSCP="C:\Program Files\WinSCP\WinSCP.com"
%WINSCP% /ini=nul /script="%~dp0upload_env.txt" /log="%~dp0winscp.log"
if errorlevel 1 (
  echo Upload failed. See %~dp0winscp.log
  exit /b 1
)
echo .env uploaded to the VPS.
