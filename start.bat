@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found. Install Python 3, then run this again.
  exit /b 1
)

where npm >nul 2>&1
if errorlevel 1 (
  echo npm was not found. Install Node.js, then run this again.
  exit /b 1
)

if not exist "plugin\node_modules\" (
  echo Installing plugin dependencies...
  pushd plugin
  call npm install
  if errorlevel 1 (
    popd
    exit /b 1
  )
  popd
)

echo Building the Figma plugin...
pushd plugin
call npm run build
if errorlevel 1 (
  popd
  exit /b 1
)
popd

echo.
echo Plugin is built. In the Figma desktop app, open:
echo   Plugins - Development - Lynnar
echo.
echo First time only, import this manifest:
echo   %~dp0plugin\manifest.json
echo.
echo Starting the backend at http://localhost:5000
echo Leave this window open.
echo.

pushd server
python server.py
set EXITCODE=%ERRORLEVEL%
popd
exit /b %EXITCODE%
