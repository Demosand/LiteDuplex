@echo off
chcp 65001 >nul
rem LiteDuplex в окне. Консольный режим без окна: start.bat
cd /d "%~dp0ui"
set "npm_config_cache=%~dp0cache\npm"
set "ELECTRON_CACHE=%~dp0cache\electron"
set "electron_config_cache=%~dp0cache\electron"
if not exist "node_modules\electron\dist\electron.exe" (
  call npm install
  node node_modules\electron\install.js
)
call npm start
