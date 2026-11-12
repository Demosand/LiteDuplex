@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

rem Find Python 3.12 / 3.11 / 3.10
for %%V in (3.12 3.11 3.10) do (
    py -%%V -c "exit()" >nul 2>&1 && (
        py -%%V app.py
        goto :end
    )
)
python app.py

:end
echo.
pause