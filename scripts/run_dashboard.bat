@echo off
REM Start the dashboard using the project's virtual environment. Pass any options after the script name, e.g.
REM   scripts\run_dashboard.bat --sim --scenario gas_leak
REM   scripts\run_dashboard.bat --port COM3
REM   scripts\run_dashboard.bat --port COM3 --camera 0 --people
REM   set HAZARD_DEVICE_KEY=your-long-random-secret && scripts\run_dashboard.bat --http-ingest
cd /d "%~dp0\.."
call venv\Scripts\activate.bat
python -m src.main --dashboard %*
pause
