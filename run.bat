@echo off
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Python virtual environment not found. Install Python and create the venv first.
    pause
    exit /b 1
)
"venv\Scripts\python.exe" -m streamlit run app.py
pause
