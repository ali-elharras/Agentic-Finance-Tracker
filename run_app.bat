@echo off
REM Double-click this file to launch the chatbot.
REM Closes cleanly with Ctrl+C in the window that opens.

cd /d "%~dp0"

set VENV_PY=C:\Users\Legion\venvs\finance-tracker\Scripts\python.exe

if not exist "%VENV_PY%" (
    echo Virtual environment not found at %VENV_PY%
    echo Create one with:  python -m venv .venv
    echo Then install:     .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting the finance agent...
echo Open http://localhost:8501 if your browser does not.
echo.

"%VENV_PY%" -m streamlit run app/app.py

pause
