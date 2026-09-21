@echo off
REM Run HRVerse from the project root
cd /d "%~dp0"
python -m uvicorn backend.main:app --reload
