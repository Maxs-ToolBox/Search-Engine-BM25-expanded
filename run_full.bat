@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  run_full.bat — launch the search engine in FULL / DEMO mode
REM
REM  Requirements:
REM    1. Python + streamlit installed  (pip install -r requirements.txt)
REM    2. TREC-Disk-4 and TREC-Disk-5 extracted into the project folder
REM    3. Index built:  python build_index.py
REM ─────────────────────────────────────────────────────────────────────────────

echo.
echo  =====================================================
echo   Field-Aware BM25F Search Engine  —  FULL MODE
echo  =====================================================
echo.

REM Activate venv if present
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

set USE_SAMPLE=0
streamlit run app.py

pause
