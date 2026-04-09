@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  run_sample.bat — launch the search engine in SAMPLE / MARKER mode
REM
REM  Requirements:
REM    1. Python + streamlit installed  (pip install -r requirements.txt)
REM    2. sample_index/ folder present in the same directory as this script
REM       (download from the project OneDrive / GitHub release and unzip here)
REM
REM  No TREC corpus or re-indexing needed.
REM ─────────────────────────────────────────────────────────────────────────────

echo.
echo  =====================================================
echo   Field-Aware BM25F Search Engine  —  SAMPLE MODE
echo  =====================================================
echo.
echo  Running on pre-built sample index (10 TREC topics).
echo  No corpus download required.
echo.

REM Activate venv if present
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

set USE_SAMPLE=1
streamlit run app.py

pause
