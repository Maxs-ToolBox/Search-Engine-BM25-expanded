#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  run_sample.sh — launch the search engine in SAMPLE / MARKER mode
#
#  Requirements:
#    1. Python + streamlit installed  (pip install -r requirements.txt)
#    2. sample_index/ folder present in the same directory as this script
#       (download from the project OneDrive / GitHub release and unzip here)
#
#  No TREC corpus or re-indexing needed.
#
#  Usage:
#    bash run_sample.sh
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo " ====================================================="
echo "  Field-Aware BM25F Search Engine  —  SAMPLE MODE"
echo " ====================================================="
echo ""
echo " Running on pre-built sample index (10 TREC topics)."
echo " No corpus download required."
echo ""

# Activate venv if present
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

export USE_SAMPLE=1
streamlit run app.py
