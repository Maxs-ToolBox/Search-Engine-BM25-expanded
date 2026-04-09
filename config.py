"""
config.py — Central configuration for all paths and hyperparameters.
Edit this file to tune the search engine behaviour.

Sample mode (for markers / reproduction without the full TREC corpus):
    Set the environment variable USE_SAMPLE=1 before launching, or run:
        run_sample.bat       (Windows)
        bash run_sample.sh   (Mac / Linux)

Full mode (default — requires TREC Disk 4 & 5 in the project folder):
    run_full.bat  /  streamlit run app.py
"""

import os

# ---------------------------------------------------------------------------
# Mode switch  —  USE_SAMPLE=1  activates the pre-built sample package
# ---------------------------------------------------------------------------
USE_SAMPLE: bool = os.getenv("USE_SAMPLE", "0").strip() == "1"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Sample-mode paths (pre-built, no corpus needed) -----------------------
SAMPLE_DIR        = os.path.join(BASE_DIR, "sample_index")
SAMPLE_INDEX_FILE      = os.path.join(SAMPLE_DIR, "inverted_index.pkl")
SAMPLE_DOC_MAP_FILE    = os.path.join(SAMPLE_DIR, "doc_map.pkl")
SAMPLE_DOC_STATS_FILE  = os.path.join(SAMPLE_DIR, "doc_stats.pkl")
SAMPLE_COLL_STATS_FILE = os.path.join(SAMPLE_DIR, "collection_stats.pkl")
SAMPLE_SNIPPETS_FILE   = os.path.join(SAMPLE_DIR, "doc_snippets.pkl")
SAMPLE_TOPICS_FILE     = os.path.join(BASE_DIR,   "sample_topics.txt")
SAMPLE_QRELS_FILE      = os.path.join(BASE_DIR,   "sample_qrels.txt")

# --- Full-mode helpers -----------------------------------------------------

def _find_disk(base: str, disk_name: str) -> str:
    """
    Search *base* recursively for a folder named *disk_name*
    (e.g. 'TREC-Disk-4'). Works regardless of what the parent
    download folder is called.
    Returns the path to the inner data folder (the one that contains
    FT/, FR94/, etc.) or raises FileNotFoundError with a helpful message.
    """
    for dirpath, dirnames, _ in os.walk(base):
        for d in dirnames:
            if d == disk_name:
                candidate = os.path.join(dirpath, d)
                # The TREC disks are double-nested: TREC-Disk-4/TREC-Disk-4/
                inner = os.path.join(candidate, disk_name)
                return inner if os.path.isdir(inner) else candidate
    raise FileNotFoundError(
        f"Could not find '{disk_name}' anywhere under '{base}'.\n"
        f"Make sure the dataset is extracted inside the CODE folder."
    )


if USE_SAMPLE:
    # ── Sample mode ── no corpus required ──────────────────────────────────
    print("[config] Running in SAMPLE MODE — using pre-built sample index.")
    DISK4 = None
    DISK5 = None
    INDEX_DIR       = SAMPLE_DIR
    INDEX_FILE      = SAMPLE_INDEX_FILE
    DOC_MAP_FILE    = SAMPLE_DOC_MAP_FILE
    DOC_STATS_FILE  = SAMPLE_DOC_STATS_FILE
    COLL_STATS_FILE = SAMPLE_COLL_STATS_FILE
    SNIPPETS_FILE   = SAMPLE_SNIPPETS_FILE
    TOPICS_FILE     = SAMPLE_TOPICS_FILE
    QRELS_FILE      = SAMPLE_QRELS_FILE
    # Empty collection list — _find_document() gracefully returns None in sample mode
    COLLECTIONS: list = []
else:
    # ── Full mode ── requires TREC Disk 4 & 5 ──────────────────────────────
    DISK4 = _find_disk(BASE_DIR, "TREC-Disk-4")
    DISK5 = _find_disk(BASE_DIR, "TREC-Disk-5")
    print(f"[config] TREC-Disk-4 = {DISK4}")
    print(f"[config] TREC-Disk-5 = {DISK5}")

    INDEX_DIR       = os.path.join(BASE_DIR, "index_data")
    INDEX_FILE      = os.path.join(INDEX_DIR, "inverted_index.pkl")
    DOC_MAP_FILE    = os.path.join(INDEX_DIR, "doc_map.pkl")
    DOC_STATS_FILE  = os.path.join(INDEX_DIR, "doc_stats.pkl")
    COLL_STATS_FILE = os.path.join(INDEX_DIR, "collection_stats.pkl")
    SNIPPETS_FILE   = os.path.join(INDEX_DIR, "doc_snippets.pkl")
    TOPICS_FILE     = os.path.join(BASE_DIR,  "topics.txt")
    QRELS_FILE      = os.path.join(BASE_DIR,  "qrels.txt")

    COLLECTIONS = [
        (os.path.join(DISK4, "FT"),         "FT"),
        (os.path.join(DISK4, "FR94"),       "FR94"),
        (os.path.join(DISK4, "CR_103RD"),   "CR"),
        (os.path.join(DISK5, "FBIS"),       "FBIS"),
        (os.path.join(DISK5, "LATIMES"),    "LATIMES"),
    ]

# ---------------------------------------------------------------------------
# Preprocessing flags
# ---------------------------------------------------------------------------
DO_STEM             = True    # Porter stemming
DO_REMOVE_STOPWORDS = True    # English stopword removal

# ---------------------------------------------------------------------------
# BM25F parameters
# ---------------------------------------------------------------------------
K1      = 1.2    # term-frequency saturation
B_TITLE = 0.75   # length normalisation — title field
B_BODY  = 0.75   # length normalisation — body field

# Field weights (how much more important the title is than the body)
W_TITLE = 5.0
W_BODY  = 1.0

# ---------------------------------------------------------------------------
# Phrase and proximity bonuses (added to BM25F score)
# ---------------------------------------------------------------------------
PHRASE_BONUS        = 1.5   # bonus when exact phrase appears
PROXIMITY_WINDOW    = 8     # words: terms within this window get a bonus
PROXIMITY_BONUS_MAX = 0.5   # max bonus per term-pair (scaled by closeness)

# ---------------------------------------------------------------------------
# Query expansion (WordNet / lexical thesaurus)
# ---------------------------------------------------------------------------
EXPANSION_GAMMA         = 0.3   # γ: weight of expanded terms vs. originals (0<γ<1)
MAX_EXPANSIONS_PER_TERM = 3     # cap on synonyms added per key term
MAX_DF_RATIO            = 0.10  # reject expansions present in >10% of docs (too generic)
MIN_COOCCURRENCE        = 1     # candidate must co-occur with ≥1 original term in ≥N docs

# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------
# Maximum number of positions stored per term per field per document.
# Positions beyond this cap are dropped — phrase/proximity matching is
# unaffected in practice since a single nearby occurrence is all that's needed.
# BM25 saturates around TF≈5 with k1=1.2, so 20 is plenty.
MAX_POSITIONS_PER_FIELD = 20

# ---------------------------------------------------------------------------
# SPIMI indexing (memory management during build)
# ---------------------------------------------------------------------------
# Flush a partial index to disk every this many documents, then merge at the
# end.  Lower = less peak RAM, more disk I/O.  20 000 ≈ 1–2 GB peak per chunk.
SPIMI_CHUNK_SIZE = 20_000

# Maximum characters read from a document body before preprocessing.
# Guards against pathologically large documents causing MemoryError.
MAX_BODY_CHARS = 200_000
