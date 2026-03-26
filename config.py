"""
config.py — Central configuration for all paths and hyperparameters.
Edit this file to tune the search engine behaviour.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(
    BASE_DIR,
    "drive-download-20260326T121407Z-1-001",
)

DISK4 = os.path.join(DATASET_DIR, "TREC-Disk-4", "TREC-Disk-4")
DISK5 = os.path.join(DATASET_DIR, "TREC-Disk-5", "TREC-Disk-5")

# Where to write / read the built index
INDEX_DIR = os.path.join(BASE_DIR, "index_data")
INDEX_FILE      = os.path.join(INDEX_DIR, "inverted_index.pkl")
DOC_MAP_FILE    = os.path.join(INDEX_DIR, "doc_map.pkl")
DOC_STATS_FILE  = os.path.join(INDEX_DIR, "doc_stats.pkl")
COLL_STATS_FILE = os.path.join(INDEX_DIR, "collection_stats.pkl")

# ---------------------------------------------------------------------------
# Collection roots  (folder path, collection-type tag)
# ---------------------------------------------------------------------------
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
