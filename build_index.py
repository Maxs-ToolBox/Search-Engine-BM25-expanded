"""
build_index.py — Stage A (offline): build and persist the inverted index.

Run once before searching:
    python build_index.py

What this script does
---------------------
A1  Normalise every document's title and body using preprocess.py
A2  Build a positional, field-aware inverted index:
        term → df, postings [(doc_id, title_tf, body_tf, title_positions, body_positions)]
A3  Collect per-document field lengths and collection-level statistics.

On-disk layout (index_data/)
----------------------------
inverted_index.pkl   — dict[term, (df, postings_list)]
doc_map.pkl          — list[docno_string]  (index = integer doc_id)
doc_stats.pkl        — list[(title_len, body_len)]
collection_stats.pkl — dict with N, avg_title_len, avg_body_len
"""

import os
import sys
import pickle
import time
from collections import defaultdict

import config
import preprocess
import parse_docs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  saved: {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Core indexing logic
# ---------------------------------------------------------------------------

def build() -> None:
    print("=" * 60)
    print("Stage A — Building inverted index")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Data structures used during the build phase
    # ------------------------------------------------------------------
    # doc_map[doc_id] = docno string
    doc_map: list[str] = []

    # doc_stats[doc_id] = (title_len, body_len)
    doc_stats: list[tuple[int, int]] = []

    # doc_snippets[doc_id] = short preview string (raw text, not stemmed)
    doc_snippets: list[str] = []

    # build_index[term][doc_id] = {'t': [pos, ...], 'b': [pos, ...]}
    # Using defaultdict of defaultdict avoids explicit key checks.
    build_idx: dict[str, dict[int, dict]] = defaultdict(lambda: defaultdict(lambda: {"t": [], "b": []}))

    # ------------------------------------------------------------------
    # A1 + A2: parse and index every document
    # ------------------------------------------------------------------
    t0 = time.time()
    doc_count = 0

    for doc in parse_docs.iter_all_collections(config.COLLECTIONS):
        doc_id = len(doc_map)
        doc_map.append(doc["docno"])

        # Normalise title and body with positions
        title_tokens = preprocess.terms_with_positions(doc["title"])
        body_tokens  = preprocess.terms_with_positions(doc["body"])

        title_len = len(title_tokens)
        body_len  = len(body_tokens)
        doc_stats.append((title_len, body_len))

        # Store a short raw snippet for display in the GUI
        raw_body = doc["body"].strip().replace("\n", " ")
        doc_snippets.append(raw_body[: config.SNIPPET_LENGTH])

        # Update index with title positions (capped to save memory)
        for term, pos in title_tokens:
            entry = build_idx[term][doc_id]["t"]
            if len(entry) < config.MAX_POSITIONS_PER_FIELD:
                entry.append(pos)

        # Update index with body positions (capped to save memory)
        for term, pos in body_tokens:
            entry = build_idx[term][doc_id]["b"]
            if len(entry) < config.MAX_POSITIONS_PER_FIELD:
                entry.append(pos)

        doc_count += 1
        if doc_count % 10_000 == 0:
            elapsed = time.time() - t0
            print(f"  processed {doc_count:>7,} docs  ({elapsed:.0f}s)", flush=True)

    print(f"\n  Total documents indexed: {doc_count:,}")
    print(f"  Unique terms:            {len(build_idx):,}")
    print(f"  Elapsed:                 {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # A2: Convert build-phase dict to compact final format
    #
    # Final format:
    #   inverted_index[term] = (
    #       df,                           # int: document frequency
    #       [                             # postings list (sorted by doc_id)
    #           (doc_id,
    #            title_tf,               # int
    #            body_tf,                # int
    #            title_positions,        # tuple[int]  (sorted)
    #            body_positions),        # tuple[int]  (sorted)
    #           ...
    #       ]
    #   )
    # ------------------------------------------------------------------
    print("\nConverting to final index format...", flush=True)
    t1 = time.time()

    inverted_index: dict[str, tuple] = {}
    for term, doc_dict in build_idx.items():
        postings = []
        for doc_id, fields in sorted(doc_dict.items()):
            t_pos = tuple(sorted(fields["t"]))
            b_pos = tuple(sorted(fields["b"]))
            postings.append((doc_id, len(t_pos), len(b_pos), t_pos, b_pos))
        inverted_index[term] = (len(postings), postings)

    del build_idx  # free memory
    print(f"  Conversion done in {time.time() - t1:.1f}s")

    # ------------------------------------------------------------------
    # A3: Collection-level statistics
    # ------------------------------------------------------------------
    N = doc_count
    avg_title_len = sum(s[0] for s in doc_stats) / N if N else 0.0
    avg_body_len  = sum(s[1] for s in doc_stats) / N if N else 0.0

    collection_stats = {
        "N":             N,
        "avg_title_len": avg_title_len,
        "avg_body_len":  avg_body_len,
    }
    print(f"\n  avg title len = {avg_title_len:.1f}")
    print(f"  avg body  len = {avg_body_len:.1f}")

    # ------------------------------------------------------------------
    # Persist everything
    # ------------------------------------------------------------------
    print("\nSaving index files...")
    _save(inverted_index,   config.INDEX_FILE)
    _save(doc_map,          config.DOC_MAP_FILE)
    _save(doc_stats,        config.DOC_STATS_FILE)
    _save(collection_stats, config.COLL_STATS_FILE)
    _save(doc_snippets,     config.SNIPPETS_FILE)

    print(f"\nDone. Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    build()
