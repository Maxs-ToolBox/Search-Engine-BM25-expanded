"""
make_sample_package.py
======================
One-off script to build the sample index package for markers.

What it does
------------
1. Reads the FULL built index (index_data/).
2. Selects a small set of TREC topic IDs (configurable below).
3. Collects every document that appears in the top-K results for those
   topics OR is listed as relevant in qrels — so the sample index is
   both searchable and evaluable.
4. Rebuilds a self-contained inverted index, doc_map, doc_stats,
   collection_stats and snippets containing ONLY those documents.
5. Writes sample_topics.txt and sample_qrels.txt filtered to the same
   topic set.
6. Saves everything to  sample_index/  (ready to zip and distribute).

Usage
-----
    python make_sample_package.py

Output layout
-------------
    sample_index/
        inverted_index.pkl
        doc_map.pkl
        doc_stats.pkl
        collection_stats.pkl
        doc_snippets.pkl
        manifest.txt
    sample_topics.txt
    sample_qrels.txt
"""

import multiprocessing
import os
import sys
import pickle
import shutil
from collections import defaultdict
from typing import Dict, List, Set, Optional

# ── Configuration ─────────────────────────────────────────────────────────────

# Topics to include in the sample package.
# Choose a small, diverse set so markers can run representative queries.
SAMPLE_TOPIC_IDS: List[str] = [
    "301", "302", "303", "304", "305",
    "306", "307", "308", "310", "311",
]

# How many top-ranked docs to harvest per topic (ensures searchable results).
TOP_K_HARVEST = 50

# ── Paths (always relative to full index) ─────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR  = os.path.join(BASE_DIR, "index_data")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_index")

FULL_INDEX      = os.path.join(INDEX_DIR, "inverted_index.pkl")
FULL_DOC_MAP    = os.path.join(INDEX_DIR, "doc_map.pkl")
FULL_DOC_STATS  = os.path.join(INDEX_DIR, "doc_stats.pkl")
FULL_COLL_STATS = os.path.join(INDEX_DIR, "collection_stats.pkl")
FULL_SNIPPETS   = os.path.join(INDEX_DIR, "doc_snippets.pkl")

FULL_TOPICS = os.path.join(BASE_DIR, "topics.txt")
FULL_QRELS  = os.path.join(BASE_DIR, "qrels.txt")

OUT_TOPICS  = os.path.join(BASE_DIR, "sample_topics.txt")
OUT_QRELS   = os.path.join(BASE_DIR, "sample_qrels.txt")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pkl_load(path: str):
    print(f"  Loading {os.path.basename(path)} … ", end="", flush=True)
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    print("done")
    return obj


def _pkl_save(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    size_mb = os.path.getsize(path) / 1_048_576
    print(f"  Saved {os.path.basename(path)}  ({size_mb:.1f} MB)")


def _parse_qrels(path: str) -> Dict[str, Dict[str, int]]:
    """Returns {topic_id: {docno: relevance_int}}."""
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 4:
                topic, _, docno, rel = parts[:4]
                qrels[topic][docno] = int(rel)
    return dict(qrels)


def _parse_topics(path: str) -> List[Dict[str, str]]:
    """Returns list of {topic_id, title, desc}."""
    import re
    topics = []
    current: Dict[str, str] = {}
    section = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip()
            if "<num>" in line.lower():
                m = re.search(r"Number:\s*(\d+)", line, re.IGNORECASE)
                if m:
                    current = {"topic_id": m.group(1), "title": "", "desc": ""}
                    section = None
            elif "<title>" in line.lower():
                current["title"] = re.sub(r"<title>\s*", "", line, flags=re.IGNORECASE).strip()
                section = "title"
            elif "<desc>" in line.lower():
                section = "desc"
            elif "<narr>" in line.lower() or "</top>" in line.lower():
                section = None
                if current.get("topic_id"):
                    topics.append(current.copy())
            elif section == "title" and line.strip() and not line.startswith("<"):
                current["title"] += " " + line.strip()
            elif section == "desc" and line.strip() and not line.startswith("<"):
                current["desc"] += " " + line.strip()
    return topics


def _write_sample_topics(topics: List[Dict], path: str, selected_ids: Set[str]):
    """Write a minimal TREC-format topics file for selected IDs."""
    selected = [t for t in topics if t["topic_id"] in selected_ids]
    with open(path, "w", encoding="utf-8") as fh:
        for t in selected:
            fh.write(f"<top>\n")
            fh.write(f"<num> Number: {t['topic_id']}\n")
            fh.write(f"<title> {t['title']}\n")
            fh.write(f"<desc> Description:\n{t['desc']}\n")
            fh.write(f"</top>\n\n")
    print(f"  Wrote {len(selected)} topics → {os.path.basename(path)}")


def _write_sample_qrels(qrels: Dict, path: str, selected_ids: Set[str]):
    """Write qrels for selected topics only."""
    total_lines = 0
    with open(path, "w", encoding="utf-8") as fh:
        for topic_id in sorted(selected_ids, key=lambda x: int(x)):
            if topic_id not in qrels:
                continue
            for docno, rel in sorted(qrels[topic_id].items()):
                fh.write(f"{topic_id} 0 {docno} {rel}\n")
                total_lines += 1
    print(f"  Wrote {total_lines} qrel lines → {os.path.basename(path)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    selected_ids = set(SAMPLE_TOPIC_IDS)
    print(f"\n=== make_sample_package.py ===")
    print(f"Selected topic IDs: {sorted(selected_ids, key=int)}")

    # ── 1. Verify full index exists ───────────────────────────────────────────
    for p in [FULL_INDEX, FULL_DOC_MAP, FULL_DOC_STATS, FULL_COLL_STATS, FULL_TOPICS, FULL_QRELS]:
        if not os.path.exists(p):
            print(f"\nERROR: missing required file: {p}")
            print("Build the full index first:  python build_index.py")
            sys.exit(1)

    # ── 2. Load full index ────────────────────────────────────────────────────
    print("\n[1] Loading full index …")
    inverted_index = _pkl_load(FULL_INDEX)
    doc_map        = _pkl_load(FULL_DOC_MAP)       # list[str]: doc_id → docno
    doc_stats      = _pkl_load(FULL_DOC_STATS)     # list[dict]
    coll_stats     = _pkl_load(FULL_COLL_STATS)    # dict

    snippets: Optional[list] = None
    if os.path.exists(FULL_SNIPPETS):
        snippets = _pkl_load(FULL_SNIPPETS)

    docno_to_id: Dict[str, int] = {docno: i for i, docno in enumerate(doc_map)}
    print(f"  Full index: {coll_stats['N']:,} docs, {len(inverted_index):,} terms")

    # ── 3. Load topics & qrels ────────────────────────────────────────────────
    print("\n[2] Loading topics & qrels …")
    all_topics = _parse_topics(FULL_TOPICS)
    all_qrels  = _parse_qrels(FULL_QRELS)

    # ── 4. Harvest candidate doc IDs ─────────────────────────────────────────
    print(f"\n[3] Harvesting top-{TOP_K_HARVEST} results + relevant docs …")

    # Lazy import — only needed here, not at runtime
    sys.path.insert(0, BASE_DIR)

    # Set USE_SAMPLE=0 so config loads the full index when search imports it
    os.environ["USE_SAMPLE"] = "0"
    import importlib
    import config as cfg
    # Force reload to pick up the env override if config was already imported
    importlib.reload(cfg)

    from search import process_query
    from variants import DEFAULT_VARIANT

    keep_doc_ids: Set[int] = set()

    for topic in all_topics:
        tid = topic["topic_id"]
        if tid not in selected_ids:
            continue
        query = topic["title"].strip()

        # Top-K retrieval results
        try:
            results = process_query(
                query_text=query,
                inverted_index=inverted_index,
                doc_map=doc_map,
                doc_stats=doc_stats,
                collection_stats=coll_stats,
                top_k=TOP_K_HARVEST,
                variant_config=DEFAULT_VARIANT,
            )
            for _, docno in results:
                if docno in docno_to_id:
                    keep_doc_ids.add(docno_to_id[docno])
        except Exception as e:
            print(f"  WARNING: query failed for topic {tid}: {e}")

        # Relevant docs from qrels (so eval metrics are reproducible)
        if tid in all_qrels:
            for docno, rel in all_qrels[tid].items():
                if rel > 0 and docno in docno_to_id:
                    keep_doc_ids.add(docno_to_id[docno])

    print(f"  Collected {len(keep_doc_ids):,} unique document IDs")

    # ── 5. Build sample structures ────────────────────────────────────────────
    print("\n[4] Building sample index structures …")

    # New contiguous doc_id numbering
    old_to_new: Dict[int, int] = {old: new for new, old in enumerate(sorted(keep_doc_ids))}
    new_doc_map   = [doc_map[old]   for old in sorted(keep_doc_ids)]
    new_doc_stats = [doc_stats[old] for old in sorted(keep_doc_ids)]

    # Rebuild inverted index — only keep entries for docs in the sample.
    # Actual format from build_index.py: {term: (df, [(doc_id, t_tf, b_tf, t_pos, b_pos), ...])}
    new_inverted: Dict = {}
    for term, value in inverted_index.items():
        if isinstance(value, tuple) and len(value) == 2:
            # Standard format: (df, postings_list)
            _, postings_list = value
            filtered = [
                (old_to_new[item[0]],) + item[1:]
                for item in postings_list
                if item[0] in old_to_new
            ]
            if filtered:
                new_inverted[term] = (len(filtered), filtered)
        elif isinstance(value, dict):
            # Dict-of-dicts fallback
            filtered = {
                old_to_new[old_id]: entry
                for old_id, entry in value.items()
                if old_id in old_to_new
            }
            if filtered:
                new_inverted[term] = filtered
        elif isinstance(value, list):
            # Plain list fallback
            filtered = [
                (old_to_new[item[0]],) + item[1:]
                for item in value
                if item[0] in old_to_new
            ]
            if filtered:
                new_inverted[term] = filtered

    # Recompute collection stats for the sample.
    # doc_stats format from build_index.py: list of (title_len, body_len) tuples
    def _stat(s, idx):
        return s[idx] if isinstance(s, (tuple, list)) else s.get(("title_len", "body_len")[idx], 0)

    avg_title = (sum(_stat(s, 0) for s in new_doc_stats) / len(new_doc_stats)
                 if new_doc_stats else 0)
    avg_body  = (sum(_stat(s, 1) for s in new_doc_stats) / len(new_doc_stats)
                 if new_doc_stats else 0)

    new_coll_stats = {
        **coll_stats,          # keep any extra keys (k1, b, etc.)
        "N": len(new_doc_map),
        "avg_body_len":  avg_body,
        "avg_title_len": avg_title,
    }

    # Snippets (optional)
    new_snippets: Optional[list] = None
    if snippets is not None:
        new_snippets = [
            snippets[old] if old < len(snippets) else ""
            for old in sorted(keep_doc_ids)
        ]

    print(f"  Sample index: {new_coll_stats['N']:,} docs, {len(new_inverted):,} terms")

    # ── 6. Save sample_index/ ─────────────────────────────────────────────────
    print(f"\n[5] Writing sample_index/ …")
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    _pkl_save(new_inverted,   os.path.join(SAMPLE_DIR, "inverted_index.pkl"))
    _pkl_save(new_doc_map,    os.path.join(SAMPLE_DIR, "doc_map.pkl"))
    _pkl_save(new_doc_stats,  os.path.join(SAMPLE_DIR, "doc_stats.pkl"))
    _pkl_save(new_coll_stats, os.path.join(SAMPLE_DIR, "collection_stats.pkl"))
    if new_snippets is not None:
        _pkl_save(new_snippets, os.path.join(SAMPLE_DIR, "doc_snippets.pkl"))

    # Manifest
    manifest_path = os.path.join(SAMPLE_DIR, "manifest.txt")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        fh.write(f"Sample package — built from full TREC Robust04 index\n")
        fh.write(f"Topics included: {', '.join(sorted(selected_ids, key=int))}\n")
        fh.write(f"Documents: {new_coll_stats['N']:,}\n")
        fh.write(f"Terms: {len(new_inverted):,}\n")
    print(f"  Wrote manifest.txt")

    # ── 7. Write sample topics & qrels ───────────────────────────────────────
    print(f"\n[6] Writing sample topics & qrels …")
    _write_sample_topics(all_topics, OUT_TOPICS, selected_ids)
    _write_sample_qrels(all_qrels,   OUT_QRELS,  selected_ids)

    # ── Done ─────────────────────────────────────────────────────────────────
    print("\n✓ Sample package complete!\n")
    print("Files created:")
    print(f"  sample_index/          ← pre-built index directory")
    print(f"  sample_topics.txt      ← TREC-format topic file")
    print(f"  sample_qrels.txt       ← relevance judgements")
    print()
    print("To run in sample mode:")
    print("  Windows:    run_sample.bat")
    print("  Mac/Linux:  USE_SAMPLE=1 streamlit run app.py")
    print()
    print("To zip for distribution:")
    print("  zip -r sample_package.zip sample_index/ sample_topics.txt sample_qrels.txt")


if __name__ == "__main__":
    multiprocessing.freeze_support()   # required on Windows for spawned subprocesses
    main()
