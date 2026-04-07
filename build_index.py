"""
build_index.py — Stage A (offline): build and persist the inverted index.

Run once before searching:
    python build_index.py

Algorithm: SPIMI (Single-Pass In-Memory Indexing)
--------------------------------------------------
Rather than accumulating all postings in RAM at once (which OOMs on the full
TREC Disk 4/5 corpus), we:
  1. Process SPIMI_CHUNK_SIZE documents at a time into a small in-memory dict.
  2. Flush each chunk to a compact partial-index pickle file on disk.
  3. After all documents are processed, sequentially merge the partial files
     into the final inverted index and remove the temporary run files.

On-disk layout (index_data/)
----------------------------
inverted_index.pkl   — dict[term, (df, postings_list)]
doc_map.pkl          — list[docno_string]  (index = integer doc_id)
doc_stats.pkl        — list[(title_len, body_len)]
collection_stats.pkl — dict with N, avg_title_len, avg_body_len
doc_snippets.pkl     — list[str]  short preview per document
"""

import os
import sys
import pickle
import shutil
import time
import itertools
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

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


def _save_checkpoint(runs_dir: str, doc_map: list, doc_stats: list,
                     doc_snippets: list, run_paths: list) -> None:
    path = os.path.join(runs_dir, "checkpoint.pkl")
    with open(path, "wb") as fh:
        pickle.dump({
            "doc_map":      doc_map,
            "doc_stats":    doc_stats,
            "doc_snippets": doc_snippets,
            "run_paths":    run_paths,
        }, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _load_checkpoint(runs_dir: str) -> dict | None:
    path = os.path.join(runs_dir, "checkpoint.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        ckpt = pickle.load(fh)
    # Verify all run files referenced by the checkpoint still exist
    if not all(os.path.exists(p) for p in ckpt["run_paths"]):
        print("  WARNING: checkpoint references missing run files — starting fresh.")
        return None
    return ckpt


def _flush_run(build_idx: dict, runs_dir: str, run_num: int) -> str:
    """
    Convert *build_idx* (nested dict) to compact postings tuples and write
    a partial-index pickle.  Returns the path of the written file.
    """
    partial: dict[str, tuple] = {}
    for term, doc_dict in build_idx.items():
        postings = []
        for doc_id, fields in sorted(doc_dict.items()):
            t_pos = tuple(sorted(fields["t"]))
            b_pos = tuple(sorted(fields["b"]))
            postings.append((doc_id, len(t_pos), len(b_pos), t_pos, b_pos))
        partial[term] = (len(postings), postings)

    path = os.path.join(runs_dir, f"run_{run_num:04d}.pkl")
    with open(path, "wb") as fh:
        pickle.dump(partial, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _merge_runs(run_paths: list[str]) -> dict[str, tuple]:
    """
    Merge all partial-index files into a single in-memory inverted index.

    Because each run covers a disjoint, contiguous range of doc_ids, postings
    from later runs always sort after earlier ones — simple concatenation gives
    sorted order.
    """
    inverted_index: dict[str, list] = {}

    for i, path in enumerate(run_paths):
        print(f"  merging run {i + 1}/{len(run_paths)} …", flush=True)
        with open(path, "rb") as fh:
            partial: dict[str, tuple] = pickle.load(fh)

        for term, (df, postings) in partial.items():
            if term in inverted_index:
                entry = inverted_index[term]
                entry[0] += df
                entry[1].extend(postings)
            else:
                inverted_index[term] = [df, list(postings)]

        del partial  # release memory before next load

    return {term: (entry[0], entry[1]) for term, entry in inverted_index.items()}


# ---------------------------------------------------------------------------
# Parallel helpers (must be module-level for pickling on Windows)
# ---------------------------------------------------------------------------

def _collect_files(collections: list) -> list[tuple[str, str]]:
    """Return a sorted list of (fpath, ctype) for every data file."""
    files = []
    for root_dir, ctype in collections:
        if not os.path.isdir(root_dir):
            print(f"[WARN] Collection directory not found: {root_dir}")
            continue
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = sorted(d for d in dirnames if d.upper() != "DTDS")
            for fname in sorted(filenames):
                if not parse_docs._should_skip(fname):
                    files.append((os.path.join(dirpath, fname), ctype))
    return files


def _process_file(args: tuple) -> list:
    """
    Parse and preprocess one data file.  Runs in a worker process.
    Returns a list of (docno, title_tokens, body_tokens, snippet).
    Positions are pre-capped per term to reduce IPC transfer size.
    """
    fpath, ctype = args
    parser = parse_docs._PARSERS.get(ctype)
    if parser is None:
        return []
    text = parse_docs._read_file(fpath)
    if not text:
        return []
    cap = config.MAX_POSITIONS_PER_FIELD
    out = []
    for raw in parse_docs._split_docs(text):
        doc = parser(raw)
        if not doc or not doc["docno"]:
            continue
        title_tok = preprocess.terms_with_positions(doc["title"])
        body_tok  = preprocess.terms_with_positions(doc["body"][:config.MAX_BODY_CHARS])
        # Cap per-term positions here to reduce data sent back to main process
        def _cap(tokens):
            counts: dict = {}
            result = []
            for term, pos in tokens:
                c = counts.get(term, 0)
                if c < cap:
                    result.append((term, pos))
                    counts[term] = c + 1
            return result
        snippet = doc["body"].strip().replace("\n", " ")[:config.SNIPPET_LENGTH]
        out.append((doc["docno"], _cap(title_tok), _cap(body_tok), snippet))
    return out


# ---------------------------------------------------------------------------
# Core indexing logic
# ---------------------------------------------------------------------------

def build() -> None:
    print("=" * 60)
    print("Stage A — Building inverted index  (SPIMI)")
    print(f"  chunk size : {config.SPIMI_CHUNK_SIZE:,} docs")
    print(f"  max pos/field: {config.MAX_POSITIONS_PER_FIELD}")
    print("=" * 60)

    runs_dir = os.path.join(config.INDEX_DIR, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Resume from checkpoint if one exists
    # ------------------------------------------------------------------
    ckpt = _load_checkpoint(runs_dir)
    if ckpt:
        doc_map      = ckpt["doc_map"]
        doc_stats    = ckpt["doc_stats"]
        doc_snippets = ckpt["doc_snippets"]
        run_paths    = ckpt["run_paths"]
        docs_to_skip = len(doc_map)
        print(f"  Resuming from checkpoint: {docs_to_skip:,} docs already processed, "
              f"{len(run_paths)} run(s) on disk.")
    else:
        doc_map      = []
        doc_stats    = []
        doc_snippets = []
        run_paths    = []
        docs_to_skip = 0

    # ------------------------------------------------------------------
    # A1 + A2: SPIMI build pass
    # ------------------------------------------------------------------
    # build_idx[term][doc_id] = {"t": [pos, ...], "b": [pos, ...]}
    build_idx: dict = defaultdict(lambda: defaultdict(lambda: {"t": [], "b": []}))

    t0 = time.time()
    doc_count = len(doc_map)

    def _ingest(docno: str, title_tok: list, body_tok: list, snippet: str) -> None:
        """Add one pre-processed document into the in-memory index structures."""
        nonlocal doc_count, build_idx
        doc_id = len(doc_map)
        doc_map.append(docno)
        doc_stats.append((len(title_tok), len(body_tok)))
        doc_snippets.append(snippet)
        for term, pos in title_tok:
            build_idx[term][doc_id]["t"].append(pos)
        for term, pos in body_tok:
            build_idx[term][doc_id]["b"].append(pos)
        doc_count += 1
        if doc_count % config.SPIMI_CHUNK_SIZE == 0:
            run_num = len(run_paths)
            print(
                f"  [{time.time() - t0:5.0f}s]  {doc_count:>7,} docs — "
                f"flushing run {run_num} ({len(build_idx):,} terms) …",
                flush=True,
            )
            path = _flush_run(build_idx, runs_dir, run_num)
            run_paths.append(path)
            _save_checkpoint(runs_dir, doc_map, doc_stats, doc_snippets, run_paths)
            build_idx.clear()
            build_idx = defaultdict(lambda: defaultdict(lambda: {"t": [], "b": []}))

    if docs_to_skip:
        # ----- Sequential resume path -----
        # Preprocessing is the bottleneck, but for a resume we need to skip
        # N docs from the start of the stream so we iterate sequentially.
        print("  (sequential mode for resume)", flush=True)
        all_docs = itertools.islice(
            parse_docs.iter_all_collections(config.COLLECTIONS), docs_to_skip, None
        )
        for doc in all_docs:
            title_tok = preprocess.terms_with_positions(doc["title"])
            body_tok  = preprocess.terms_with_positions(doc["body"][:config.MAX_BODY_CHARS])
            snippet   = doc["body"].strip().replace("\n", " ")[:config.SNIPPET_LENGTH]
            _ingest(doc["docno"], title_tok, body_tok, snippet)
    else:
        # ----- Parallel path for fresh builds -----
        # Each worker parses + preprocesses one file; main process builds index.
        n_workers = max(1, multiprocessing.cpu_count() - 1)
        all_files = _collect_files(config.COLLECTIONS)
        print(f"  Workers: {n_workers}  |  Files: {len(all_files):,}", flush=True)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for file_docs in executor.map(_process_file, all_files, chunksize=8):
                for docno, title_tok, body_tok, snippet in file_docs:
                    _ingest(docno, title_tok, body_tok, snippet)

    # flush any remaining documents
    if build_idx:
        run_num = len(run_paths)
        print(
            f"  [{time.time() - t0:5.0f}s]  {doc_count:>7,} docs — "
            f"flushing final run {run_num} ({len(build_idx):,} terms) …",
            flush=True,
        )
        path = _flush_run(build_idx, runs_dir, run_num)
        run_paths.append(path)
        _save_checkpoint(runs_dir, doc_map, doc_stats, doc_snippets, run_paths)
        del build_idx

    print(f"\n  Total documents : {doc_count:,}")
    print(f"  Runs written    : {len(run_paths)}")
    print(f"  Elapsed so far  : {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Merge partial runs
    # ------------------------------------------------------------------
    print("\nMerging partial runs …", flush=True)
    t1 = time.time()
    inverted_index = _merge_runs(run_paths)
    print(f"  Unique terms    : {len(inverted_index):,}")
    print(f"  Merge time      : {time.time() - t1:.1f}s")

    # Remove temporary run files
    shutil.rmtree(runs_dir, ignore_errors=True)

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
    print("\nSaving index files …")
    _save(inverted_index,   config.INDEX_FILE)
    _save(doc_map,          config.DOC_MAP_FILE)
    _save(doc_stats,        config.DOC_STATS_FILE)
    _save(collection_stats, config.COLL_STATS_FILE)
    _save(doc_snippets,     config.SNIPPETS_FILE)

    print(f"\nDone. Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    build()
