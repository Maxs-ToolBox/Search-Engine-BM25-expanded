"""
search.py — Stage B (online): load index and answer queries.

Usage
-----
Interactive mode (REPL):
    python search.py

Single query:
    python search.py "child support enforcement"

Options
-------
--top-k     Number of results to display (default 10)
--no-expand Disable query expansion
--debug     Print expanded query and per-term weights
"""

import sys
import os
import pickle
import argparse
import time

import config
import preprocess
import query_expand
import rank as ranking


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

def load_index() -> tuple[dict, list, list, dict]:
    """
    Load and return (inverted_index, doc_map, doc_stats, collection_stats).
    Exits with a helpful message if the index has not been built yet.
    """
    required = [config.INDEX_FILE, config.DOC_MAP_FILE,
                config.DOC_STATS_FILE, config.COLL_STATS_FILE]
    missing  = [p for p in required if not os.path.exists(p)]
    if missing:
        print("Index not found. Run:  python build_index.py")
        sys.exit(1)

    def _load(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)

    print("Loading index … ", end="", flush=True)
    t0 = time.time()
    inverted_index   = _load(config.INDEX_FILE)
    doc_map          = _load(config.DOC_MAP_FILE)
    doc_stats        = _load(config.DOC_STATS_FILE)
    collection_stats = _load(config.COLL_STATS_FILE)
    print(f"done ({time.time() - t0:.1f}s)  |  "
          f"{collection_stats['N']:,} docs  |  "
          f"{len(inverted_index):,} terms")
    return inverted_index, doc_map, doc_stats, collection_stats


# ---------------------------------------------------------------------------
# Query processing pipeline (Stage B)
# ---------------------------------------------------------------------------

def process_query(
    query_text: str,
    inverted_index: dict,
    doc_map: list,
    doc_stats: list,
    collection_stats: dict,
    top_k: int = 10,
    expand: bool = True,
    debug: bool = False,
) -> list[tuple[float, str]]:
    """
    B1  Parse and normalise the query.
    B2  Expand with WordNet synonyms (optional).
    B3  Score and rank documents.

    Returns list of (score, docno) sorted descending.
    """
    # B1 — Normalise query (same pipeline as documents)
    norm = preprocess.normalise(query_text)
    if not norm:
        return []

    surface_tokens = [surface for surface, _, _ in norm]
    query_terms    = [stemmed for _, stemmed, _ in norm]

    # Remove duplicates while preserving order
    seen: set[str] = set()
    query_terms_unique: list[str] = []
    for t in query_terms:
        if t not in seen:
            seen.add(t)
            query_terms_unique.append(t)

    if debug:
        print(f"\n  [B1] Normalised terms: {query_terms_unique}")

    # B2 — Thesaurus expansion
    if expand and len(query_terms_unique) > 0:
        term_weights = query_expand.expand_query(
            query_terms_unique,
            surface_tokens,
            inverted_index,
            collection_stats,
        )
    else:
        term_weights = {t: 1.0 for t in query_terms_unique}

    if debug:
        print(f"  [B2] Term weights:")
        for t, w in sorted(term_weights.items(), key=lambda x: -x[1]):
            src = "orig" if w >= 1.0 else "exp "
            print(f"        {src}  {t:<30s}  w={w:.3f}")

    if not term_weights:
        return []

    # B3 — Rank
    t0 = time.time()
    ranked = ranking.rank_documents(
        term_weights,
        inverted_index,
        doc_stats,
        collection_stats,
        top_k=top_k,
    )
    if debug:
        print(f"  [B3] Ranking took {time.time() - t0:.3f}s  "
              f"({len(ranked)} results)")

    return [(score, doc_map[doc_id]) for score, doc_id in ranked]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_results(results: list[tuple[float, str]], top_k: int) -> None:
    if not results:
        print("  No results found.")
        return
    print(f"\n  {'Rank':<5}  {'Score':>8}  DocNo")
    print(f"  {'----':<5}  {'-----':>8}  -----")
    for rank, (score, docno) in enumerate(results[:top_k], start=1):
        print(f"  {rank:<5}  {score:>8.4f}  {docno}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TREC search engine (BM25F + phrase/proximity + WordNet expansion)"
    )
    parser.add_argument("query", nargs="?", default=None,
                        help="Query string (omit for interactive mode)")
    parser.add_argument("--top-k",    type=int,  default=10,
                        help="Number of results to show (default 10)")
    parser.add_argument("--no-expand", action="store_true",
                        help="Disable WordNet query expansion")
    parser.add_argument("--debug",    action="store_true",
                        help="Show expansion details and timing")
    args = parser.parse_args()

    inverted_index, doc_map, doc_stats, collection_stats = load_index()
    expand = not args.no_expand

    if args.query:
        # Single query mode
        results = process_query(
            args.query,
            inverted_index, doc_map, doc_stats, collection_stats,
            top_k=args.top_k,
            expand=expand,
            debug=args.debug,
        )
        print(f"\nQuery: {args.query!r}")
        _print_results(results, args.top_k)
    else:
        # Interactive REPL
        print("\nSearch engine ready. Type a query and press Enter.")
        print("Commands:  :quit  :top=N  :expand=on/off  :debug=on/off\n")
        top_k = args.top_k
        debug = args.debug

        while True:
            try:
                raw = input("Query> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not raw:
                continue
            if raw == ":quit":
                print("Bye.")
                break
            if raw.startswith(":top="):
                try:
                    top_k = int(raw.split("=", 1)[1])
                    print(f"  top-k set to {top_k}")
                except ValueError:
                    print("  Usage: :top=<int>")
                continue
            if raw == ":expand=on":
                expand = True;  print("  Expansion ON");  continue
            if raw == ":expand=off":
                expand = False; print("  Expansion OFF"); continue
            if raw == ":debug=on":
                debug = True;   print("  Debug ON");      continue
            if raw == ":debug=off":
                debug = False;  print("  Debug OFF");     continue

            t0 = time.time()
            results = process_query(
                raw,
                inverted_index, doc_map, doc_stats, collection_stats,
                top_k=top_k,
                expand=expand,
                debug=debug,
            )
            elapsed = time.time() - t0
            print(f"\nQuery: {raw!r}  ({elapsed:.3f}s)")
            _print_results(results, top_k)


if __name__ == "__main__":
    main()
