"""
search.py — load index and answer queries using configurable retrieval variants.
"""

import sys
import os
import pickle
import argparse
import time
from typing import Dict, List, Tuple

import config
import preprocess
import query_expand
import rank as ranking
from variants import DEFAULT_VARIANT, VARIANTS, get_variant_by_name


def load_index() -> tuple[dict, list, list, dict]:
    required = [
        config.INDEX_FILE,
        config.DOC_MAP_FILE,
        config.DOC_STATS_FILE,
        config.COLL_STATS_FILE,
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print("Index not found.")
        print("Run: python build_index.py")
        sys.exit(1)

    def _load(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)

    print("Loading index … ", end="", flush=True)
    t0 = time.time()
    inverted_index = _load(config.INDEX_FILE)
    doc_map = _load(config.DOC_MAP_FILE)
    doc_stats = _load(config.DOC_STATS_FILE)
    collection_stats = _load(config.COLL_STATS_FILE)
    print(
        f"done ({time.time() - t0:.1f}s) | "
        f"{collection_stats['N']:,} docs | "
        f"{len(inverted_index):,} terms"
    )

    return inverted_index, doc_map, doc_stats, collection_stats


def load_snippets():
    if hasattr(config, "SNIPPETS_FILE") and os.path.exists(config.SNIPPETS_FILE):
        with open(config.SNIPPETS_FILE, "rb") as fh:
            return pickle.load(fh)
    return None


def _prepare_query_terms(query_text: str) -> Tuple[List[str], List[str]]:
    norm = preprocess.normalise(query_text)
    if not norm:
        return [], []

    surface_tokens = [surface for surface, _, _ in norm]
    query_terms = [stemmed for _, stemmed, _ in norm]

    seen = set()
    unique_terms = []
    for term in query_terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)

    return surface_tokens, unique_terms


def process_query(
    query_text: str,
    inverted_index: dict,
    doc_map: list,
    doc_stats: list,
    collection_stats: dict,
    top_k: int = 10,
    variant_config: Dict = None,
    debug: bool = False,
    snippets=None,
) -> list[tuple[float, str]]:
    if variant_config is None:
        variant_config = DEFAULT_VARIANT

    surface_tokens, original_terms = _prepare_query_terms(query_text)
    if not original_terms:
        return []

    if debug:
        print(f"\n[B1] Normalised terms: {original_terms}")
        print(f"[B1] Variant: {variant_config['name']}")

    if variant_config["use_query_expansion"]:
        term_weights = query_expand.expand_query(
            original_terms,
            surface_tokens,
            inverted_index,
            collection_stats,
        )
    else:
        term_weights = {term: 1.0 for term in original_terms}

    if debug:
        print("[B2] Term weights:")
        for term, weight in sorted(term_weights.items(), key=lambda x: -x[1]):
            source = "orig" if weight >= 1.0 else "exp "
            print(f"  {source} {term:<30s} w={weight:.3f}")

    if not term_weights:
        return []

    t0 = time.time()
    ranked = ranking.rank_documents(
        term_weights=term_weights,
        original_terms=original_terms,
        inverted_index=inverted_index,
        doc_stats=doc_stats,
        collection_stats=collection_stats,
        variant_config=variant_config,
        top_k=max(top_k, variant_config.get("rerank_depth", 0), 100),
    )

    if variant_config.get("use_neural_rerank", False):
        from reranker import rerank_results
        
        ranked = rerank_results(
            query_text=query_text,
            ranked_results=ranked,
            doc_map=doc_map,
            rerank_depth=variant_config.get("rerank_depth", 50),
            snippets=snippets,
        )

    ranked = ranked[:top_k]

    if debug:
        print(f"[B3] Ranking took {time.time() - t0:.3f}s ({len(ranked)} results)")

    return [(score, doc_map[doc_id]) for score, doc_id in ranked]


def _print_results(results: list[tuple[float, str]], top_k: int) -> None:
    if not results:
        print("No results found.")
        return

    print(f"\n{'Rank':<5} {'Score':>10} DocNo")
    print(f"{'-'*5:<5} {'-'*10:>10} {'-'*20}")
    for rank, (score, docno) in enumerate(results[:top_k], start=1):
        print(f"{rank:<5} {score:>10.4f} {docno}")


def _list_variants() -> None:
    print("\nAvailable variants:")
    for variant in VARIANTS:
        print(f" - {variant['name']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TREC search engine with configurable ablation variants"
    )
    parser.add_argument("query", nargs="?", default=None, help="Query string")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to show")
    parser.add_argument("--variant", type=str, default=DEFAULT_VARIANT["name"], help="Variant name")
    parser.add_argument("--list-variants", action="store_true", help="List available variants and exit")
    parser.add_argument("--debug", action="store_true", help="Show query details and timing")

    args = parser.parse_args()

    if args.list_variants:
        _list_variants()
        return

    try:
        variant_config = get_variant_by_name(args.variant)
    except ValueError as e:
        print(e)
        _list_variants()
        sys.exit(1)

    inverted_index, doc_map, doc_stats, collection_stats = load_index()
    snippets = load_snippets()

    if args.query:
        results = process_query(
            query_text=args.query,
            inverted_index=inverted_index,
            doc_map=doc_map,
            doc_stats=doc_stats,
            collection_stats=collection_stats,
            top_k=args.top_k,
            variant_config=variant_config,
            debug=args.debug,
            snippets=snippets,
        )
        print(f"\nQuery: {args.query!r}")
        print(f"Variant: {variant_config['name']}")
        _print_results(results, args.top_k)
    else:
        print("\nSearch engine ready.")
        print("Type a query and press Enter.")
        print("Commands:")
        print("  :quit")
        print("  :top=N")
        print("  :variant=NAME")
        print("  :variants")
        print("  :debug=on / :debug=off\n")

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
                    print(f"top-k set to {top_k}")
                except ValueError:
                    print("Usage: :top=10")
                continue
            if raw.startswith(":variant="):
                name = raw.split("=", 1)[1].strip()
                try:
                    variant_config = get_variant_by_name(name)
                    print(f"Variant set to {variant_config['name']}")
                except ValueError as e:
                    print(e)
                    _list_variants()
                continue
            if raw == ":variants":
                _list_variants()
                continue
            if raw == ":debug=on":
                debug = True
                print("Debug ON")
                continue
            if raw == ":debug=off":
                debug = False
                print("Debug OFF")
                continue

            t0 = time.time()
            results = process_query(
                query_text=raw,
                inverted_index=inverted_index,
                doc_map=doc_map,
                doc_stats=doc_stats,
                collection_stats=collection_stats,
                top_k=top_k,
                variant_config=variant_config,
                debug=debug,
                snippets=snippets,
            )
            elapsed = time.time() - t0

            print(f"\nQuery: {raw!r} ({elapsed:.3f}s)")
            print(f"Variant: {variant_config['name']}")
            _print_results(results, top_k)
            print()


if __name__ == "__main__":
    main()