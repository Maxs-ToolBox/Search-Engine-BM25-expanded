import csv
from statistics import mean
from typing import Dict, List, Tuple

from metrics import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    r_precision,
)
from qrels_parser import parse_qrels
from topics_parser import parse_topics
from search import load_index, load_snippets, process_query
from variants import VARIANTS


TOPICS_FILE = "topics.txt"
QRELS_FILE = "qrels.txt"

SUMMARY_OUTPUT_CSV = "evaluation_results.csv"
PER_QUERY_OUTPUT_CSV = "per_query_results.csv"

BASELINE_VARIANT = "BM25_flattened"


def evaluate_single_query(
    retrieved_docnos: List[str],
    relevance_dict: Dict[str, int],
) -> Dict[str, float]:
    relevant_set = {docno for docno, rel in relevance_dict.items() if rel > 0}

    return {
        "AP": average_precision(retrieved_docnos, relevant_set),
        "P@10": precision_at_k(retrieved_docnos, relevant_set, 10),
        "nDCG@10": ndcg_at_k(retrieved_docnos, relevance_dict, 10),
        "Recall@100": recall_at_k(retrieved_docnos, relevant_set, 100),
        "R-Precision": r_precision(retrieved_docnos, relevant_set),
    }


def evaluate_variant(
    variant_config: Dict,
    topics: List[Dict[str, str]],
    qrels: Dict[str, Dict[str, int]],
    inverted_index,
    doc_map,
    doc_stats,
    collection_stats,
    top_k: int = 100,
    debug: bool = False,
    snippets=None
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    variant_name = variant_config["name"]

    per_query_rows = []

    ap_scores = []
    p10_scores = []
    ndcg10_scores = []
    recall100_scores = []
    rprec_scores = []

    for topic in topics:
        topic_id = topic["topic_id"]
        query = topic["query"]

        if topic_id not in qrels:
            continue

        relevance_dict = qrels[topic_id]
        relevant_set = {docno for docno, rel in relevance_dict.items() if rel > 0}

        if not relevant_set:
            continue

        results = process_query(
            query_text=query,
            inverted_index=inverted_index,
            doc_map=doc_map,
            doc_stats=doc_stats,
            collection_stats=collection_stats,
            top_k=top_k,
            variant_config=variant_config,
            debug=debug,
            snippets=snippets,
        )

        retrieved_docnos = [docno for _, docno in results]
        metrics_row = evaluate_single_query(retrieved_docnos, relevance_dict)

        row = {
            "variant": variant_name,
            "topic_id": topic_id,
            "query": query,
            "num_relevant": len(relevant_set),
            **metrics_row,
        }
        per_query_rows.append(row)

        ap_scores.append(metrics_row["AP"])
        p10_scores.append(metrics_row["P@10"])
        ndcg10_scores.append(metrics_row["nDCG@10"])
        recall100_scores.append(metrics_row["Recall@100"])
        rprec_scores.append(metrics_row["R-Precision"])

    if not per_query_rows:
        return {
            "variant": variant_name,
            "queries_evaluated": 0,
            "MAP": 0.0,
            "P@10": 0.0,
            "nDCG@10": 0.0,
            "Recall@100": 0.0,
            "R-Precision": 0.0,
        }, []

    summary = {
        "variant": variant_name,
        "queries_evaluated": len(per_query_rows),
        "MAP": mean(ap_scores),
        "P@10": mean(p10_scores),
        "nDCG@10": mean(ndcg10_scores),
        "Recall@100": mean(recall100_scores),
        "R-Precision": mean(rprec_scores),
    }

    return summary, per_query_rows


def build_comparison_rows(
    all_per_query_rows: List[Dict[str, float]],
    baseline_variant: str = BASELINE_VARIANT,
    comparison_metric: str = "AP",
) -> List[Dict[str, object]]:
    by_variant_then_topic = {}

    for row in all_per_query_rows:
        variant = row["variant"]
        topic_id = row["topic_id"]
        by_variant_then_topic.setdefault(variant, {})[topic_id] = row

    if baseline_variant not in by_variant_then_topic:
        return []

    baseline_rows = by_variant_then_topic[baseline_variant]
    comparison_rows = []

    for variant, variant_rows in by_variant_then_topic.items():
        if variant == baseline_variant:
            continue

        improved = 0
        degraded = 0
        unchanged = 0
        common_topics = 0
        deltas = []

        for topic_id, base_row in baseline_rows.items():
            if topic_id not in variant_rows:
                continue

            var_row = variant_rows[topic_id]
            base_score = float(base_row[comparison_metric])
            var_score = float(var_row[comparison_metric])
            delta = var_score - base_score

            deltas.append(delta)
            common_topics += 1

            if delta > 1e-12:
                improved += 1
            elif delta < -1e-12:
                degraded += 1
            else:
                unchanged += 1

        if common_topics == 0:
            comparison_rows.append({
                "baseline_variant": baseline_variant,
                "variant": variant,
                "comparison_metric": comparison_metric,
                "topics_compared": 0,
                "improved": 0,
                "degraded": 0,
                "unchanged": 0,
                "avg_delta": 0.0,
                "improved_pct": 0.0,
                "degraded_pct": 0.0,
                "unchanged_pct": 0.0,
            })
        else:
            comparison_rows.append({
                "baseline_variant": baseline_variant,
                "variant": variant,
                "comparison_metric": comparison_metric,
                "topics_compared": common_topics,
                "improved": improved,
                "degraded": degraded,
                "unchanged": unchanged,
                "avg_delta": mean(deltas),
                "improved_pct": improved / common_topics,
                "degraded_pct": degraded / common_topics,
                "unchanged_pct": unchanged / common_topics,
            })

    return comparison_rows


def print_summary_table(results: List[Dict[str, float]]) -> None:
    print("\nOverall Evaluation Results")
    print("-" * 110)
    print(
        f"{'Variant':<32}"
        f"{'Queries':>10}"
        f"{'MAP':>12}"
        f"{'P@10':>12}"
        f"{'nDCG@10':>12}"
        f"{'Recall@100':>15}"
        f"{'R-Prec':>12}"
    )
    print("-" * 110)

    for row in results:
        print(
            f"{row['variant']:<32}"
            f"{row['queries_evaluated']:>10}"
            f"{row['MAP']:>12.4f}"
            f"{row['P@10']:>12.4f}"
            f"{row['nDCG@10']:>12.4f}"
            f"{row['Recall@100']:>15.4f}"
            f"{row['R-Precision']:>12.4f}"
        )

    print("-" * 110)


def print_comparison_table(comparison_rows: List[Dict[str, object]]) -> None:
    if not comparison_rows:
        return

    print("\nBaseline Comparison (per-query AP vs BM25_flattened)")
    print("-" * 110)
    print(
        f"{'Variant':<32}"
        f"{'Topics':>10}"
        f"{'Improved':>12}"
        f"{'Degraded':>12}"
        f"{'Unchanged':>12}"
        f"{'Avg ΔAP':>12}"
        f"{'Imp %':>10}"
        f"{'Deg %':>10}"
    )
    print("-" * 110)

    for row in comparison_rows:
        print(
            f"{row['variant']:<32}"
            f"{row['topics_compared']:>10}"
            f"{row['improved']:>12}"
            f"{row['degraded']:>12}"
            f"{row['unchanged']:>12}"
            f"{row['avg_delta']:>12.4f}"
            f"{row['improved_pct']:>10.2%}"
            f"{row['degraded_pct']:>10.2%}"
        )

    print("-" * 110)


def save_summary_csv(results: List[Dict[str, float]], output_file: str = SUMMARY_OUTPUT_CSV) -> None:
    fieldnames = [
        "variant",
        "queries_evaluated",
        "MAP",
        "P@10",
        "nDCG@10",
        "Recall@100",
        "R-Precision",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved summary results to {output_file}")


def save_per_query_csv(
    per_query_rows: List[Dict[str, float]],
    comparison_rows: List[Dict[str, object]],
    output_file: str = PER_QUERY_OUTPUT_CSV,
) -> None:
    per_query_fieldnames = [
        "variant",
        "topic_id",
        "query",
        "num_relevant",
        "AP",
        "P@10",
        "nDCG@10",
        "Recall@100",
        "R-Precision",
    ]

    comparison_fieldnames = [
        "baseline_variant",
        "variant",
        "comparison_metric",
        "topics_compared",
        "improved",
        "degraded",
        "unchanged",
        "avg_delta",
        "improved_pct",
        "degraded_pct",
        "unchanged_pct",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["PER_QUERY_RESULTS"])
        writer.writerow(per_query_fieldnames)
        for row in per_query_rows:
            writer.writerow([row[field] for field in per_query_fieldnames])

        writer.writerow([])
        writer.writerow(["BASELINE_COMPARISON"])
        writer.writerow(comparison_fieldnames)
        for row in comparison_rows:
            writer.writerow([row[field] for field in comparison_fieldnames])

    print(f"Saved per-query results to {output_file}")


def main() -> None:
    print("Loading topics...")
    topics = parse_topics(TOPICS_FILE)
    print(f"Loaded {len(topics)} topics")

    print("Loading qrels...")
    qrels = parse_qrels(QRELS_FILE)
    print(f"Loaded qrels for {len(qrels)} topics")

    print("Loading search index...")
    inverted_index, doc_map, doc_stats, collection_stats = load_index()
    snippets = load_snippets()

    summary_results = []
    all_per_query_rows = []

    for variant in VARIANTS:
        print(f"\nEvaluating {variant['name']} ...")

        summary_row, per_query_rows = evaluate_variant(
            variant_config=variant,
            topics=topics,
            qrels=qrels,
            inverted_index=inverted_index,
            doc_map=doc_map,
            doc_stats=doc_stats,
            collection_stats=collection_stats,
            top_k=100,
            debug=False,
            snippets=snippets,
        )

        summary_results.append(summary_row)
        all_per_query_rows.extend(per_query_rows)

    comparison_rows = build_comparison_rows(
        all_per_query_rows=all_per_query_rows,
        baseline_variant=BASELINE_VARIANT,
        comparison_metric="AP",
    )

    print_summary_table(summary_results)
    print_comparison_table(comparison_rows)

    save_summary_csv(summary_results, SUMMARY_OUTPUT_CSV)
    save_per_query_csv(all_per_query_rows, comparison_rows, PER_QUERY_OUTPUT_CSV)


if __name__ == "__main__":
    main()