import math
from typing import Dict, List, Set


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Precision@k:
    proportion of the top-k retrieved documents that are relevant.
    """
    if k <= 0:
        return 0.0

    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0

    hits = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Recall@k:
    proportion of all relevant documents that appear in the top-k retrieved documents.
    """
    if k <= 0 or not relevant:
        return 0.0

    retrieved_k = retrieved[:k]
    hits = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / len(relevant)


def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Average Precision (AP):
    average of precision values at the ranks where a relevant document is retrieved.

    MAP is computed later in evaluate.py by averaging AP over all queries.
    """
    if not relevant:
        return 0.0

    score = 0.0
    hits = 0

    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            hits += 1
            score += hits / rank

    return score / len(relevant)


def r_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """
    R-Precision:
    precision at R, where R is the total number of relevant documents for the query.
    """
    if not relevant:
        return 0.0

    r = len(relevant)
    retrieved_r = retrieved[:r]
    hits = sum(1 for doc in retrieved_r if doc in relevant)
    return hits / r


def dcg_at_k(retrieved: List[str], relevance_dict: Dict[str, int], k: int) -> float:
    """
    Discounted Cumulative Gain (DCG@k):
    graded relevance metric that rewards highly relevant documents appearing near the top.
    """
    if k <= 0:
        return 0.0

    dcg = 0.0

    for rank, doc in enumerate(retrieved[:k], start=1):
        rel = relevance_dict.get(doc, 0)
        dcg += (2**rel - 1) / math.log2(rank + 1)

    return dcg


def ndcg_at_k(retrieved: List[str], relevance_dict: Dict[str, int], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain (nDCG@k):
    DCG@k normalized by the ideal DCG@k for the same query.
    """
    if k <= 0:
        return 0.0

    dcg = dcg_at_k(retrieved, relevance_dict, k)

    ideal_rels = sorted(relevance_dict.values(), reverse=True)[:k]
    ideal_dcg = 0.0

    for rank, rel in enumerate(ideal_rels, start=1):
        ideal_dcg += (2**rel - 1) / math.log2(rank + 1)

    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg