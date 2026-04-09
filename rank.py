"""
rank.py — modular scoring functions for ablation experiments.

Supported variants:
- BM25_flattened
- BM25_separate_unweighted
- BM25F
- BM25F_phrase_proximity
- BM25F_phrase_proximity_expand
"""

import math
from itertools import combinations
from typing import Dict, List, Tuple

import config


Posting = Tuple[int, int, int, tuple, tuple, int]
# (doc_id, title_tf, body_tf, title_positions, body_positions, df)


def _idf(df: int, N: int) -> float:
    """
    Robertson-Sparck Jones IDF with +1 smoothing.
    """
    if df <= 0 or N <= 0:
        return 0.0
    return math.log((N - df + 0.5) / (df + 0.5) + 1.0)


def _bm25_single_field(
    tf: int,
    doc_len: int,
    avg_len: float,
    df: int,
    N: int,
    b: float,
    term_weight: float = 1.0,
) -> float:
    """
    Standard BM25 contribution for one field.
    """
    if tf <= 0 or df <= 0 or N <= 0:
        return 0.0

    if avg_len > 0:
        norm_tf = tf / (1 - b + b * doc_len / avg_len)
    else:
        norm_tf = tf

    idf = _idf(df, N)
    return term_weight * idf * norm_tf / (config.K1 + norm_tf)


def _bm25_flattened_term(
    title_tf: int,
    body_tf: int,
    title_len: int,
    body_len: int,
    avg_title: float,
    avg_body: float,
    df: int,
    N: int,
    term_weight: float = 1.0,
) -> float:
    """
    Approximate flattened BM25 by merging title and body at scoring time.
    """
    tf = title_tf + body_tf
    doc_len = title_len + body_len
    avg_len = avg_title + avg_body

    return _bm25_single_field(
        tf=tf,
        doc_len=doc_len,
        avg_len=avg_len,
        df=df,
        N=N,
        b=config.B_BODY,
        term_weight=term_weight,
    )


def _bm25_separate_unweighted_term(
    title_tf: int,
    body_tf: int,
    title_len: int,
    body_len: int,
    avg_title: float,
    avg_body: float,
    df: int,
    N: int,
    term_weight: float = 1.0,
) -> float:
    """
    Separate-field BM25 with equal field weighting.
    """
    score_title = _bm25_single_field(
        tf=title_tf,
        doc_len=title_len,
        avg_len=avg_title,
        df=df,
        N=N,
        b=config.B_TITLE,
        term_weight=term_weight,
    )

    score_body = _bm25_single_field(
        tf=body_tf,
        doc_len=body_len,
        avg_len=avg_body,
        df=df,
        N=N,
        b=config.B_BODY,
        term_weight=term_weight,
    )

    return score_title + score_body


def _bm25f_term(
    title_tf: int,
    body_tf: int,
    title_len: int,
    body_len: int,
    avg_title: float,
    avg_body: float,
    df: int,
    N: int,
    title_weight: float,
    body_weight: float,
    term_weight: float = 1.0,
) -> float:
    """
    BM25F contribution for one term in one document.
    """
    if df <= 0 or N <= 0:
        return 0.0

    norm_title = (
        title_tf / (1 - config.B_TITLE + config.B_TITLE * title_len / avg_title)
        if avg_title > 0 else title_tf
    )
    norm_body = (
        body_tf / (1 - config.B_BODY + config.B_BODY * body_len / avg_body)
        if avg_body > 0 else body_tf
    )

    combined_tf = title_weight * norm_title + body_weight * norm_body

    if combined_tf <= 0:
        return 0.0

    idf = _idf(df, N)
    return term_weight * idf * combined_tf / (config.K1 + combined_tf)


def _exact_phrase_positions(positions_a: tuple, positions_b: tuple) -> bool:
    """
    True if term B immediately follows term A in a field.
    """
    if not positions_a or not positions_b:
        return False

    set_a = set(positions_a)
    for pb in positions_b:
        if (pb - 1) in set_a:
            return True
    return False


def _min_gap(positions_a: tuple, positions_b: tuple) -> int:
    """
    Minimum absolute distance between two sorted position lists.
    """
    if not positions_a or not positions_b:
        return config.PROXIMITY_WINDOW + 1

    i = 0
    j = 0
    min_g = abs(positions_a[0] - positions_b[0])

    while i < len(positions_a) and j < len(positions_b):
        gap = abs(positions_a[i] - positions_b[j])
        if gap < min_g:
            min_g = gap

        if min_g == 1:
            break

        if positions_a[i] < positions_b[j]:
            i += 1
        else:
            j += 1

    return min_g


def _phrase_bonus(original_terms: List[str], posting_map: Dict[str, Posting], enabled: bool) -> float:
    if not enabled or len(original_terms) < 2:
        return 0.0

    score = 0.0

    for i in range(len(original_terms) - 1):
        t_a = original_terms[i]
        t_b = original_terms[i + 1]

        if t_a not in posting_map or t_b not in posting_map:
            continue

        _, _, _, t_pos_a, b_pos_a, _ = posting_map[t_a]
        _, _, _, t_pos_b, b_pos_b, _ = posting_map[t_b]

        if _exact_phrase_positions(t_pos_a, t_pos_b) or _exact_phrase_positions(b_pos_a, b_pos_b):
            score += config.PHRASE_BONUS

    return score


def _proximity_bonus(original_terms: List[str], posting_map: Dict[str, Posting], enabled: bool) -> float:
    if not enabled or len(original_terms) < 2:
        return 0.0

    score = 0.0
    window = config.PROXIMITY_WINDOW

    terms_in_doc = [t for t in original_terms if t in posting_map]

    for t_a, t_b in combinations(terms_in_doc, 2):
        _, _, _, t_pos_a, b_pos_a, _ = posting_map[t_a]
        _, _, _, t_pos_b, b_pos_b, _ = posting_map[t_b]

        gap_t = _min_gap(t_pos_a, t_pos_b)
        if gap_t <= window:
            score += config.PROXIMITY_BONUS_MAX * (1.0 - gap_t / window)

        gap_b = _min_gap(b_pos_a, b_pos_b)
        if gap_b <= window:
            score += config.PROXIMITY_BONUS_MAX * (1.0 - gap_b / window)

    return score


def _normalise_posting(doc_id: int, payload, df: int) -> Posting:
    """
    Convert different local posting formats into a standard form:
    (doc_id, title_tf, body_tf, title_positions, body_positions, df)
    """
    # payload might be:
    # (title_tf, body_tf, title_positions, body_positions)
    # or (title_tf, body_tf)
    # or dict-like

    if isinstance(payload, dict):
        title_tf = payload.get("title_tf", 0)
        body_tf = payload.get("body_tf", 0)
        title_positions = tuple(payload.get("title_positions", ()))
        body_positions = tuple(payload.get("body_positions", ()))
        return (doc_id, title_tf, body_tf, title_positions, body_positions, df)

    if isinstance(payload, (list, tuple)):
        if len(payload) >= 4:
            title_tf = payload[0]
            body_tf = payload[1]
            title_positions = tuple(payload[2]) if payload[2] else ()
            body_positions = tuple(payload[3]) if payload[3] else ()
            return (doc_id, title_tf, body_tf, title_positions, body_positions, df)

        if len(payload) == 2:
            title_tf = payload[0]
            body_tf = payload[1]
            return (doc_id, title_tf, body_tf, (), (), df)

    raise ValueError(f"Unsupported posting payload format for doc_id={doc_id}: {payload}")


def _iter_normalised_postings(postings) -> List[Posting]:
    """
    Support these posting container formats:

    1. dict:
       {doc_id: payload}

    2. plain postings list:
       [(doc_id, title_tf, body_tf, title_pos, body_pos), ...]

    3. tuple/list with metadata + postings list:
       (meta, [(doc_id, title_tf, body_tf, title_pos, body_pos), ...])
       [meta, [(doc_id, title_tf, body_tf, title_pos, body_pos), ...]]

    Returns a list of standardised postings:
       (doc_id, title_tf, body_tf, title_positions, body_positions, df)
    """
    normalised = []

    # Case 1: dict format
    if isinstance(postings, dict):
        df = len(postings)
        for doc_id, payload in postings.items():
            normalised.append(_normalise_posting(doc_id, payload, df))
        return normalised

    # Case 2/3: list or tuple
    if isinstance(postings, (list, tuple)):
        # Detect wrapped format: (meta, postings_list)
        if (
            len(postings) == 2
            and isinstance(postings[0], int)
            and isinstance(postings[1], (list, tuple))
        ):
            postings = postings[1]

        # Detect wrapped format: [meta, postings_list]
        elif (
            len(postings) >= 2
            and isinstance(postings[0], int)
            and isinstance(postings[1], (list, tuple))
            and len(postings[1]) > 0
            and isinstance(postings[1][0], (list, tuple))
        ):
            postings = postings[1]

        df = len(postings)

        for item in postings:
            if isinstance(item, (list, tuple)) and len(item) >= 5:
                doc_id = item[0]
                title_tf = item[1]
                body_tf = item[2]
                title_positions = tuple(item[3]) if item[3] else ()
                body_positions = tuple(item[4]) if item[4] else ()
                normalised.append(
                    (doc_id, title_tf, body_tf, title_positions, body_positions, df)
                )
            else:
                raise ValueError(f"Unsupported posting item format: {item}")

        return normalised

    raise ValueError(f"Unsupported postings container type: {type(postings)}")


def score_document(
    posting_map: Dict[str, Posting],
    term_weights: Dict[str, float],
    doc_stats: Tuple[int, int],
    coll_stats: Dict,
    original_terms: List[str],
    variant_config: Dict,
) -> float:
    """
    Score one document under the selected variant.
    """
    N = coll_stats["N"]
    avg_title = coll_stats["avg_title_len"]
    avg_body = coll_stats["avg_body_len"]
    title_len, body_len = doc_stats

    retrieval_score = 0.0

    for term, weight in term_weights.items():
        entry = posting_map.get(term)
        if entry is None:
            continue

        _, title_tf, body_tf, _, _, df = entry

        if not variant_config["use_fields"]:
            retrieval_score += _bm25_flattened_term(
                title_tf=title_tf,
                body_tf=body_tf,
                title_len=title_len,
                body_len=body_len,
                avg_title=avg_title,
                avg_body=avg_body,
                df=df,
                N=N,
                term_weight=weight,
            )
        elif not variant_config["use_bm25f"]:
            retrieval_score += _bm25_separate_unweighted_term(
                title_tf=title_tf,
                body_tf=body_tf,
                title_len=title_len,
                body_len=body_len,
                avg_title=avg_title,
                avg_body=avg_body,
                df=df,
                N=N,
                term_weight=weight,
            )
        else:
            retrieval_score += _bm25f_term(
                title_tf=title_tf,
                body_tf=body_tf,
                title_len=title_len,
                body_len=body_len,
                avg_title=avg_title,
                avg_body=avg_body,
                df=df,
                N=N,
                title_weight=variant_config["title_weight"],
                body_weight=variant_config["body_weight"],
                term_weight=weight,
            )

    phrase_score = _phrase_bonus(
        original_terms=original_terms,
        posting_map=posting_map,
        enabled=variant_config["use_phrase_bonus"],
    )

    proximity_score = _proximity_bonus(
        original_terms=original_terms,
        posting_map=posting_map,
        enabled=variant_config["use_proximity_bonus"],
    )

    return retrieval_score + phrase_score + proximity_score


def rank_documents(
    term_weights: Dict[str, float],
    original_terms: List[str],
    inverted_index: Dict,
    doc_stats: List[Tuple[int, int]],
    collection_stats: Dict,
    variant_config: Dict,
    top_k: int = 1000,
) -> List[Tuple[float, int]]:
    """
    Rank all candidate documents for a query under a selected variant.
    Returns list of (score, doc_id), sorted descending.
    """
    candidate_docs: Dict[int, Dict[str, Posting]] = {}

    for term in term_weights:
        postings = inverted_index.get(term)
        if not postings:
            continue

        normalised_postings = _iter_normalised_postings(postings)

        for posting in normalised_postings:
            doc_id, title_tf, body_tf, title_pos, body_pos, df = posting

            if doc_id not in candidate_docs:
                candidate_docs[doc_id] = {}

            candidate_docs[doc_id][term] = (
                doc_id,
                title_tf,
                body_tf,
                title_pos,
                body_pos,
                df,
            )

    scored = []

    for doc_id, posting_map in candidate_docs.items():
        score = score_document(
            posting_map=posting_map,
            term_weights=term_weights,
            doc_stats=doc_stats[doc_id],
            coll_stats=collection_stats,
            original_terms=original_terms,
            variant_config=variant_config,
        )

        if score > 0:
            scored.append((score, doc_id))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]