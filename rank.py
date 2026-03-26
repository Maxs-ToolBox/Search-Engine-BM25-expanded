"""
rank.py — Scoring functions (BM25F + phrase bonus + proximity bonus).

Scoring formula (Section 7):
    Score(doc, query) = BM25F(doc, original_terms)
                      + γ · BM25F(doc, expanded_terms)
                      + phrase_bonus
                      + proximity_bonus

BM25F (Option 2 — Zaragoza et al. 2004):
    Combine per-field normalised TFs *before* applying saturation, so
    saturation operates on the combined evidence once (avoids double-
    application of the k1 ceiling that Option 1 suffers from).

    normalised_tf_field = tf_field / (1 - b_field + b_field * len_field / avg_len_field)
    combined_tf         = w_title * norm_title_tf + w_body * norm_body_tf
    idf                 = log((N - df + 0.5) / (df + 0.5) + 1)
    bm25f               = idf * combined_tf / (k1 + combined_tf)

Phrase bonus (Metzler & Croft 2005 — term dependence model):
    +PHRASE_BONUS  for each pair of consecutive query terms that appear
    as an exact ordered sequence in either the title or the body.

Proximity bonus:
    For each pair of query terms, +bonus if they appear within
    PROXIMITY_WINDOW tokens of each other (unordered).
    Bonus = PROXIMITY_BONUS_MAX * (1 - gap / PROXIMITY_WINDOW).
"""

import math
from itertools import combinations

import config


# ---------------------------------------------------------------------------
# IDF
# ---------------------------------------------------------------------------

def _idf(df: int, N: int) -> float:
    """Robertson-Sparck Jones IDF with +1 smoothing."""
    return math.log((N - df + 0.5) / (df + 0.5) + 1.0)


# ---------------------------------------------------------------------------
# BM25F score for a single term in a single document
# ---------------------------------------------------------------------------

def _bm25f_term(
    title_tf:    int,
    body_tf:     int,
    title_len:   int,
    body_len:    int,
    avg_title:   float,
    avg_body:    float,
    df:          int,
    N:           int,
    term_weight: float = 1.0,
) -> float:
    """
    Return the BM25F contribution for one (term, document) pair.

    *term_weight* < 1 for expanded terms (γ already applied by the caller).
    """
    if df == 0 or N == 0:
        return 0.0

    # Normalised TF per field (length normalisation)
    norm_title = (title_tf / (1 - config.B_TITLE + config.B_TITLE * title_len / avg_title)
                  if avg_title > 0 else title_tf)
    norm_body  = (body_tf  / (1 - config.B_BODY  + config.B_BODY  * body_len  / avg_body)
                  if avg_body  > 0 else body_tf)

    # Combined TF with field weights
    combined_tf = config.W_TITLE * norm_title + config.W_BODY * norm_body

    idf = _idf(df, N)

    # BM25F saturation (once, on combined TF)
    return term_weight * idf * combined_tf / (config.K1 + combined_tf)


# ---------------------------------------------------------------------------
# Phrase bonus
# ---------------------------------------------------------------------------

def _exact_phrase_positions(positions_a: tuple[int, ...],
                             positions_b: tuple[int, ...]) -> bool:
    """
    Return True if term B immediately follows term A anywhere
    in the position lists (i.e., pos_b == pos_a + 1 for some pair).
    """
    set_a = set(positions_a)
    for pb in positions_b:
        if (pb - 1) in set_a:
            return True
    return False


def _phrase_bonus(query_terms: list[str],
                  posting: tuple) -> float:
    """
    Sum of PHRASE_BONUS for each consecutive query-term pair that forms
    an exact phrase in either the title or the body positions.

    *posting* layout: (doc_id, title_tf, body_tf, title_pos, body_pos)
    """
    if len(query_terms) < 2:
        return 0.0

    _, _, _, title_pos, body_pos = posting
    bonus = 0.0

    for i in range(len(query_terms) - 1):
        # We need position data for both terms in the same doc; however
        # at the point this function is called we only have one posting.
        # The caller passes pre-fetched position tuples directly.
        pass  # handled in score_document

    return bonus


# ---------------------------------------------------------------------------
# Proximity bonus
# ---------------------------------------------------------------------------

def _min_gap(positions_a: tuple[int, ...],
             positions_b: tuple[int, ...]) -> int:
    """
    Find the minimum absolute distance between any position in A and any in B.
    Uses a two-pointer merge since both sequences are sorted.
    """
    if not positions_a or not positions_b:
        return config.PROXIMITY_WINDOW + 1  # signals "no proximity"

    i = j = 0
    min_g = abs(positions_a[0] - positions_b[0])
    while i < len(positions_a) and j < len(positions_b):
        gap = abs(positions_a[i] - positions_b[j])
        if gap < min_g:
            min_g = gap
        if min_g == 1:      # can't do better (adjacent words)
            break
        if positions_a[i] < positions_b[j]:
            i += 1
        else:
            j += 1
    return min_g


# ---------------------------------------------------------------------------
# Main scoring entry point
# ---------------------------------------------------------------------------

def score_document(
    posting_map: dict[str, tuple],   # term → (doc_id, ttf, btf, t_pos, b_pos)
    term_weights: dict[str, float],  # term → weight (1.0 for orig, γ for expanded)
    doc_stats:   tuple[int, int],    # (title_len, body_len)
    coll_stats:  dict,
    original_terms: list[str],       # used for phrase/proximity (unweighted)
) -> float:
    """
    Compute the full score for one document.

    *posting_map* contains only the terms that actually appear in this document.
    *term_weights* covers all query terms (original + expanded).
    """
    N          = coll_stats["N"]
    avg_title  = coll_stats["avg_title_len"]
    avg_body   = coll_stats["avg_body_len"]
    title_len, body_len = doc_stats

    bm25f_score = 0.0

    # -----------------------------------------------------------------------
    # BM25F component
    # -----------------------------------------------------------------------
    for term, weight in term_weights.items():
        entry = posting_map.get(term)
        if entry is None:
            continue  # term not in this doc
        _doc_id, title_tf, body_tf, _t_pos, _b_pos = entry

        # df is fetched from the global inverted index by the caller (passed via posting_map)
        df = entry[5] if len(entry) > 5 else 0
        bm25f_score += _bm25f_term(
            title_tf, body_tf,
            title_len, body_len,
            avg_title, avg_body,
            df, N,
            term_weight=weight,
        )

    # -----------------------------------------------------------------------
    # Phrase bonus (consecutive original query terms)
    # -----------------------------------------------------------------------
    phrase_score = 0.0
    orig_in_doc  = [t for t in original_terms if t in posting_map]

    for i in range(len(orig_in_doc) - 1):
        t_a = orig_in_doc[i]
        t_b = orig_in_doc[i + 1]
        _, _, _, t_pos_a, b_pos_a = posting_map[t_a][:5]
        _, _, _, t_pos_b, b_pos_b = posting_map[t_b][:5]

        if (_exact_phrase_positions(t_pos_a, t_pos_b) or
                _exact_phrase_positions(b_pos_a, b_pos_b)):
            phrase_score += config.PHRASE_BONUS

    # -----------------------------------------------------------------------
    # Proximity bonus (unordered window, all pairs of original terms)
    # -----------------------------------------------------------------------
    prox_score = 0.0
    W = config.PROXIMITY_WINDOW

    for t_a, t_b in combinations(orig_in_doc, 2):
        _, _, _, t_pos_a, b_pos_a = posting_map[t_a][:5]
        _, _, _, t_pos_b, b_pos_b = posting_map[t_b][:5]

        # Check title positions
        gap_t = _min_gap(t_pos_a, t_pos_b)
        if gap_t <= W:
            prox_score += config.PROXIMITY_BONUS_MAX * (1.0 - gap_t / W)

        # Check body positions
        gap_b = _min_gap(b_pos_a, b_pos_b)
        if gap_b <= W:
            prox_score += config.PROXIMITY_BONUS_MAX * (1.0 - gap_b / W)

    return bm25f_score + phrase_score + prox_score


# ---------------------------------------------------------------------------
# Batch ranking: score all candidate documents for a query
# ---------------------------------------------------------------------------

def rank_documents(
    term_weights: dict[str, float],
    inverted_index: dict,
    doc_stats: list[tuple[int, int]],
    collection_stats: dict,
    top_k: int = 1000,
) -> list[tuple[float, int]]:
    """
    Retrieve and score all documents that contain at least one query term.

    Returns a list of (score, doc_id) sorted descending by score, capped at top_k.

    Algorithm
    ---------
    1. For each query term, look up its posting list.
    2. For each document in those posting lists, accumulate the BM25F term
       contribution *directly* (avoiding a full posting_map per document).
    3. Add phrase and proximity bonuses using stored position data.
    """
    N         = collection_stats["N"]
    avg_title = collection_stats["avg_title_len"]
    avg_body  = collection_stats["avg_body_len"]

    # doc_id → accumulated score
    scores: dict[int, float] = {}
    # doc_id → {term: posting_tuple}  (needed for phrase/proximity)
    doc_postings: dict[int, dict[str, tuple]] = {}

    # -----------------------------------------------------------------------
    # BM25F accumulation
    # -----------------------------------------------------------------------
    for term, weight in term_weights.items():
        entry = inverted_index.get(term)
        if entry is None:
            continue
        df, postings = entry

        for posting in postings:
            doc_id, title_tf, body_tf, t_pos, b_pos = posting
            title_len, body_len = doc_stats[doc_id]

            term_score = _bm25f_term(
                title_tf, body_tf,
                title_len, body_len,
                avg_title, avg_body,
                df, N,
                term_weight=weight,
            )
            scores[doc_id] = scores.get(doc_id, 0.0) + term_score

            # Store posting for phrase/proximity (use weight=1.0 marker as 6th element)
            if doc_id not in doc_postings:
                doc_postings[doc_id] = {}
            # Store (doc_id, ttf, btf, t_pos, b_pos, df) so score_document can read df
            doc_postings[doc_id][term] = (doc_id, title_tf, body_tf, t_pos, b_pos, df)

    if not scores:
        return []

    # -----------------------------------------------------------------------
    # Phrase + proximity bonus (only for documents that matched ≥2 orig terms)
    # -----------------------------------------------------------------------
    original_terms = [t for t, w in term_weights.items() if w >= 1.0]

    W = config.PROXIMITY_WINDOW
    for doc_id, p_map in doc_postings.items():
        orig_in_doc = [t for t in original_terms if t in p_map]
        if len(orig_in_doc) < 2:
            continue

        phrase_score = 0.0
        prox_score   = 0.0

        # Phrase bonus
        for i in range(len(orig_in_doc) - 1):
            t_a, t_b = orig_in_doc[i], orig_in_doc[i + 1]
            t_pos_a, b_pos_a = p_map[t_a][3], p_map[t_a][4]
            t_pos_b, b_pos_b = p_map[t_b][3], p_map[t_b][4]
            if (_exact_phrase_positions(t_pos_a, t_pos_b) or
                    _exact_phrase_positions(b_pos_a, b_pos_b)):
                phrase_score += config.PHRASE_BONUS

        # Proximity bonus (all pairs)
        for t_a, t_b in combinations(orig_in_doc, 2):
            t_pos_a, b_pos_a = p_map[t_a][3], p_map[t_a][4]
            t_pos_b, b_pos_b = p_map[t_b][3], p_map[t_b][4]

            gap_t = _min_gap(t_pos_a, t_pos_b)
            if gap_t <= W:
                prox_score += config.PROXIMITY_BONUS_MAX * (1.0 - gap_t / W)

            gap_b = _min_gap(b_pos_a, b_pos_b)
            if gap_b <= W:
                prox_score += config.PROXIMITY_BONUS_MAX * (1.0 - gap_b / W)

        scores[doc_id] += phrase_score + prox_score

    # -----------------------------------------------------------------------
    # Sort and return top-k
    # -----------------------------------------------------------------------
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_id) for doc_id, score in ranked[:top_k]]
