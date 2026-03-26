"""
query_expand.py — Stage B2: Thesaurus-based query expansion via WordNet.

Design (drift-controlled expansion)
------------------------------------
B2.1  Select expandable terms     → nouns only, high-IDF only
B2.2  Sense handling              → WSD-lite via gloss/example overlap
B2.3  Generate candidates         → synonyms (same synset) only
B2.4  Filter candidates           → stopword, DF, co-occurrence, cap M
B2.5  Weight expansions           → γ < 1 relative to original terms

Public API
----------
expand_query(query_terms, inverted_index, collection_stats, pos_tags=None)
    -> dict[term: str, weight: float]

The returned dict includes original terms (weight=1.0) and expansions
(weight=EXPANSION_GAMMA), ready to be used by the ranker.
"""

import math
import re
from collections import defaultdict

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords as nltk_sw

import config
import preprocess

# Ensure required NLTK data is available
for _r in ("wordnet", "stopwords", "averaged_perceptron_tagger",
           "averaged_perceptron_tagger_eng"):
    try:
        nltk.data.find(f"corpora/{_r}")
    except LookupError:
        try:
            nltk.download(_r, quiet=True)
        except Exception:
            pass

_STOPWORDS = set(nltk_sw.words("english"))
_RE_ALPHA  = re.compile(r"^[a-z]+$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _idf(term: str, inverted_index: dict, N: int) -> float:
    """Robertson-Sparck Jones IDF (with smoothing)."""
    entry = inverted_index.get(term)
    df = entry[0] if entry else 0
    return math.log((N - df + 0.5) / (df + 0.5) + 1)


def _pos_tag_query(surface_tokens: list[str]) -> list[tuple[str, str]]:
    """POS-tag the original surface tokens and return (token, coarse_pos) pairs."""
    try:
        tagged = nltk.pos_tag(surface_tokens)
        # Coarsen Penn Treebank tags: NN* → 'n', VB* → 'v', JJ* → 'j', RB* → 'r'
        coarse = []
        for tok, tag in tagged:
            if tag.startswith("NN"):
                coarse.append((tok, "n"))
            elif tag.startswith("VB"):
                coarse.append((tok, "v"))
            elif tag.startswith("JJ"):
                coarse.append((tok, "j"))
            elif tag.startswith("RB"):
                coarse.append((tok, "r"))
            else:
                coarse.append((tok, "other"))
        return coarse
    except Exception:
        return [(t, "other") for t in surface_tokens]


def _wsd_lite(surface_token: str, wn_pos: str, query_surface_tokens: list[str]
              ) -> wn.Synset | None:
    """
    Lightweight Word Sense Disambiguation (B2.2).

    Choose the synset whose definition + examples share the most words
    with the rest of the query (context bag-of-words).  Falls back to
    the most-frequent sense (synsets()[0]) if no overlap is found.
    """
    synsets = wn.synsets(surface_token, pos=wn_pos)
    if not synsets:
        return None

    # Build a context set from the other query tokens
    context = set(query_surface_tokens) - {surface_token} - _STOPWORDS

    if not context:
        return synsets[0]   # MFS fallback

    best_synset = synsets[0]
    best_overlap = -1

    for ss in synsets:
        gloss_words = set(re.findall(r"[a-z]+", ss.definition().lower()))
        for ex in ss.examples():
            gloss_words |= set(re.findall(r"[a-z]+", ex.lower()))
        overlap = len(gloss_words & context)
        if overlap > best_overlap:
            best_overlap = overlap
            best_synset  = ss

    return best_synset


def _cooccurrence_ok(candidate: str, original_terms: list[str],
                     inverted_index: dict) -> bool:
    """
    B2.4 co-occurrence filter.

    Return True if *candidate* co-occurs (shares documents) with at least
    one original query term in at least MIN_COOCCURRENCE documents.
    We approximate this by checking posting-list intersection size ≥ 1.
    """
    cand_entry = inverted_index.get(candidate)
    if cand_entry is None:
        return False
    cand_doc_ids = {p[0] for p in cand_entry[1]}

    for orig in original_terms:
        orig_entry = inverted_index.get(orig)
        if orig_entry is None:
            continue
        orig_doc_ids = {p[0] for p in orig_entry[1]}
        if len(cand_doc_ids & orig_doc_ids) >= config.MIN_COOCCURRENCE:
            return True
    return False


# ---------------------------------------------------------------------------
# Map NLTK coarse POS to WordNet POS constant
# ---------------------------------------------------------------------------
_POS_MAP = {"n": wn.NOUN, "v": wn.VERB, "j": wn.ADJ, "r": wn.ADV}


# ---------------------------------------------------------------------------
# Main expansion function
# ---------------------------------------------------------------------------

def expand_query(
    query_terms: list[str],           # stemmed terms (from preprocess)
    surface_tokens: list[str],        # original surface tokens (pre-stem)
    inverted_index: dict,
    collection_stats: dict,
) -> dict[str, float]:
    """
    Expand *query_terms* using WordNet synonyms with seven drift controls.

    Returns a weight dict:
        {term: weight, ...}
    Original terms have weight 1.0; expansions have weight EXPANSION_GAMMA.
    """
    N = collection_stats["N"]

    # Start with original terms (weight 1.0)
    weighted: dict[str, float] = {t: 1.0 for t in query_terms}

    if N == 0:
        return weighted

    # -----------------------------------------------------------------------
    # B2.1  Select expandable terms
    # -----------------------------------------------------------------------
    # POS-tag the surface tokens to identify nouns
    tagged = _pos_tag_query(surface_tokens)

    # Map surface_token → coarse POS for nouns we're willing to expand
    expandable: list[tuple[str, str, str]] = []  # (surface, stemmed, wn_pos)
    for (surface, coarse_pos), stemmed in zip(tagged, query_terms):
        # Only expand nouns (safest for topic preservation)
        if coarse_pos != "n":
            continue
        # Only expand terms that are already in the index (non-zero IDF boost)
        if stemmed not in inverted_index:
            continue
        # Only expand reasonably specific terms (IDF > threshold)
        term_idf = _idf(stemmed, inverted_index, N)
        if term_idf < 1.0:   # skip very generic terms
            continue
        expandable.append((surface, stemmed, wn.NOUN))

    # -----------------------------------------------------------------------
    # B2.2 → B2.4: For each expandable term, pick sense + synonyms + filter
    # -----------------------------------------------------------------------
    for surface, stemmed, wn_pos in expandable:
        # B2.2 WSD-lite sense selection
        synset = _wsd_lite(surface, wn_pos, surface_tokens)
        if synset is None:
            continue

        # B2.3 Collect synonym lemma names (synonyms only — no hypernyms/hyponyms)
        candidates: list[str] = []
        for lemma in synset.lemmas():
            cand = lemma.name().lower().replace("_", " ").replace("-", " ")
            # Keep only single-word candidates
            if " " in cand:
                continue
            if not _RE_ALPHA.match(cand):
                continue
            if cand == surface or cand == stemmed:
                continue
            candidates.append(cand)

        # B2.4 Filter candidates
        accepted: list[tuple[str, float]] = []  # (stemmed_cand, idf)
        for cand_surface in candidates:
            # Stopword / punctuation cleanup
            if cand_surface in _STOPWORDS:
                continue
            if len(cand_surface) < 2:
                continue

            # Stem the candidate so it matches index terms
            cand_stemmed = preprocess.terms(cand_surface)
            if not cand_stemmed:
                continue
            cand_stem = cand_stemmed[0]

            # Don't add if already in the query (even in stemmed form)
            if cand_stem in weighted:
                continue

            # High-DF filter: reject if candidate appears in >MAX_DF_RATIO of docs
            entry = inverted_index.get(cand_stem)
            if entry is None:
                continue
            df_ratio = entry[0] / N
            if df_ratio > config.MAX_DF_RATIO:
                continue

            # Co-occurrence filter
            if not _cooccurrence_ok(cand_stem, query_terms, inverted_index):
                continue

            cand_idf = _idf(cand_stem, inverted_index, N)
            accepted.append((cand_stem, cand_idf))

        # Sort by IDF descending (more specific expansions preferred) and cap
        accepted.sort(key=lambda x: x[1], reverse=True)
        accepted = accepted[: config.MAX_EXPANSIONS_PER_TERM]

        # B2.5 Weight expansions
        # Base weight = EXPANSION_GAMMA; optionally scale by relative IDF
        orig_idf = _idf(stemmed, inverted_index, N)
        for cand_stem, cand_idf in accepted:
            # IDF-relative scaling: expansion IDF / original IDF, clipped to [0.5, 1.0]
            idf_scale = min(1.0, max(0.5, cand_idf / orig_idf)) if orig_idf > 0 else 1.0
            weight = config.EXPANSION_GAMMA * idf_scale
            weighted[cand_stem] = weight

    return weighted
