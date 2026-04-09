from typing import List, Tuple, Optional
from sentence_transformers import CrossEncoder


# Good small/strong default reranker
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model_cache = {}


def load_reranker(model_name: str = DEFAULT_RERANK_MODEL) -> CrossEncoder:
    """
    Load and cache the cross-encoder reranker model.
    """
    if model_name not in _model_cache:
        _model_cache[model_name] = CrossEncoder(model_name)
    return _model_cache[model_name]


def build_doc_text(
    doc_id: int,
    doc_map: List[str],
    doc_store: Optional[dict] = None,
    snippets: Optional[list] = None,
) -> str:
    """
    Build text for reranking.
    Priority:
    1. doc_store full text if available
    2. snippets if available
    3. docno fallback
    """
    if doc_store is not None and doc_id in doc_store:
        record = doc_store[doc_id]
        title = record.get("title", "") if isinstance(record, dict) else ""
        body = record.get("body", "") if isinstance(record, dict) else ""
        return f"{title} {body}".strip()

    if snippets is not None and doc_id < len(snippets):
        snippet = snippets[doc_id]
        if snippet:
            return str(snippet)

    return doc_map[doc_id]


def rerank_results(
    query_text: str,
    ranked_results: List[Tuple[float, int]],
    doc_map: List[str],
    rerank_depth: int = 50,
    model_name: str = DEFAULT_RERANK_MODEL,
    doc_store: Optional[dict] = None,
    snippets: Optional[list] = None,
) -> List[Tuple[float, int]]:
    """
    Rerank the top-N lexical results using a cross-encoder.
    Input ranked_results must be [(lexical_score, doc_id), ...]
    Output is [(rerank_score, doc_id), ...] with top-N reranked and the rest appended unchanged.
    """
    if not ranked_results:
        return []

    top = ranked_results[:rerank_depth]
    tail = ranked_results[rerank_depth:]

    model = load_reranker(model_name)

    pairs = []
    top_doc_ids = []

    for _, doc_id in top:
        doc_text = build_doc_text(
            doc_id=doc_id,
            doc_map=doc_map,
            doc_store=doc_store,
            snippets=snippets,
        )
        pairs.append((query_text, doc_text))
        top_doc_ids.append(doc_id)

    scores = model.predict(pairs)

    reranked_top = sorted(
        zip(scores, top_doc_ids),
        key=lambda x: x[0],
        reverse=True,
    )

    # Keep tail after reranked top
    return reranked_top + tail