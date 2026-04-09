from typing import Dict, List
import config


def get_variant_by_name(name: str) -> Dict:
    for variant in VARIANTS:
        if variant["name"] == name:
            return variant
    raise ValueError(f"Unknown variant: {name}")


VARIANTS: List[Dict] = [
    {
        "name": "BM25_flattened",
        "use_fields": False,
        "use_bm25f": False,
        "title_weight": 1.0,
        "body_weight": 1.0,
        "use_phrase_bonus": False,
        "use_proximity_bonus": False,
        "use_query_expansion": False,
        "use_neural_rerank": False,
        "rerank_depth": 0,
    },
    {
        "name": "BM25_separate_unweighted",
        "use_fields": True,
        "use_bm25f": False,
        "title_weight": 1.0,
        "body_weight": 1.0,
        "use_phrase_bonus": False,
        "use_proximity_bonus": False,
        "use_query_expansion": False,
        "use_neural_rerank": False,
        "rerank_depth": 0,
    },
    {
        "name": "BM25F",
        "use_fields": True,
        "use_bm25f": True,
        "title_weight": getattr(config, "W_TITLE", 2.0),
        "body_weight": getattr(config, "W_BODY", 1.0),
        "use_phrase_bonus": False,
        "use_proximity_bonus": False,
        "use_query_expansion": False,
        "use_neural_rerank": False,
        "rerank_depth": 0,
    },
    {
        "name": "BM25F_phrase_proximity",
        "use_fields": True,
        "use_bm25f": True,
        "title_weight": getattr(config, "W_TITLE", 2.0),
        "body_weight": getattr(config, "W_BODY", 1.0),
        "use_phrase_bonus": True,
        "use_proximity_bonus": True,
        "use_query_expansion": False,
        "use_neural_rerank": False,
        "rerank_depth": 0,
    },
    {
        "name": "BM25F_phrase_proximity_expand",
        "use_fields": True,
        "use_bm25f": True,
        "title_weight": getattr(config, "W_TITLE", 2.0),
        "body_weight": getattr(config, "W_BODY", 1.0),
        "use_phrase_bonus": True,
        "use_proximity_bonus": True,
        "use_query_expansion": True,
        "use_neural_rerank": False,
        "rerank_depth": 0,
    },
    {
        "name": "BM25F_phrase_proximity_expand_rerank50",
        "use_fields": True,
        "use_bm25f": True,
        "title_weight": getattr(config, "W_TITLE", 2.0),
        "body_weight": getattr(config, "W_BODY", 1.0),
        "use_phrase_bonus": True,
        "use_proximity_bonus": True,
        "use_query_expansion": True,
        "use_neural_rerank": True,
        "rerank_depth": 50,
    },
]

# Final chosen main retrieval system
DEFAULT_VARIANT = get_variant_by_name("BM25F_phrase_proximity_expand")