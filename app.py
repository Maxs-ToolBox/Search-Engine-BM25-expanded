"""
app.py — Field-Aware BM25F Search Engine (ECS736PU CW1/CW2)

Field-Aware BM25F Search Engine with Phrase/Proximity Modelling
and Controlled Thesaurus-Based Query Expansion

Run with:
    streamlit run app.py
"""

import os
import sys
import pickle
import subprocess
import time
import re
from typing import Dict, List, Optional, Tuple

import streamlit as st

import config
from search import load_index, process_query
from variants import VARIANTS, DEFAULT_VARIANT, get_variant_by_name


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BM25F Search Engine — ECS736PU",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---------- global typography ---------- */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ---------- hero header ---------- */
.hero-title {
    font-size: 1.85rem;
    font-weight: 800;
    color: #cdd6f4;
    letter-spacing: -0.02em;
    line-height: 1.2;
    margin-bottom: 4px;
}
.hero-sub {
    font-size: 0.82rem;
    color: #6c7086;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 2px;
}
.hero-team {
    font-size: 0.78rem;
    color: #585b70;
}

/* ---------- query chip row ---------- */
.chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 16px 0; }
.q-chip {
    display: inline-block;
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.78rem;
    color: #89b4fa;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
}
.q-chip:hover { background: #313244; border-color: #89b4fa; }

/* ---------- result card ---------- */
.result-card {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 12px;
    padding: 16px 20px 14px 20px;
    margin-bottom: 12px;
    transition: border-color 0.15s;
}
.result-card:hover { border-color: #89b4fa; }

/* card header row */
.card-header {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 6px;
}
.rank-badge {
    font-size: 0.78rem; font-weight: 700;
    color: #a6adc8; background: #181825;
    border: 1px solid #313244; border-radius: 6px;
    padding: 2px 8px; white-space: nowrap; flex-shrink: 0;
    margin-top: 2px;
}
.card-title-block { flex: 1; min-width: 0; }
.card-title {
    font-size: 0.97rem; font-weight: 600;
    color: #cdd6f4; line-height: 1.35;
    margin-bottom: 2px;
}
.card-docno {
    font-family: 'Courier New', monospace;
    font-size: 0.75rem; color: #585b70;
}
.source-badge {
    font-size: 0.70rem; font-weight: 700;
    border-radius: 999px; padding: 2px 9px;
    white-space: nowrap; flex-shrink: 0;
    margin-top: 3px; opacity: 0.95;
}
.score-block { text-align: right; flex-shrink: 0; }
.score-value {
    font-size: 0.88rem; font-weight: 700;
    color: #cdd6f4; display: block;
}
.score-label {
    font-size: 0.68rem; font-weight: 600;
    border-radius: 999px; padding: 1px 7px;
    display: inline-block; margin-top: 2px;
}
.label-strong { background: #1e3a2e; color: #a6e3a1; }
.label-good   { background: #3a3020; color: #f9e2af; }
.label-weak   { background: #2e1e1e; color: #f38ba8; }

/* score bar */
.score-bar-track {
    height: 3px; background: #313244;
    border-radius: 99px; margin: 8px 0 10px 0; overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #89b4fa 0%, #a6e3a1 100%);
    border-radius: 99px;
}

/* snippet */
.snippet {
    font-size: 0.875rem; color: #a6adc8;
    line-height: 1.6; margin-top: 4px;
}
.snippet em { color: #45475a; font-style: italic; }
.snippet mark {
    background: #2a2d3e; color: #f9e2af;
    border-radius: 3px; padding: 0 2px;
}

/* ---------- results summary bar ---------- */
.results-summary {
    font-size: 0.82rem; color: #6c7086;
    margin-bottom: 14px; padding-bottom: 10px;
    border-bottom: 1px solid #1e1e2e;
}

/* ---------- dataset badge strip ---------- */
.ds-badge {
    display: inline-block;
    background: #181825; border: 1px solid #313244;
    border-radius: 6px; padding: 4px 10px;
    font-size: 0.73rem; color: #89b4fa;
    margin: 3px 4px 3px 0;
}

/* ---------- score info box ---------- */
.score-info-box {
    background: #181825; border: 1px solid #313244;
    border-radius: 8px; padding: 10px 14px;
    font-size: 0.80rem; color: #a6adc8; line-height: 1.6;
}
.score-info-box strong { color: #cdd6f4; }

/* ---------- feature chips (sidebar) ---------- */
.feature-chip {
    display: inline-block; font-size: 0.72rem; font-weight: 600;
    border-radius: 6px; padding: 2px 8px;
    margin: 2px 3px 2px 0;
}
.chip-on  { background: #1e3a2e; color: #a6e3a1; border: 1px solid #2e5a3e; }
.chip-off { background: #2a1e2e; color: #f38ba8; border: 1px solid #5a2e3e; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Constants — variant metadata
# ---------------------------------------------------------------------------
_VARIANT_LABELS_KNOWN: Dict[str, str] = {
    "BM25_flattened":                        "BM25 Flattened  (baseline)",
    "BM25_separate_unweighted":              "BM25 Separate Fields  (unweighted)",
    "BM25F":                                 "BM25F  (field-weighted)",
    "BM25F_phrase_proximity":                "BM25F + Phrase & Proximity",
    "BM25F_phrase_proximity_expand":         "BM25F + Phrase/Prox + WordNet  ⭐ best",
    "BM25F_phrase_proximity_expand_rerank50":"BM25F + Phrase/Prox + WordNet + Neural Rerank",
}

def _variant_label(name: str) -> str:
    """Return a human-readable label for any variant name, with a safe fallback."""
    return _VARIANT_LABELS_KNOWN.get(name, name.replace("_", " "))

# Public alias used throughout the file
VARIANT_LABELS: Dict[str, str] = {v["name"]: _variant_label(v["name"]) for v in VARIANTS}

VARIANT_DESCRIPTIONS: Dict[str, str] = {
    "BM25_flattened":
        "Title and body merged into one field before scoring. "
        "Simplest baseline — loses all field-weight information.",
    "BM25_separate_unweighted":
        "Title and body scored separately then summed with equal weight. "
        "Captures field structure but does not boost title matches.",
    "BM25F":
        "BM25F (Zaragoza 2004): title weighted ×5 over body. "
        "Field-aware normalisation gives better ranking precision.",
    "BM25F_phrase_proximity":
        "BM25F plus a phrase bonus when query terms appear consecutively, "
        "and a proximity bonus when terms appear within an 8-word window.",
    "BM25F_phrase_proximity_expand":
        "Full system: BM25F + phrase/proximity bonuses + WordNet query "
        "expansion (γ=0.3). Expanded synonyms weighted at 0.3 vs originals. "
        "Best MAP and nDCG@10 of all lexical variants.",
    "BM25F_phrase_proximity_expand_rerank50":
        "Full lexical system + neural cross-encoder reranking applied to the "
        "top 50 candidates. Stage C of the pipeline (optional). "
        "Note: reranking reduced MAP in evaluation — lexical variant preferred.",
}

def _variant_description(name: str) -> str:
    """Return a description for any variant, with a safe fallback."""
    return VARIANT_DESCRIPTIONS.get(
        name,
        name.replace("_", " ") + " — custom variant.",
    )

FEATURE_FLAGS = [
    ("use_fields",          "Field-aware"),
    ("use_bm25f",           "BM25F weighted"),
    ("use_phrase_bonus",    "Phrase bonus"),
    ("use_proximity_bonus", "Proximity bonus"),
    ("use_query_expansion", "WordNet expansion"),
]

# ---------------------------------------------------------------------------
# Constants — dataset descriptions
# ---------------------------------------------------------------------------
DATASETS = [
    {
        "code": "FT",
        "name": "Financial Times",
        "years": "1992–1994",
        "docs": "~210,000",
        "color": "#1a3a5e",
        "badge_color": "#3a6aae",
        "desc": "Business and financial news from the Financial Times newspaper. "
                "Covers markets, companies, international economics, and politics.",
    },
    {
        "code": "FR94",
        "name": "Federal Register",
        "years": "1994",
        "docs": "~55,000",
        "color": "#1e3a1a",
        "badge_color": "#3a7a3a",
        "desc": "Official US government regulatory notices, proposed rules, and "
                "executive orders published in 1994. Dense legal and technical language.",
    },
    {
        "code": "CR",
        "name": "Congressional Record",
        "years": "1993",
        "docs": "~51,000",
        "color": "#3a1a2e",
        "badge_color": "#8a3a6e",
        "desc": "Verbatim record of proceedings and debates from the 103rd US Congress. "
                "Includes floor speeches, amendments, and committee reports.",
    },
    {
        "code": "FBIS",
        "name": "Foreign Broadcast Information Service",
        "years": "1996",
        "docs": "~130,000",
        "color": "#1a2e3e",
        "badge_color": "#3a6e8e",
        "desc": "Translated foreign news monitored and broadcast by the US government "
                "intelligence service. Covers news from Asia, Europe, Middle East, and Africa.",
    },
    {
        "code": "LA",
        "name": "LA Times",
        "years": "1989–1990",
        "docs": "~131,000",
        "color": "#3a2a0a",
        "badge_color": "#8a6a2a",
        "desc": "General news, politics, sports, and culture from the Los Angeles Times. "
                "Wide topic breadth makes it one of the most query-diverse collections.",
    },
]

_SOURCE_MAP: Dict[str, Tuple[str, str]] = {
    ds["code"]: (ds["name"], ds["badge_color"]) for ds in DATASETS
}

# ---------------------------------------------------------------------------
# Constants — example queries from TREC Robust04 topics
# ---------------------------------------------------------------------------
EXAMPLE_QUERIES = [
    ("Hubble Telescope Achievements",    "303"),
    ("Endangered Species Mammals",       "304"),
    ("International Organized Crime",    "301"),
    ("Industrial Espionage Trade Secrets","311"),
    ("Radio Waves Brain Cancer",         "310"),
    ("New Hydroelectric Projects",       "307"),
    ("African Civilian Deaths War",      "306"),
    ("Implant Dentistry Advantages",     "308"),
    ("Poliomyelitis Post-Polio",         "302"),
    ("Vehicle Crashworthiness Safety",   "305"),
]


# ---------------------------------------------------------------------------
# Helpers — source badge
# ---------------------------------------------------------------------------
def _get_source(docno: str) -> Tuple[str, str]:
    prefix = docno[:2].upper()
    return _SOURCE_MAP.get(prefix, ("Unknown", "#313244"))


# ---------------------------------------------------------------------------
# Helpers — pseudo-title from snippet
# ---------------------------------------------------------------------------
def _extract_title(snippet: str, max_chars: int = 90) -> str:
    """
    Extract the first sentence of a snippet to use as a display title.
    Falls back to a word-boundary cut at max_chars.
    """
    if not snippet:
        return ""
    for delim in [". ", "! ", "? ", "; "]:
        idx = snippet.find(delim)
        if 5 < idx <= max_chars:
            return snippet[:idx + 1].strip()
    if len(snippet) <= max_chars:
        return snippet.strip()
    cut = snippet[:max_chars].rsplit(" ", 1)[0]
    return cut.strip() + "…"


def _remaining_snippet(snippet: str, title: str) -> str:
    """
    Return the snippet text after the title portion (for the body preview).
    """
    if title and snippet.startswith(title.rstrip("…").rstrip(".")):
        rest = snippet[len(title.rstrip("…").rstrip(".")) :].lstrip(". ").strip()
        return rest
    return snippet


# ---------------------------------------------------------------------------
# Helpers — term highlighting
# ---------------------------------------------------------------------------
def _highlight_terms(text: str, stemmed_terms: set) -> str:
    if not text or not stemmed_terms:
        return text or ""
    try:
        import preprocess as _pp

        def _replace(match):
            word = match.group(0)
            norm = _pp.normalise(word)
            if norm:
                _, stemmed, _ = norm[0]
                if stemmed in stemmed_terms:
                    return f"<mark>{word}</mark>"
            return word

        return re.sub(r"\b[A-Za-z']+\b", _replace, text)
    except Exception:
        return text


def _truncate(text: str, max_chars: int = 240) -> str:
    if not text or len(text) <= max_chars:
        return text or ""
    return text[:max_chars].rsplit(" ", 1)[0] + " …"


# ---------------------------------------------------------------------------
# Helpers — score quality label
# ---------------------------------------------------------------------------
def _score_quality(score: float, max_score: float) -> Tuple[str, str]:
    ratio = score / max_score if max_score > 0 else 0
    if ratio >= 0.70:
        return "Strong", "label-strong"
    elif ratio >= 0.40:
        return "Good", "label-good"
    else:
        return "Weak", "label-weak"


# ---------------------------------------------------------------------------
# Index loading — cached
# ---------------------------------------------------------------------------
def _index_exists() -> bool:
    return all(os.path.exists(p) for p in [
        config.INDEX_FILE, config.DOC_MAP_FILE,
        config.DOC_STATS_FILE, config.COLL_STATS_FILE,
    ])


@st.cache_resource(show_spinner="⏳ Loading index…")
def _load_index_cached():
    def _pkl(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)

    inverted_index   = _pkl(config.INDEX_FILE)
    doc_map          = _pkl(config.DOC_MAP_FILE)
    doc_stats        = _pkl(config.DOC_STATS_FILE)
    collection_stats = _pkl(config.COLL_STATS_FILE)

    snippets: Optional[List[str]] = None
    if hasattr(config, "SNIPPETS_FILE") and os.path.exists(config.SNIPPETS_FILE):
        snippets = _pkl(config.SNIPPETS_FILE)

    # O(1) docno → doc_id lookup for snippet retrieval
    docno_to_id: Dict[str, int] = {docno: i for i, docno in enumerate(doc_map)}

    return inverted_index, doc_map, doc_stats, collection_stats, snippets, docno_to_id


def _get_snippet(docno: str, snippets, docno_to_id) -> str:
    if snippets is None:
        return ""
    doc_id = docno_to_id.get(docno)
    if doc_id is None or doc_id >= len(snippets):
        return ""
    return snippets[doc_id] or ""


# ---------------------------------------------------------------------------
# Session state — clicked example query
# ---------------------------------------------------------------------------
if "example_query" not in st.session_state:
    st.session_state["example_query"] = ""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔍 Search Engine")
    st.caption("ECS736PU · Information Retrieval")
    st.divider()

    # Index status
    if _index_exists():
        st.success("✅ Index ready")
    else:
        st.warning("⚠️ Index not built yet")

    with st.expander("🔨 Build Index", expanded=not _index_exists()):
        st.caption("Run once on the TREC dataset (10–30 min).")
        if st.button("Build Index", type="primary", use_container_width=True):
            st.info("Building index — streaming live output…")
            log_box = st.empty()
            log_text = ""
            process = subprocess.Popen(
                [sys.executable, "-u", "build_index.py"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            for line in iter(process.stdout.readline, ""):
                log_text += line
                log_box.code(log_text[-3000:], language=None)
            process.wait()
            if process.returncode == 0:
                st.success("Done. Reload the page to search.")
                st.cache_resource.clear()
            else:
                st.error("Build failed — see log above.")

    st.divider()

    # ── Variant selector ──
    st.markdown("### Retrieval Variant")

    variant_display = [VARIANT_LABELS[v["name"]] for v in VARIANTS]
    default_display = VARIANT_LABELS[DEFAULT_VARIANT["name"]]
    default_idx     = variant_display.index(default_display)

    selected_display = st.selectbox(
        "Variant",
        options=variant_display,
        index=default_idx,
        label_visibility="collapsed",
        help="⭐ marks the best-performing variant from ablation evaluation.",
    )
    selected_name    = [v["name"] for v in VARIANTS
                        if VARIANT_LABELS[v["name"]] == selected_display][0]
    selected_variant = get_variant_by_name(selected_name)

    # Feature chips
    st.markdown(
        " ".join(
            f'<span class="feature-chip {"chip-on" if selected_variant[flag] else "chip-off"}">'
            f'{"✓" if selected_variant[flag] else "✗"} {label}</span>'
            for flag, label in FEATURE_FLAGS
        ),
        unsafe_allow_html=True,
    )
    st.caption(_variant_description(selected_name))

    st.divider()

    # ── Parameters ──
    st.markdown("### Parameters")
    top_k = st.slider("Results to show", min_value=5, max_value=100, value=10, step=5)

    show_expansion = st.toggle(
        "Show expansion details",
        value=False,
        disabled=not selected_variant["use_query_expansion"],
        help="Show which WordNet synonyms were added and their weights.",
    )
    show_debug = st.toggle(
        "Show query debug info",
        value=False,
        help="Show normalised query terms and timing breakdown.",
    )

    st.divider()

    # ── Score guide ──
    st.markdown("### Score Guide")
    st.markdown("""
<div class="score-info-box">
BM25F scores are <strong>not normalised</strong> — they reflect evidence
accumulation and are comparable only within one result set.<br><br>
<span class="score-label label-strong">Strong</span> Top ~30% of results
for this query<br>
<span class="score-label label-good">Good</span> Mid-range results<br>
<span class="score-label label-weak">Weak</span> Low overlap with query<br><br>
Relative <em>rank</em> matters more than absolute score value.
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.caption(
        "**Stemmer:** Porter · stopwords removed\n\n"
        "**Fields:** title ×5 + body ×1\n\n"
        f"**k₁={config.K1}  b_t={config.B_TITLE}  b_b={config.B_BODY}**"
    )


# ---------------------------------------------------------------------------
# Main area — hero header
# ---------------------------------------------------------------------------
st.markdown("""
<div style="margin-bottom: 6px;">
  <div class="hero-sub">ECS736PU · Information Retrieval · Queen Mary University of London</div>
  <div class="hero-title">
    Field-Aware BM25F Search Engine
  </div>
  <div style="font-size:0.82rem; color:#6c7086; margin-top:2px;">
    Phrase/Proximity Modelling · Controlled Thesaurus-Based Query Expansion
  </div>
  <div class="hero-team" style="margin-top:6px;">
    Blazej Olszta · Muhamad Husaam Ateeq · Max Monaghan · Sulaiman Bhatti
  </div>
</div>
""", unsafe_allow_html=True)

if not _index_exists():
    st.warning(
        "The index has not been built yet. "
        "Open **Build Index** in the sidebar to get started."
    )
    st.stop()

# Load index (cached after first run)
inverted_index, doc_map, doc_stats, collection_stats, snippets, docno_to_id = (
    _load_index_cached()
)

N = collection_stats["N"]

# ── Dataset info strip ──
with st.expander(
    f"📚 {N:,} documents indexed across 5 collections — click for dataset details",
    expanded=False,
):
    for ds in DATASETS:
        st.markdown(
            f"**{ds['name']}** ({ds['code']}) &nbsp;·&nbsp; "
            f"{ds['years']} &nbsp;·&nbsp; {ds['docs']} docs  \n"
            f"{ds['desc']}",
        )
        st.markdown("---")
    st.caption(
        "Collection: **TREC Disk 4 & 5 (Robust04)**. "
        "Used in TREC Robust Track 2004 to evaluate retrieval robustness "
        "across 249 topics (301–700). Queries range from single-word to "
        "complex multi-concept topics."
    )

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Search box
# ---------------------------------------------------------------------------
col_input, col_btn = st.columns([7, 1])
with col_input:
    # Allow example query clicks to pre-fill the box
    default_text = st.session_state.get("example_query", "")
    query = st.text_input(
        label="Search query",
        value=default_text,
        placeholder="e.g.  Hubble Telescope Achievements  ·  industrial espionage trade secrets",
        label_visibility="collapsed",
        key="query_input",
    )
with col_btn:
    search_clicked = st.button("Search", type="primary", use_container_width=True)

# ── Example query chips ──
st.markdown(
    "<div style='font-size:0.75rem; color:#6c7086; margin:6px 0 4px 0;'>"
    "Try a TREC Robust04 topic:</div>",
    unsafe_allow_html=True,
)

# Render example chips as buttons in a compact row
chip_cols = st.columns(len(EXAMPLE_QUERIES))
for idx, (q_text, q_num) in enumerate(EXAMPLE_QUERIES):
    with chip_cols[idx]:
        if st.button(
            q_text,
            key=f"chip_{q_num}",
            help=f"TREC topic {q_num}: {q_text}",
            use_container_width=True,
        ):
            st.session_state["example_query"] = q_text
            st.rerun()

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Execute search
# ---------------------------------------------------------------------------
active_query = query or st.session_state.get("example_query", "")

if active_query and (search_clicked or active_query):

    t0 = time.time()
    ranked = process_query(
        query_text=active_query,
        inverted_index=inverted_index,
        doc_map=doc_map,
        doc_stats=doc_stats,
        collection_stats=collection_stats,
        top_k=top_k,
        variant_config=selected_variant,
        debug=False,
    )
    elapsed_ms = (time.time() - t0) * 1000

    # ── Stemmed terms for highlighting ──
    try:
        import preprocess as _pp
        norm = _pp.normalise(active_query)
        _stems = set(st_t for _, st_t, _ in norm)
    except Exception:
        _stems = set()

    # ── Debug info ──
    if show_debug:
        try:
            import preprocess as _pp
            norm = _pp.normalise(active_query)
            stemmed_display = [st_t for _, st_t, _ in norm]
        except Exception:
            stemmed_display = []
        with st.expander("🔬 Query debug", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Results", len(ranked))
            c2.metric("Time (ms)", f"{elapsed_ms:.0f}")
            c3.metric("Variant", selected_name.split("_")[0])
            if stemmed_display:
                st.markdown(
                    "**Normalised stems:** "
                    + "  ".join(f"`{t}`" for t in stemmed_display)
                )

    # ── Expansion details ──
    if show_expansion and selected_variant["use_query_expansion"]:
        try:
            import preprocess as _pp
            import query_expand as _qe
            norm = _pp.normalise(active_query)
            surface_tokens = [s for s, _, _ in norm]
            original_terms = list(dict.fromkeys(st_t for _, st_t, _ in norm))
            term_weights = _qe.expand_query(
                original_terms, surface_tokens, inverted_index, collection_stats
            )
            orig_terms = {t: w for t, w in term_weights.items() if w >= 1.0}
            exp_terms  = {t: w for t, w in term_weights.items() if w <  1.0}
            with st.expander(
                f"📖 WordNet expansion — {len(exp_terms)} synonym(s) added",
                expanded=True,
            ):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Original terms** `weight 1.0`")
                    for t in orig_terms:
                        st.code(t, language=None)
                with c2:
                    gamma = getattr(config, "EXPANSION_GAMMA", 0.3)
                    st.markdown(f"**Expanded synonyms** `weight {gamma}`")
                    if exp_terms:
                        for t, w in sorted(exp_terms.items(), key=lambda x: -x[1]):
                            st.code(f"{t}  ·  {w:.3f}", language=None)
                    else:
                        st.caption("No synonyms found for this query.")
        except Exception as e:
            st.warning(f"Could not show expansion details: {e}")

    # ── No results ──
    if not ranked:
        st.info("No results found. Try different terms or switch retrieval variant.")
        st.stop()

    max_score = ranked[0][0] if ranked else 1.0

    # ── Results summary ──
    st.markdown(
        f'<div class="results-summary">'
        f'<b>{len(ranked)}</b> results for <b>"{active_query}"</b>'
        f'&nbsp;·&nbsp;{elapsed_ms:.0f} ms'
        f'&nbsp;·&nbsp;variant: <b>{selected_name.replace("_", " ")}</b>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Result cards ──
    for rank_pos, (score, docno) in enumerate(ranked, start=1):

        raw_snippet  = _get_snippet(docno, snippets, docno_to_id)
        pseudo_title = _extract_title(raw_snippet)
        body_preview = _truncate(_remaining_snippet(raw_snippet, pseudo_title))
        hl_title     = _highlight_terms(pseudo_title, _stems)
        hl_body      = _highlight_terms(body_preview, _stems)

        source_label, badge_color = _get_source(docno)
        score_pct = min(100, int((score / max_score) * 100)) if max_score > 0 else 0
        quality_text, quality_cls = _score_quality(score, max_score)

        # Snippet content
        snippet_html = (
            f'<span class="card-title">{hl_title}</span><br>'
            if pseudo_title else ""
        )
        if hl_body:
            snippet_html += f'<div class="snippet">{hl_body}</div>'
        elif not pseudo_title:
            snippet_html += '<div class="snippet"><em>No preview available.</em></div>'

        card_html = f"""
<div class="result-card">
  <div class="card-header">
    <span class="rank-badge">#{rank_pos}</span>
    <div class="card-title-block">
      {snippet_html}
      <span class="card-docno">{docno}</span>
    </div>
    <span class="source-badge" style="background:{badge_color};color:#e0e0f0;">
      {source_label}
    </span>
    <div class="score-block">
      <span class="score-value">{score:.4f}</span>
      <span class="score-label {quality_cls}">{quality_text}</span>
    </div>
  </div>
  <div class="score-bar-track">
    <div class="score-bar-fill" style="width:{score_pct}%;"></div>
  </div>
</div>
"""
        st.markdown(card_html, unsafe_allow_html=True)

    # ── Footer ──
    st.markdown("---")
    score_params = (
        f"BM25F · k₁={config.K1} · b_title={config.B_TITLE} · b_body={config.B_BODY} · "
        f"w_title={config.W_TITLE}"
        + (f" · phrase λ={config.PHRASE_BONUS} · prox window={config.PROXIMITY_WINDOW}"
           if selected_variant["use_phrase_bonus"] else "")
        + (f" · expansion γ={getattr(config, 'EXPANSION_GAMMA', 0.3)}"
           if selected_variant["use_query_expansion"] else "")
    )
    st.caption(
        f"Showing top {len(ranked)} of up to {top_k} candidates · {score_params} · "
        "Snippet = first 200 chars of document body · "
        "Title = first sentence of snippet"
    )
