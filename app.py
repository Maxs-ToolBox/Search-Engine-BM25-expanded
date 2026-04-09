"""
app.py  —  Field-Aware BM25F Search Engine
           Phrase/Proximity Modelling · Controlled Thesaurus-Based Query Expansion

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

import pandas as pd
import streamlit as st

import config
import parse_docs
from search import process_query
from variants import VARIANTS, DEFAULT_VARIANT, get_variant_by_name


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Field-Aware BM25F Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* ── hero ── */
.hero-title {
    font-size: 1.75rem;
    font-weight: 800;
    color: #cdd6f4;
    letter-spacing: -0.02em;
    line-height: 1.2;
    margin-bottom: 3px;
}
.hero-sub {
    font-size: 0.80rem;
    color: #6c7086;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 2px;
}
.hero-team {
    font-size: 0.76rem;
    color: #585b70;
    margin-top: 5px;
}

/* ── result card ── */
.result-card {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 12px;
    padding: 15px 18px 12px 18px;
    margin-bottom: 8px;
    transition: border-color 0.15s;
}
.result-card:hover {
    border-color: #89b4fa;
}

.card-header {
    display: flex;
    align-items: flex-start;
    gap: 12px;
}

.rank-badge {
    font-size: 0.75rem;
    font-weight: 700;
    color: #a6adc8;
    background: #181825;
    border: 1px solid #313244;
    border-radius: 6px;
    padding: 2px 7px;
    white-space: nowrap;
    flex-shrink: 0;
    margin-top: 2px;
}

.card-title-block {
    flex: 1;
    min-width: 0;
}

.card-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #cdd6f4;
    line-height: 1.35;
    margin-bottom: 4px;
}

.card-docno {
    font-family: 'Courier New', monospace;
    font-size: 0.73rem;
    color: #585b70;
    margin-top: 6px;
}

.snippet {
    font-size: 0.875rem;
    color: #a6adc8;
    line-height: 1.6;
    margin-top: 3px;
}
.snippet em {
    color: #45475a;
    font-style: italic;
}
.snippet mark,
.card-title mark {
    background: #2a2d3e;
    color: #f9e2af;
    border-radius: 3px;
    padding: 0 2px;
}

.card-side {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 6px;
    flex-shrink: 0;
}

.badge-row {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    justify-content: flex-end;
}

.source-badge {
    font-size: 0.68rem;
    font-weight: 700;
    border-radius: 999px;
    padding: 2px 9px;
    white-space: nowrap;
    opacity: 0.95;
    color: #e0e0f0;
}

.rel-badge-yes {
    font-size: 0.68rem;
    font-weight: 700;
    border-radius: 999px;
    padding: 2px 9px;
    background: #1e3a2e;
    color: #a6e3a1;
    border: 1px solid #2e5a3e;
    white-space: nowrap;
}

.rel-badge-no {
    font-size: 0.68rem;
    font-weight: 700;
    border-radius: 999px;
    padding: 2px 9px;
    background: #2a1a1a;
    color: #f38ba8;
    border: 1px solid #5a2e2e;
    white-space: nowrap;
}

.score-block {
    text-align: right;
}

.score-value {
    font-size: 0.90rem;
    font-weight: 700;
    color: #cdd6f4;
    display: block;
}

.score-label {
    font-size: 0.66rem;
    font-weight: 600;
    border-radius: 999px;
    padding: 1px 7px;
    display: inline-block;
    margin-top: 2px;
}
.label-strong { background: #1e3a2e; color: #a6e3a1; }
.label-good   { background: #3a3020; color: #f9e2af; }
.label-weak   { background: #2e1e1e; color: #f38ba8; }

.score-bar-track {
    height: 3px;
    background: #313244;
    border-radius: 99px;
    margin: 10px 0 6px 0;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #89b4fa 0%, #a6e3a1 100%);
    border-radius: 99px;
}

/* ── article viewer ── */
.article-panel {
    background: #181825;
    border: 1px solid #45475a;
    border-radius: 10px;
    padding: 18px 22px;
    margin: 6px 0 14px 0;
}
.article-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #cdd6f4;
    margin-bottom: 10px;
}
.article-body {
    font-size: 0.875rem;
    color: #bac2de;
    line-height: 1.7;
    white-space: pre-wrap;
}

/* ── results summary ── */
.results-summary {
    font-size: 0.80rem;
    color: #6c7086;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e1e2e;
}

/* ── topic description box ── */
.topic-desc-box {
    background: #181825;
    border-left: 3px solid #89b4fa;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: #a6adc8;
    line-height: 1.6;
    margin-bottom: 10px;
}

/* ── score info box (sidebar) ── */
.score-info-box {
    background: #181825;
    border: 1px solid #313244;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.78rem;
    color: #a6adc8;
    line-height: 1.7;
}
.score-info-box strong { color: #cdd6f4; }

/* ── feature chips ── */
.feature-chip {
    display: inline-block;
    font-size: 0.70rem;
    font-weight: 600;
    border-radius: 6px;
    padding: 2px 7px;
    margin: 2px 2px 2px 0;
}
.chip-on  { background: #1e3a2e; color: #a6e3a1; border: 1px solid #2e5a3e; }
.chip-off { background: #2a1e2e; color: #f38ba8; border: 1px solid #5a2e3e; }

/* ── search history ── */
.history-item {
    display: inline-block;
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 0.75rem;
    color: #a6adc8;
    margin: 2px 3px;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Variant metadata
# ─────────────────────────────────────────────────────────────────────────────
_VARIANT_LABELS_KNOWN: Dict[str, str] = {
    "BM25_flattened": "BM25 Flattened  (baseline)",
    "BM25_separate_unweighted": "BM25 Separate Fields  (unweighted)",
    "BM25F": "BM25F  (field-weighted)",
    "BM25F_phrase_proximity": "BM25F + Phrase & Proximity",
    "BM25F_phrase_proximity_expand": "BM25F + Phrase/Prox + WordNet  (Best lexical variant)",
    "BM25F_phrase_proximity_expand_rerank50": "BM25F + Phrase/Prox + WordNet + Neural Rerank",
}


def _variant_label(name: str) -> str:
    return _VARIANT_LABELS_KNOWN.get(name, name.replace("_", " "))


VARIANT_LABELS: Dict[str, str] = {v["name"]: _variant_label(v["name"]) for v in VARIANTS}

_VARIANT_DESC: Dict[str, str] = {
    "BM25_flattened":
        "Title and body merged into one field. Simplest baseline — loses all field-weight information.",
    "BM25_separate_unweighted":
        "Title and body scored separately, summed with equal weight. Captures field structure but does not boost title matches.",
    "BM25F":
        "BM25F (Zaragoza 2004): title weighted ×5 over body. Field-aware normalisation — better ranking precision than unweighted.",
    "BM25F_phrase_proximity":
        "BM25F + phrase bonus when query terms appear consecutively, and proximity bonus when terms appear within an 8-word window.",
    "BM25F_phrase_proximity_expand":
        "Full system: BM25F + phrase/proximity + WordNet expansion (γ=0.3). Best MAP and nDCG@10 across all lexical variants.",
    "BM25F_phrase_proximity_expand_rerank50":
        "Full lexical system + neural cross-encoder reranking on top 50 candidates. Note: reranking reduced MAP in evaluation — lexical variant preferred.",
}


def _variant_desc(name: str) -> str:
    return _VARIANT_DESC.get(name, name.replace("_", " ") + " — custom variant.")


FEATURE_FLAGS = [
    ("use_fields", "Field-aware"),
    ("use_bm25f", "BM25F weighted"),
    ("use_phrase_bonus", "Phrase bonus"),
    ("use_proximity_bonus", "Proximity bonus"),
    ("use_query_expansion", "WordNet expansion"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Dataset descriptions
# ─────────────────────────────────────────────────────────────────────────────
DATASETS = [
    {"code": "FT", "name": "Financial Times", "years": "1992–1994", "docs": "~210,000", "badge_color": "#2a5a9e", "desc": "Business and financial news. Covers markets, companies, economics and politics."},
    {"code": "FR94", "name": "Federal Register", "years": "1994", "docs": "~55,000", "badge_color": "#3a7a3a", "desc": "US government regulatory notices, proposed rules and executive orders. Dense legal language."},
    {"code": "CR", "name": "Congressional Record", "years": "1993", "docs": "~51,000", "badge_color": "#7a3a6a", "desc": "Verbatim record of 103rd US Congress proceedings, floor speeches and amendments."},
    {"code": "FBIS", "name": "FBIS", "years": "1996", "docs": "~130,000", "badge_color": "#3a6e8e", "desc": "Translated foreign news from Asia, Europe, Middle East and Africa."},
    {"code": "LA", "name": "LA Times", "years": "1989–1990", "docs": "~131,000", "badge_color": "#8a6a2a", "desc": "General news, politics, sport and culture. Most query-diverse collection."},
]

_SOURCE_MAP: Dict[str, Tuple[str, str]] = {
    ds["code"]: (ds["name"], ds["badge_color"]) for ds in DATASETS
}

# ─────────────────────────────────────────────────────────────────────────────
# Example queries
# ─────────────────────────────────────────────────────────────────────────────
EXAMPLE_QUERIES = [
    ("Hubble Telescope Achievements", "303"),
    ("Endangered Species Mammals", "304"),
    ("International Organized Crime", "301"),
    ("Industrial Espionage Trade Secrets", "311"),
    ("Radio Waves Brain Cancer", "310"),
    ("New Hydroelectric Projects", "307"),
    ("African Civilian Deaths War", "306"),
    ("Implant Dentistry Advantages", "308"),
    ("Poliomyelitis Post-Polio", "302"),
    ("Vehicle Crashworthiness Safety", "305"),
]

_BASE = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────────────────────
def _index_exists() -> bool:
    return all(os.path.exists(p) for p in [
        config.INDEX_FILE,
        config.DOC_MAP_FILE,
        config.DOC_STATS_FILE,
        config.COLL_STATS_FILE,
    ])


@st.cache_resource(show_spinner="Loading index…")
def _load_index_cached():
    def _pkl(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)

    inverted_index = _pkl(config.INDEX_FILE)
    doc_map = _pkl(config.DOC_MAP_FILE)
    doc_stats = _pkl(config.DOC_STATS_FILE)
    collection_stats = _pkl(config.COLL_STATS_FILE)

    snippets: Optional[List[str]] = None
    if hasattr(config, "SNIPPETS_FILE") and os.path.exists(config.SNIPPETS_FILE):
        snippets = _pkl(config.SNIPPETS_FILE)

    docno_to_id: Dict[str, int] = {docno: i for i, docno in enumerate(doc_map)}
    return inverted_index, doc_map, doc_stats, collection_stats, snippets, docno_to_id


@st.cache_data(show_spinner=False)
def _load_qrels() -> Dict[str, set]:
    path = config.QRELS_FILE
    if not os.path.exists(path):
        return {}
    qrels: Dict[str, set] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 4:
                topic, _, docno, relevance = parts[:4]
                if int(relevance) > 0:
                    qrels.setdefault(topic, set()).add(docno)
    return qrels


@st.cache_data(show_spinner=False)
def _load_topics() -> Dict[str, Dict[str, str]]:
    path = config.TOPICS_FILE
    if not os.path.exists(path):
        return {}
    topics: Dict[str, Dict[str, str]] = {}
    current: Dict[str, str] = {}
    section = None

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip()
            if "<num>" in line.lower():
                m = re.search(r"Number:\s*(\d+)", line, re.IGNORECASE)
                if m:
                    current = {"num": m.group(1), "title": "", "desc": ""}
                    section = None
            elif "<title>" in line.lower():
                current["title"] = re.sub(r"<title>\s*", "", line, flags=re.IGNORECASE).strip()
                section = "title"
            elif "<desc>" in line.lower():
                section = "desc"
            elif "<narr>" in line.lower() or "</top>" in line.lower():
                section = None
                if current.get("num"):
                    topics[current["num"]] = current.copy()
            elif section == "title" and line.strip() and not line.startswith("<"):
                current["title"] += " " + line.strip()
            elif section == "desc" and line.strip() and not line.startswith("<"):
                current["desc"] += " " + line.strip()

    return topics


@st.cache_data(show_spinner=False)
def _load_eval_results() -> Optional[pd.DataFrame]:
    path = os.path.join(_BASE, "evaluation_results.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=86400)
def _find_document(docno: str) -> Optional[dict]:
    prefix4 = docno[:4].upper()
    prefix2 = docno[:2].upper()

    if prefix4 == "FBIS":
        ctype = "FBIS"
    elif prefix2 == "LA":
        ctype = "LATIMES"
    elif prefix2 == "FT":
        ctype = "FT"
    elif prefix2 == "FR":
        ctype = "FR94"
    elif prefix2 == "CR":
        ctype = "CR"
    else:
        return None

    root_dir = None
    for rdir, ct in config.COLLECTIONS:
        if ct == ctype:
            root_dir = rdir
            break

    if not root_dir or not os.path.isdir(root_dir):
        return None

    parser = parse_docs._PARSERS.get(ctype)
    if parser is None:
        return None

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d.upper() != "DTDS"]
        for fname in sorted(filenames):
            if parse_docs._should_skip(fname):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                text = parse_docs._read_file(fpath)
                if not text or docno not in text:
                    continue
                for raw in parse_docs._split_docs(text):
                    doc = parser(raw)
                    if doc and doc.get("docno", "").strip() == docno:
                        return doc
            except Exception:
                continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _get_source(docno: str) -> Tuple[str, str]:
    prefix = docno[:4].upper() if docno[:4].upper() == "FBIS" else docno[:2].upper()
    return _SOURCE_MAP.get(prefix, _SOURCE_MAP.get(docno[:2].upper(), ("Unknown", "#313244")))


def _get_snippet(docno: str, snippets, docno_to_id) -> str:
    if snippets is None:
        return ""
    doc_id = docno_to_id.get(docno)
    if doc_id is None or doc_id >= len(snippets):
        return ""
    return snippets[doc_id] or ""


def _extract_title(snippet: str, max_chars: int = 90) -> str:
    if not snippet:
        return ""
    for delim in [". ", "! ", "? ", "; "]:
        idx = snippet.find(delim)
        if 5 < idx <= max_chars:
            return snippet[:idx + 1].strip()
    if len(snippet) <= max_chars:
        return snippet.strip()
    return snippet[:max_chars].rsplit(" ", 1)[0].strip() + "…"


def _remaining_snippet(snippet: str, title: str) -> str:
    if title:
        base = title.rstrip("…").rstrip(".")
        if snippet.startswith(base):
            return snippet[len(base):].lstrip(". ").strip()
    return snippet


def _highlight_terms(text: str, stems: set) -> str:
    if not text or not stems:
        return text or ""
    try:
        import preprocess as _pp

        def _sub(m):
            word = m.group(0)
            n = _pp.normalise(word)
            return f"<mark>{word}</mark>" if n and n[0][1] in stems else word

        return re.sub(r"\b[A-Za-z']+\b", _sub, text)
    except Exception:
        return text


def _truncate(text: str, n: int = 240) -> str:
    if not text or len(text) <= n:
        return text or ""
    return text[:n].rsplit(" ", 1)[0] + " …"


def _score_quality(score: float, max_score: float) -> Tuple[str, str]:
    r = score / max_score if max_score > 0 else 0
    if r >= 0.70:
        return "Strong", "label-strong"
    if r >= 0.40:
        return "Good", "label-good"
    return "Weak", "label-weak"


def _escape_html(text: str) -> str:
    if text is None:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _safe_highlight_html(text: str, stems: set) -> str:
    highlighted = _highlight_terms(text or "", stems)
    highlighted = highlighted.replace("<mark>", "___MARK_OPEN___")
    highlighted = highlighted.replace("</mark>", "___MARK_CLOSE___")
    highlighted = _escape_html(highlighted)
    highlighted = highlighted.replace("___MARK_OPEN___", "<mark>")
    highlighted = highlighted.replace("___MARK_CLOSE___", "</mark>")
    return highlighted


def _render_static_table(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        st.caption("No table data available.")
        return

    header_html = "".join(
        f'<th style="text-align:left;padding:10px 12px;border-bottom:1px solid #313244;color:#89b4fa;">{_escape_html(col)}</th>'
        for col in df.columns
    )

    rows_html = []
    for _, row in df.iterrows():
        cells = "".join(
            f'<td style="padding:10px 12px;border-bottom:1px solid #1e1e2e;color:#cdd6f4;">{_escape_html(val)}</td>'
            for val in row.tolist()
        )
        rows_html.append(f"<tr>{cells}</tr>")

    table_html = (
        '<div style="overflow-x:auto;border:1px solid #313244;border-radius:10px;background:#181825;">'
        '<table style="width:100%;border-collapse:collapse;font-size:0.84rem;">'
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def _build_result_card_html(
    rank_pos: int,
    docno: str,
    src_label: str,
    badge_color: str,
    score: float,
    quality_text: str,
    qual_cls: str,
    score_pct: int,
    title_html: str,
    body_html: str,
    rel_html: str,
) -> str:
    return (
        '<div class="result-card">'
        '<div class="card-header">'
        f'<span class="rank-badge">#{rank_pos}</span>'
        '<div class="card-title-block">'
        f'{title_html}'
        f'{body_html}'
        f'<div class="card-docno">{_escape_html(docno)}</div>'
        '</div>'
        '<div class="card-side">'
        '<div class="badge-row">'
        f'<span class="source-badge" style="background:{badge_color};">{_escape_html(src_label)}</span>'
        f'{rel_html}'
        '</div>'
        '<div class="score-block">'
        f'<span class="score-value">{score:.4f}</span>'
        f'<span class="score-label {qual_cls}">{_escape_html(quality_text)}</span>'
        '</div>'
        '</div>'
        '</div>'
        '<div class="score-bar-track">'
        f'<div class="score-bar-fill" style="width:{score_pct}%;"></div>'
        '</div>'
        '</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
for _k, _v in {
    "example_query": "",
    "query_history": [],
    "active_topic_num": None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Search Engine")
    st.divider()

    if _index_exists():
        st.success("Index ready")
    else:
        st.warning("Index not built")

    if not config.USE_SAMPLE:
        with st.expander("Build Index", expanded=not _index_exists()):
            st.caption("Run once on the TREC dataset (10–30 min).")
            if st.button("Build Index", type="primary", use_container_width=True):
                st.info("Building index — streaming live output…")
                log_box = st.empty()
                log_text = ""
                proc = subprocess.Popen(
                    [sys.executable, "-u", "build_index.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=_BASE,
                )
                for line in iter(proc.stdout.readline, ""):
                    log_text += line
                    log_box.code(log_text[-3000:], language=None)
                proc.wait()
                if proc.returncode == 0:
                    st.success("Done. Reload the page.")
                    st.cache_resource.clear()
                else:
                    st.error("Build failed — see log above.")
    else:
        st.info("🗂️ Sample mode — pre-built index loaded. No corpus required.")

    st.divider()

    st.markdown("**Retrieval Variant**")
    variant_display = [VARIANT_LABELS[v["name"]] for v in VARIANTS]
    default_idx = variant_display.index(VARIANT_LABELS[DEFAULT_VARIANT["name"]])

    selected_display = st.selectbox(
        "Variant",
        options=variant_display,
        index=default_idx,
        label_visibility="collapsed",
        help="⭐ marks the best variant from ablation evaluation.",
    )
    selected_name = next(v["name"] for v in VARIANTS if VARIANT_LABELS[v["name"]] == selected_display)
    selected_variant = get_variant_by_name(selected_name)

    st.markdown(
        " ".join(
            f'<span class="feature-chip {"chip-on" if selected_variant[f] else "chip-off"}">'
            f'{"✓" if selected_variant[f] else "✗"} {lbl}</span>'
            for f, lbl in FEATURE_FLAGS
        ),
        unsafe_allow_html=True,
    )
    st.caption(_variant_desc(selected_name))

    st.divider()

    st.markdown("**Parameters**")
    top_k = st.slider("Results to show", min_value=5, max_value=100, value=10, step=5)

    show_expansion = st.toggle(
        "Show expansion details",
        value=False,
        disabled=not selected_variant["use_query_expansion"],
        help="Show which WordNet synonyms were added and their weights.",
    )
    show_debug = st.toggle(
        "Show query debug",
        value=False,
        help="Show normalised stems and timing.",
    )

    st.divider()

    st.markdown("**Score Guide**")
    st.markdown("""
<div class="score-info-box">
BM25F scores are <strong>not normalised</strong> — comparable only within one result set.<br><br>
<span class="score-label label-strong">Strong</span> Top 30% of results<br>
<span class="score-label label-good">Good</span> Mid-range<br>
<span class="score-label label-weak">Weak</span> Low query overlap<br><br>
Relative <em>rank</em> matters more than absolute score.
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.caption(
        f"Stemmer: Porter · stopwords removed  \n"
        f"Fields: title ×{int(config.W_TITLE)} + body ×{int(config.W_BODY)}  \n"
        f"k₁ = {config.K1}  ·  b_title = {config.B_TITLE}  ·  b_body = {config.B_BODY}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:8px;">
  <div class="hero-title">Field-Aware BM25F Search Engine</div>
  <div style="font-size:0.80rem; color:#6c7086; margin-top:2px;">
    Phrase/Proximity Modelling · Controlled Thesaurus-Based Query Expansion · TREC Robust04
  </div>
  <div class="hero-team">
    Blazej Olszta · Muhamad Husaam Ateeq · Max Monaghan · Sulaiman Bhatti
  </div>
</div>
""", unsafe_allow_html=True)

if config.USE_SAMPLE:
    st.markdown("""
<div style="background:#1e2a3a;border:1px solid #3a6ea8;border-radius:8px;
            padding:10px 16px;margin-bottom:14px;font-size:0.83rem;color:#a6c8e8;">
  🔬 <strong>Marker Sample Mode</strong> &nbsp;—&nbsp;
  Running on a pre-built index of a licensed subset of TREC Robust04.
  No corpus download or re-indexing required.
  Sample topics and qrels are available in the <em>Evaluation Results</em> tab.
</div>
""", unsafe_allow_html=True)

if not _index_exists():
    st.warning("Index not built yet. Open **Build Index** in the sidebar.")
    st.stop()

inverted_index, doc_map, doc_stats, collection_stats, snippets, docno_to_id = _load_index_cached()
qrels = _load_qrels()
topics = _load_topics()
eval_df = _load_eval_results()

N = collection_stats["N"]

tab_search, tab_eval = st.tabs(["Search", "Evaluation Results"])


# ─────────────────────────────────────────────────────────────────────────────
# Search tab
# ─────────────────────────────────────────────────────────────────────────────
with tab_search:
    with st.expander(
        f"📚  {N:,} documents across 5 collections — click for details",
        expanded=False,
    ):
        cols_ds = st.columns(len(DATASETS))
        for col, ds in zip(cols_ds, DATASETS):
            with col:
                st.markdown(
                    f'<div style="border:1px solid #313244;border-radius:8px;padding:10px;background:#181825;">'
                    f'<div style="font-size:0.72rem;font-weight:700;color:#89b4fa;">{ds["code"]}</div>'
                    f'<div style="font-size:0.82rem;font-weight:600;color:#cdd6f4;margin:3px 0;">{ds["name"]}</div>'
                    f'<div style="font-size:0.70rem;color:#6c7086;">{ds["years"]} · {ds["docs"]} docs</div>'
                    f'<div style="font-size:0.72rem;color:#a6adc8;margin-top:5px;">{ds["desc"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.caption(
            "TREC Disk 4 & 5 (Robust04) — used in TREC Robust Track 2004 to evaluate retrieval robustness across 249 topics (301–700)."
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    topic_options = (
        ["— free text query —"] + [
            f"Topic {t['num']}: {t['title'].strip()}"
            for t in sorted(topics.values(), key=lambda x: int(x["num"]))
        ]
        if topics else ["— free text query —"]
    )

    def _on_topic_change():
        val = st.session_state["_topic_select"]
        if val.startswith("Topic"):
            num = val.split(":")[0].replace("Topic", "").strip()
            t = topics.get(num, {})
            st.session_state["example_query"] = t.get("title", "").strip()
            st.session_state["active_topic_num"] = num
        else:
            st.session_state["active_topic_num"] = None

    st.selectbox(
        "TREC Topic",
        options=topic_options,
        index=0,
        key="_topic_select",
        on_change=_on_topic_change,
        help="Select an official TREC Robust04 topic to auto-fill the query. When a topic is selected, results show ground-truth relevance badges.",
    )

    active_topic_num = st.session_state.get("active_topic_num")
    if active_topic_num and active_topic_num in topics:
        desc = topics[active_topic_num].get("desc", "").strip()
        if desc:
            st.markdown(
                f'<div class="topic-desc-box"><strong>Topic {active_topic_num} description:</strong> {_escape_html(desc)}</div>',
                unsafe_allow_html=True,
            )

    col_input, col_btn = st.columns([7, 1])
    with col_input:
        default_text = st.session_state.get("example_query", "")
        query = st.text_input(
            "Query",
            value=default_text,
            placeholder="e.g.  Hubble Telescope Achievements  ·  industrial espionage",
            label_visibility="collapsed",
            key="query_input",
        )
    with col_btn:
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    st.markdown(
        "<div style='font-size:0.72rem;color:#6c7086;margin:5px 0 3px 0;'>Example TREC Robust04 topics:</div>",
        unsafe_allow_html=True,
    )

    for row_start in range(0, len(EXAMPLE_QUERIES), 5):
        row_qs = EXAMPLE_QUERIES[row_start:row_start + 5]
        chip_cols = st.columns(len(row_qs))
        for col, (q_text, q_num) in zip(chip_cols, row_qs):
            with col:
                if st.button(
                    q_text,
                    key=f"chip_{q_num}",
                    help=f"TREC topic {q_num}",
                    use_container_width=True,
                ):
                    st.session_state["example_query"] = q_text
                    st.session_state["active_topic_num"] = None
                    st.rerun()

    history = st.session_state.get("query_history", [])
    if history:
        st.markdown(
            "<div style='font-size:0.72rem;color:#6c7086;margin:6px 0 2px 0;'>Recent searches:</div>"
            + "".join(f'<span class="history-item">{_escape_html(h)}</span>' for h in history[-5:]),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    active_query = query or st.session_state.get("example_query", "")

    if not active_query:
        st.markdown("""
<div style="background:#1e1e2e;border:1px solid #313244;border-radius:12px;
            padding:28px 32px;margin-top:16px;text-align:center;">
  <div style="font-size:2rem;margin-bottom:8px;">🔍</div>
  <div style="font-size:1.1rem;font-weight:600;color:#cdd6f4;margin-bottom:6px;">
    Enter a query above to search 500,000+ TREC documents
  </div>
  <div style="font-size:0.82rem;color:#6c7086;max-width:500px;margin:0 auto;">
    Choose an example topic, pick a TREC topic from the dropdown, or type your own query.
    Use the <strong>Evaluation Results</strong> tab to see system performance metrics.
  </div>
</div>
""", unsafe_allow_html=True)
    else:
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

        history = st.session_state.get("query_history", [])
        if active_query not in history:
            history.append(active_query)
            st.session_state["query_history"] = history[-10:]

        try:
            import preprocess as _pp
            _stems = set(st_t for _, st_t, _ in _pp.normalise(active_query))
        except Exception:
            _stems = set()

        if show_debug:
            with st.expander("Query debug", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Results", len(ranked))
                c2.metric("Time (ms)", f"{elapsed_ms:.0f}")
                c3.metric("Variant", selected_name.split("_")[0])
                if _stems:
                    st.markdown("**Stems:** " + "  ".join(f"`{t}`" for t in sorted(_stems)))

        if show_expansion and selected_variant["use_query_expansion"]:
            try:
                import preprocess as _pp
                import query_expand as _qe

                _norm = _pp.normalise(active_query)
                _surf = [s for s, _, _ in _norm]
                _orig = list(dict.fromkeys(st_t for _, st_t, _ in _norm))
                _tw = _qe.expand_query(_orig, _surf, inverted_index, collection_stats)
                _exp = {t: w for t, w in _tw.items() if w < 1.0}

                with st.expander(f"WordNet expansion — {len(_exp)} synonym(s) added", expanded=True):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Original terms** `w = 1.0`")
                        for t in [t for t, w in _tw.items() if w >= 1.0]:
                            st.code(t, language=None)
                    with c2:
                        gamma = getattr(config, "EXPANSION_GAMMA", 0.3)
                        st.markdown(f"**Expanded synonyms** `w = {gamma}`")
                        for t, w in sorted(_exp.items(), key=lambda x: -x[1]):
                            st.code(f"{t}  ·  {w:.3f}", language=None)
            except Exception as e:
                st.warning(f"Expansion detail error: {e}")

        if not ranked:
            st.info("No results found. Try different terms or switch variant.")
        else:
            max_score = ranked[0][0]

            relevant_set: set = set()
            if active_topic_num:
                relevant_set = qrels.get(str(active_topic_num), set())

            st.markdown(
                f'<div class="results-summary">'
                f'<b>{len(ranked)}</b> results for <b>"{_escape_html(active_query)}"</b>'
                f'&nbsp;·&nbsp;{elapsed_ms:.0f} ms'
                f'&nbsp;·&nbsp;variant: <b>{_escape_html(selected_name.replace("_", " "))}</b>'
                + (
                    f'&nbsp;·&nbsp;<span style="color:#a6e3a1;">topic {active_topic_num} — {len(relevant_set)} relevant docs in qrels</span>'
                    if active_topic_num and relevant_set else ""
                )
                + '</div>',
                unsafe_allow_html=True,
            )

            for rank_pos, (score, docno) in enumerate(ranked, start=1):
                raw_snip = _get_snippet(docno, snippets, docno_to_id)
                pseudo_title = _extract_title(raw_snip)
                body_preview = _truncate(_remaining_snippet(raw_snip, pseudo_title))

                safe_title = _safe_highlight_html(pseudo_title, _stems)
                safe_body = _safe_highlight_html(body_preview, _stems)

                src_label, badge_color = _get_source(docno)
                score_pct = min(100, int((score / max_score) * 100)) if max_score > 0 else 0
                quality_text, qual_cls = _score_quality(score, max_score)

                rel_html = ""
                if active_topic_num:
                    rel_html = (
                        '<span class="rel-badge-yes">✓ Relevant</span>'
                        if docno in relevant_set
                        else '<span class="rel-badge-no">✗ Not judged relevant</span>'
                    )

                title_html = f'<div class="card-title">{safe_title}</div>' if pseudo_title else ""
                body_html = (
                    f'<div class="snippet">{safe_body}</div>'
                    if safe_body
                    else ('<div class="snippet"><em>No preview available.</em></div>' if not pseudo_title else "")
                )

                card_html = _build_result_card_html(
                    rank_pos=rank_pos,
                    docno=docno,
                    src_label=src_label,
                    badge_color=badge_color,
                    score=score,
                    quality_text=quality_text,
                    qual_cls=qual_cls,
                    score_pct=score_pct,
                    title_html=title_html,
                    body_html=body_html,
                    rel_html=rel_html,
                )
                st.markdown(card_html, unsafe_allow_html=True)

                art_key = f"article_{docno}"
                load_key = f"load_{rank_pos}_{docno}"
                dl_key = f"dl_{rank_pos}_{docno}"

                if config.USE_SAMPLE:
                    st.caption("Full-article loading is disabled in sample mode.")
                else:
                    btn_col, dl_col, spacer = st.columns([1.6, 1.6, 8])

                    with btn_col:
                        if st.button("📄 Load Full Article", key=load_key, use_container_width=True):
                            with st.spinner(f"Searching corpus for {docno}…"):
                                doc = _find_document(docno)
                            st.session_state[art_key] = doc

                    if art_key in st.session_state:
                        doc = st.session_state[art_key]
                        if doc:
                            body_text = doc.get("body", "") or ""
                            title_text = doc.get("title", "") or ""
                            dl_content = (
                                f"DOCNO: {docno}\n"
                                + (f"Title: {title_text}\n" if title_text else "")
                                + f"\n{body_text}"
                            )

                            with dl_col:
                                st.download_button(
                                    "⬇ Download .txt",
                                    data=dl_content.encode("utf-8"),
                                    file_name=f"{docno}.txt",
                                    mime="text/plain",
                                    key=dl_key,
                                    use_container_width=True,
                                )

                            article_title_html = (
                                f'<div class="article-title">{_escape_html(title_text)}</div>'
                                if title_text else
                                f'<div class="article-title" style="color:#585b70;">[No title — {_escape_html(src_label)}]</div>'
                            )

                            article_body = _escape_html(body_text[:8000])
                            if len(body_text) > 8000:
                                article_body += "\n\n[truncated — download for full text]"

                            st.markdown(
                                f'<div class="article-panel">'
                                f'{article_title_html}'
                                f'<div style="font-size:0.70rem;color:#585b70;margin-bottom:10px;">'
                                f'DOCNO: {_escape_html(docno)} &nbsp;·&nbsp; {_escape_html(src_label)}'
                                f'</div>'
                                f'<div class="article-body">{article_body}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.warning(
                                f"Article {docno} not found in corpus. Ensure the TREC data files are present at the paths in config.py."
                            )

            st.markdown("---")
            st.caption(
                f"Top {len(ranked)} of up to {top_k} candidates · "
                f"k₁={config.K1} · b_title={config.B_TITLE} · b_body={config.B_BODY} · "
                f"w_title={config.W_TITLE}"
                + (f" · phrase λ={config.PHRASE_BONUS} · prox window={config.PROXIMITY_WINDOW}"
                   if selected_variant["use_phrase_bonus"] else "")
                + (f" · expansion γ={getattr(config, 'EXPANSION_GAMMA', 0.3)}"
                   if selected_variant["use_query_expansion"] else "")
                + " · Snippet = first 200 chars of body · Title = first sentence of snippet"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation tab
# ─────────────────────────────────────────────────────────────────────────────
with tab_eval:
    st.markdown("### Ablation Evaluation — TREC Robust04")
    st.caption(
        "249 topics (301–549) evaluated using official qrels. Metrics: MAP (primary), P@10, nDCG@10, Recall@100, R-Precision."
    )

    if eval_df is None:
        st.warning(
            "evaluation_results.csv not found. Run evaluate.py against the TREC qrels to generate this file."
        )
    else:
        display_df = eval_df.copy()
        display_df["System"] = display_df["variant"].map(lambda n: _variant_label(n))
        metric_cols = ["MAP", "P@10", "nDCG@10", "Recall@100", "R-Precision"]
        available = [c for c in metric_cols if c in display_df.columns]

        for c in available:
            display_df[c] = display_df[c].round(4)

        best_idx = display_df["MAP"].idxmax()

        st.markdown("#### Key Metrics")
        metric_display = available[:4]
        col_metrics = st.columns(len(metric_display))
        for col, metric in zip(col_metrics, metric_display):
            best_val = display_df[metric].max()
            best_sys = display_df.loc[display_df[metric].idxmax(), "System"]
            worst_val = display_df[metric].min()
            delta_pct = f"+{(best_val - worst_val) / worst_val * 100:.1f}% vs baseline"
            with col:
                st.metric(
                    label=metric,
                    value=f"{best_val:.4f}",
                    delta=delta_pct,
                    help=f"Best system: {best_sys}",
                )

        st.markdown("#### Results Table")
        st.caption("⭐ = best-performing system per MAP")

        headers = ["System"] + available + ["Queries"]
        rows = []
        for _, row in display_df.iterrows():
            star = " ⭐" if row.name == best_idx else ""
            sys_name = row["System"] + star
            vals = [f"{row[c]:.4f}" for c in available]
            q = str(int(row["queries_evaluated"])) if "queries_evaluated" in row else "—"
            rows.append([sys_name] + vals + [q])

        table_df = pd.DataFrame(rows, columns=headers)
        _render_static_table(table_df)

        st.markdown("#### MAP by System")
        chart_df = display_df[["System", "MAP"]].set_index("System")
        st.bar_chart(chart_df, use_container_width=True)

        if "nDCG@10" in display_df.columns:
            st.markdown("#### nDCG@10 by System")
            ndcg_df = display_df[["System", "nDCG@10"]].set_index("System")
            st.bar_chart(ndcg_df, use_container_width=True)

        st.markdown("#### Interpretation")
        st.markdown("""
**Pattern of improvement across variants:**

- **BM25 Separate (unweighted)** underperforms even BM25 Flattened — scoring fields separately without weighting distorts IDF accumulation and hurts precision.
- **BM25F** (field-weighted) recovers and slightly improves over flattened BM25, confirming that title weighting (×5) gives more signal than flat field merging.
- **BM25F + Phrase & Proximity** shows the largest single gain — term-dependence modelling captures documents that truly discuss a topic together rather than mentioning terms incidentally.
- **BM25F + Phrase/Prox + WordNet** maintains nearly identical MAP to phrase/proximity alone, but improves R-Precision and Recall@100, showing that expansion helps recall without significantly hurting precision — consistent with controlled γ=0.3 weighting.
- **Neural reranking** (where evaluated) did not improve MAP, consistent with findings that cross-encoder reranking of 50 candidates is sensitive to first-stage candidate quality and may not help when lexical retrieval already ranks well.

**Are the scores good?**  
MAP ≈ 0.20 on Robust04 is a reasonable result for a classical lexical system without neural retrieval. State-of-the-art neural dense retrieval achieves MAP ≈ 0.35–0.45 on this collection, but requires GPU and large pre-trained models. The key contribution of this system is demonstrating that careful engineering of BM25F + term-dependence + controlled expansion yields meaningful, interpretable gains over a flat BM25 baseline.
""")

        pq_path = os.path.join(_BASE, "per_query_results.csv")
        if os.path.exists(pq_path):
            st.markdown("#### Per-Query Results (top 10 topics by AP, best variant)")
            try:
                # More tolerant CSV parsing for rows where query text contains commas
                pq_df = pd.read_csv(
                    pq_path,
                    skiprows=1,
                    engine="python",
                    on_bad_lines="skip",
                )

                # Clean whitespace from column names just in case
                pq_df.columns = [str(c).strip() for c in pq_df.columns]

                if "variant" in pq_df.columns and "AP" in pq_df.columns:
                    best_variant_name = display_df.loc[best_idx, "variant"]

                    # Make metrics numeric safely
                    for col in ["AP", "P@10", "nDCG@10"]:
                        if col in pq_df.columns:
                            pq_df[col] = pd.to_numeric(pq_df[col], errors="coerce")

                    pq_best = (
                        pq_df[pq_df["variant"] == best_variant_name]
                        .dropna(subset=["AP"])
                        .sort_values("AP", ascending=False)
                        .head(10)
                    )

                    show_cols = [
                        c for c in ["topic_id", "query", "AP", "P@10", "nDCG@10"]
                        if c in pq_best.columns
                    ]

                    if not pq_best.empty and show_cols:
                        _render_static_table(
                            pq_best[show_cols].reset_index(drop=True)
                        )
                    
                    else:
                        st.caption("Per-query results file loaded, but no matching rows were available for display.")
                else:
                    st.caption("Per-query results file loaded, but expected columns were not found.")
            except Exception as e:
                st.caption(f"Could not load per-query results: {e}")