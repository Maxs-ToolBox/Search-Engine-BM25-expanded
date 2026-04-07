"""
app.py — Streamlit GUI for the TREC BM25F Search Engine.

Run with:
    streamlit run app.py
"""

import os
import sys
import pickle
import subprocess
import time

import streamlit as st

import config
from index_store import SQLiteIndex

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TREC Search Engine",
    page_icon="magnifying-glass",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Index loading (cached so it only loads once per session)
# ---------------------------------------------------------------------------

def _index_exists() -> bool:
    return all(os.path.exists(p) for p in [
        config.INDEX_FILE, config.DOC_MAP_FILE,
        config.DOC_STATS_FILE, config.COLL_STATS_FILE,
    ])


@st.cache_resource(show_spinner="Loading index...")
def load_index():
    """Open the shelve index and load auxiliary pickle files (cached across reruns)."""
    def _load(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)

    inverted_index   = SQLiteIndex(config.INDEX_FILE)
    doc_map          = _load(config.DOC_MAP_FILE)
    doc_stats        = _load(config.DOC_STATS_FILE)
    collection_stats = _load(config.COLL_STATS_FILE)

    snippets = None
    if os.path.exists(config.SNIPPETS_FILE):
        snippets = _load(config.SNIPPETS_FILE)

    return inverted_index, doc_map, doc_stats, collection_stats, snippets


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Search Engine")
    st.caption("BM25F + Phrase/Proximity + WordNet Expansion")
    st.divider()

    # --- Index status ---
    if _index_exists():
        st.success("Index ready")
    else:
        st.warning("Index not built yet")

    # --- Build Index ---
    st.subheader("Build Index")
    st.caption("Run once on the TREC dataset. Takes 10-30 minutes.")

    if st.button("Build Index", type="primary", use_container_width=True):
        st.info("Building index — this window will stream live progress.")
        log_box = st.empty()
        log_text = ""

        process = subprocess.Popen(
            [sys.executable, "-u", "build_index.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

        for line in iter(process.stdout.readline, ""):
            log_text += line
            log_box.code(log_text, language=None)

        process.wait()

        if process.returncode == 0:
            st.success("Index built successfully! Reload the page to search.")
            st.cache_resource.clear()
        else:
            st.error("Build failed — check the log above for errors.")

    st.divider()

    # --- Search parameters ---
    st.subheader("Search Parameters")

    top_k = st.slider("Number of results", min_value=5, max_value=100,
                       value=10, step=5)

    expand = st.toggle("WordNet query expansion", value=True,
                        help="Adds synonyms to the query to improve recall.")

    show_debug = st.toggle("Show expansion details", value=False,
                            help="Displays expanded terms and their weights.")

    st.divider()
    st.caption(
        "**Scoring:**\n"
        "BM25F (title x5 + body) + phrase bonus + proximity bonus"
        + (" + WordNet expansion (γ=0.3)" if expand else "")
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("TREC Search Engine")

if not _index_exists():
    st.warning(
        "The index has not been built yet. "
        "Click **Build Index** in the sidebar to get started."
    )
    st.stop()

# Load index
inverted_index, doc_map, doc_stats, collection_stats, snippets = load_index()

N = collection_stats["N"]
st.caption(f"Searching {N:,} documents across FT, FR94, CR, FBIS and LA Times")

# Search box
query = st.text_input(
    label="Query",
    placeholder="e.g. child support enforcement",
    label_visibility="collapsed",
)

search_clicked = st.button("Search", type="primary")

# ---------------------------------------------------------------------------
# Run search
# ---------------------------------------------------------------------------
if query and (search_clicked or query):
    # Import here so we don't slow down the initial page load
    import preprocess
    import query_expand
    import rank as ranking

    t0 = time.time()

    # B1 — normalise
    norm = preprocess.normalise(query)
    if not norm:
        st.warning("Query produced no searchable terms after normalisation.")
        st.stop()

    surface_tokens      = [s for s, _, _ in norm]
    query_terms         = list(dict.fromkeys(stemmed for _, stemmed, _ in norm))

    # B2 — expand
    if expand:
        term_weights = query_expand.expand_query(
            query_terms, surface_tokens, inverted_index, collection_stats
        )
    else:
        term_weights = {t: 1.0 for t in query_terms}

    # B3 — rank
    ranked = ranking.rank_documents(
        term_weights, inverted_index, doc_stats, collection_stats, top_k=top_k
    )

    elapsed_ms = (time.time() - t0) * 1000

    # ---------------------------------------------------------------------------
    # Display expansion details
    # ---------------------------------------------------------------------------
    if show_debug and expand:
        with st.expander("Expanded query terms", expanded=True):
            orig  = {t: w for t, w in term_weights.items() if w >= 1.0}
            expnd = {t: w for t, w in term_weights.items() if w < 1.0}
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original terms** (weight 1.0)")
                for t in orig:
                    st.code(t)
            with col2:
                st.markdown(f"**Expanded terms** (weight {config.EXPANSION_GAMMA})")
                for t, w in expnd.items():
                    st.code(f"{t}  ({w:.3f})")

    # ---------------------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------------------
    if not ranked:
        st.info("No results found.")
    else:
        st.markdown(
            f"**{len(ranked)} results** &nbsp;·&nbsp; {elapsed_ms:.0f} ms",
            unsafe_allow_html=True,
        )
        st.divider()

        for rank, (score, doc_id) in enumerate(ranked, start=1):
            docno   = doc_map[doc_id]
            snippet = (snippets[doc_id] if snippets else "") if doc_id < len(snippets or []) else ""

            with st.container():
                col_rank, col_main, col_score = st.columns([1, 10, 2])

                with col_rank:
                    st.markdown(f"### {rank}")

                with col_main:
                    st.markdown(f"**{docno}**")
                    if snippet:
                        st.caption(snippet + ("..." if len(snippet) == config.SNIPPET_LENGTH else ""))

                with col_score:
                    st.metric(label="Score", value=f"{score:.4f}")

            st.divider()
