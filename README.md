# TREC Search Engine

A full-text search engine built on the TREC Disk 4 & 5 corpus.

**Ranking:** BM25F (field-aware) + phrase bonus + proximity bonus + WordNet query expansion.

---

## Prerequisites

- Python 3.10 or later — download from [python.org](https://www.python.org/downloads/)
- The TREC Disk 4 & 5 dataset (not included in this repo — see step 3 below)

---

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/Maxs-ToolBox/Search-Engine-BM25-expanded.git
cd Search-Engine-BM25-expanded
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the TREC dataset

Extract the TREC Disk 4 & 5 dataset anywhere inside the repo folder. The folder name does not matter — the code will find it automatically as long as the internal structure looks like this:

```
Search-Engine-BM25-expanded/
└── <any folder name>/
    ├── TREC-Disk-4/
    │   └── TREC-Disk-4/
    │       ├── FT/
    │       ├── FR94/
    │       └── CR_103RD/
    └── TREC-Disk-5/
        └── TREC-Disk-5/
            ├── FBIS/
            └── LATIMES/
```

### 4. Download NLTK data

```bash
python setup_nltk.py
```

### 5. Launch the app

```bash
streamlit run app.py
```

This opens the search engine in your browser. On first use, click **Build Index** in the sidebar — this processes all documents and saves the index to disk. It takes **10–30 minutes** and needs around **4 GB of RAM**. You only need to do this once.

---

## Using the GUI

Once the index is built, type a query in the search box and press Enter.

Use the sidebar to control:

| Setting | Description |
|---|---|
| **Number of results** | How many results to show (5–100) |
| **WordNet query expansion** | Automatically adds synonyms to improve recall |
| **Show expansion details** | Shows which terms were added and their weights |

---

## Command line (alternative to the GUI)

**Interactive mode:**
```bash
python search.py
```

**Single query:**
```bash
python search.py "child support enforcement"
python search.py --top-k 20 "international trade"
python search.py --no-expand "information retrieval"
python search.py --debug "jet aircraft flight"
```

---

## How it works

### Offline — building the index

1. **Parse** — extracts `docno`, `title`, and `body` from SGML files across all 5 collections (FT, FR94, CR, FBIS, LA Times)
2. **Normalise** — lowercase → tokenise → remove stopwords → Porter stemming
3. **Index** — builds a positional, field-aware inverted index storing term frequency and word positions per field
4. **Statistics** — stores per-document field lengths and collection averages needed for BM25F

### Online — answering a query

1. **Normalise** — same pipeline applied to the query as to documents
2. **Expand** — WordNet synonyms added with drift controls (nouns only, WSD-lite sense selection, synonyms only, frequency filter, co-occurrence filter, capped at 3 per term)
3. **Score** — BM25F across title and body fields, plus phrase and proximity bonuses
4. **Rank** — return top-k documents sorted by score

### Scoring formula

```
Score = BM25F(original terms, weight=1.0)
      + 0.3 x BM25F(expanded terms)
      + phrase_bonus    (exact adjacent phrase match)
      + proximity_bonus (terms within 8-word window)
```

---

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit GUI |
| `config.py` | All paths and tunable hyperparameters |
| `preprocess.py` | Text normalisation pipeline |
| `parse_docs.py` | SGML document parser for all 5 collections |
| `build_index.py` | Offline index builder |
| `query_expand.py` | WordNet-based query expansion |
| `rank.py` | BM25F + phrase + proximity scoring |
| `search.py` | Command line search interface |
| `setup_nltk.py` | Downloads required NLTK data |
