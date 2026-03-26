# TREC Search Engine

A full-text search engine built on the TREC Disk 4 & 5 corpus.

**Ranking:** BM25F (field-aware) + phrase bonus + proximity bonus + WordNet query expansion.

---

## Requirements

- Python 3.10 or later
- The TREC Disk 4 & 5 dataset folder placed inside this directory (see layout below)

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLTK data

```bash
python setup_nltk.py
```

### 3. Check dataset layout

The dataset folder must be structured like this inside the `CODE/` directory:

```
CODE/
├── drive-download-20260326T121407Z-1-001/
│   ├── TREC-Disk-4/
│   │   └── TREC-Disk-4/
│   │       ├── FT/
│   │       ├── FR94/
│   │       └── CR_103RD/
│   └── TREC-Disk-5/
│       └── TREC-Disk-5/
│           ├── FBIS/
│           └── LATIMES/
```

If your dataset folder has a different name, update `DATASET_DIR` in `config.py`.

### 4. Build the index (run once)

```bash
python build_index.py
```

This processes all documents and writes the index to `index_data/`.
It takes **10–30 minutes** on the full corpus and uses ~4 GB of RAM.
You only need to do this once — the index is saved to disk.

---

## Searching

### Interactive mode

```bash
python search.py
```

Type a query at the `Query>` prompt and press Enter.

```
Query> child support enforcement
Query> international trade policy
Query> jet aircraft
```

**Runtime commands:**

| Command | Effect |
|---|---|
| `:top=20` | Show top 20 results (default: 10) |
| `:expand=off` | Disable WordNet query expansion |
| `:expand=on` | Re-enable query expansion |
| `:debug=on` | Show expanded terms and their weights |
| `:debug=off` | Hide debug output |
| `:quit` | Exit |

### Single query (command line)

```bash
python search.py "child support enforcement"
python search.py --top-k 20 "international trade"
python search.py --no-expand "information retrieval"
python search.py --debug "jet aircraft flight"
```

---

## How it works

### Offline (build_index.py)

1. **Parse** — extracts `docno`, `title`, and `body` from SGML files across all 5 collections (FT, FR94, CR, FBIS, LA Times)
2. **Normalise** — lowercase → tokenise → remove stopwords → Porter stemming
3. **Index** — builds a positional, field-aware inverted index storing term frequency and word positions per field
4. **Statistics** — stores per-document field lengths and collection averages for BM25F

### Online (search.py)

1. **Parse query** — same normalisation pipeline as documents
2. **Expand** — adds WordNet synonyms with drift controls (nouns only, WSD-lite sense selection, synonyms only, DF filter, co-occurrence filter, capped at 3 per term)
3. **Score** — BM25F across title + body fields, plus phrase and proximity bonuses
4. **Rank** — return top-k documents sorted by score

### Scoring formula

```
Score = BM25F(original terms, weight=1.0)
      + 0.3 × BM25F(expanded terms)
      + phrase_bonus    (exact adjacent phrase match)
      + proximity_bonus (terms within 8-word window)
```

---

## Files

| File | Purpose |
|---|---|
| `config.py` | All paths and tunable hyperparameters |
| `preprocess.py` | Text normalisation pipeline |
| `parse_docs.py` | SGML document parser for all 5 collections |
| `build_index.py` | Offline index builder |
| `query_expand.py` | WordNet-based query expansion |
| `rank.py` | BM25F + phrase + proximity scoring |
| `search.py` | Online query processor and CLI |
| `setup_nltk.py` | Downloads required NLTK data |
