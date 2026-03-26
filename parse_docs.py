"""
parse_docs.py — Regex-based SGML document parser for all 5 TREC collections.

Each collection has its own tag conventions; we handle them with targeted
regexes rather than a strict XML parser (the SGML is not valid XML).

Public API
----------
iter_collection(root_dir, collection_type) -> Iterator[dict]
    Yields dicts with keys: 'docno', 'title', 'body'

iter_all_collections(collections) -> Iterator[dict]
    Convenience wrapper over config.COLLECTIONS.
"""

import os
import re
from typing import Iterator

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _tag(name: str) -> re.Pattern:
    """Compile a pattern that extracts the content of a single SGML tag."""
    return re.compile(
        rf"<{name}[^>]*>(.*?)</{name}>",
        re.DOTALL | re.IGNORECASE,
    )

def _extract(pattern: re.Pattern, text: str, default: str = "") -> str:
    m = pattern.search(text)
    return m.group(1).strip() if m else default

def _strip_inner_tags(text: str) -> str:
    """Remove inner XML/SGML tags, keeping their text content."""
    return re.sub(r"<[^>]+>", " ", text)

def _strip_pjg(text: str) -> str:
    """Remove PJG processing instructions used in FR94."""
    # <!-- PJG ... --> style comments
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    return text

# Compiled patterns for each field
_P_DOCNO    = _tag("DOCNO")
_P_TEXT     = _tag("TEXT")
_P_HEADLINE = _tag("HEADLINE")
_P_TI       = _tag("TI")
_P_H3       = _tag("H3")
_P_TTL      = _tag("TTL")       # Congressional Record title (inside TEXT)
_P_SO       = _tag("SO")        # SO section at end of CR TEXT (speaker info)

# DOC splitter — one file may hold many <DOC>…</DOC> records
_P_DOC = re.compile(r"<DOC>(.*?)</DOC>", re.DOTALL | re.IGNORECASE)


def _split_docs(file_text: str) -> list[str]:
    return _P_DOC.findall(file_text)


# ---------------------------------------------------------------------------
# Per-collection parsers
# ---------------------------------------------------------------------------

def _parse_ft(raw: str) -> dict | None:
    """Financial Times: HEADLINE → title, TEXT → body."""
    docno = _extract(_P_DOCNO, raw).strip()
    if not docno:
        return None
    title = _strip_inner_tags(_extract(_P_HEADLINE, raw))
    body  = _strip_inner_tags(_extract(_P_TEXT, raw))
    return {"docno": docno, "title": title, "body": body}


def _parse_fbis(raw: str) -> dict | None:
    """FBIS: TI inside H3 → title, TEXT → body."""
    docno = _extract(_P_DOCNO, raw).strip()
    if not docno:
        return None
    # Title may be inside <H3><TI>…</TI></H3> or standalone <TI>…</TI>
    h3_block = _extract(_P_H3, raw)
    if h3_block:
        title = _strip_inner_tags(_extract(_P_TI, h3_block) or h3_block)
    else:
        title = _strip_inner_tags(_extract(_P_TI, raw))
    body = _strip_inner_tags(_extract(_P_TEXT, raw))
    return {"docno": docno, "title": title, "body": body}


def _parse_fr94(raw: str) -> dict | None:
    """
    Federal Register 1994: heavily markup-laden TEXT, no dedicated title tag.
    We strip PJG instructions and use an empty title (FR94 docs are body-only
    for ranking purposes).
    """
    docno = _extract(_P_DOCNO, raw).strip()
    if not docno:
        return None
    text_raw = _extract(_P_TEXT, raw)
    body = _strip_inner_tags(_strip_pjg(text_raw))
    return {"docno": docno, "title": "", "body": body}


def _parse_cr(raw: str) -> dict | None:
    """
    Congressional Record: <TTL>…</TTL> sits *inside* the <TEXT> block.
    Title = TTL content; body = TEXT minus TTL and SO sections.
    """
    docno = _extract(_P_DOCNO, raw).strip()
    if not docno:
        return None
    text_raw = _extract(_P_TEXT, raw)
    # Extract title from TTL (may be several; use first)
    title = _strip_inner_tags(_extract(_P_TTL, text_raw))
    # Remove TTL and SO blocks from body
    body_raw = re.sub(r"<TTL>.*?</TTL>", " ", text_raw, flags=re.DOTALL | re.IGNORECASE)
    body_raw = re.sub(r"<SO>.*?</SO>",   " ", body_raw,  flags=re.DOTALL | re.IGNORECASE)
    body = _strip_inner_tags(body_raw)
    return {"docno": docno, "title": title, "body": body}


def _parse_latimes(raw: str) -> dict | None:
    """LA Times: HEADLINE → title (contains <P> tags), TEXT → body."""
    docno = _extract(_P_DOCNO, raw).strip()
    if not docno:
        return None
    title = _strip_inner_tags(_extract(_P_HEADLINE, raw))
    body  = _strip_inner_tags(_extract(_P_TEXT, raw))
    return {"docno": docno, "title": title, "body": body}


# Map collection type tag → parser function
_PARSERS = {
    "FT":      _parse_ft,
    "FBIS":    _parse_fbis,
    "FR94":    _parse_fr94,
    "CR":      _parse_cr,
    "LATIMES": _parse_latimes,
}

# ---------------------------------------------------------------------------
# File / directory walkers
# ---------------------------------------------------------------------------

_SKIP_NAMES = {".DS_Store", "_.DS_Store", "MD5SUM", "READMEFT", "READMEFR",
               "READCHG", "READFRCG", "CREDTD", "CRHDTD", "FR94DTD",
               "FTDTD", "FBIS.DTD", "LA.DTD"}
_SKIP_EXTS  = {".sgml", ".dtd", ".xml"}


def _should_skip(name: str) -> bool:
    if name.startswith("._"):
        return True
    if name in _SKIP_NAMES:
        return True
    ext = os.path.splitext(name)[1].lower()
    if ext in _SKIP_EXTS:
        return True
    return False


def _read_file(path: str) -> str:
    """Read a TREC data file with Latin-1 fallback (some files are not UTF-8)."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=enc, errors="replace") as fh:
                return fh.read()
        except Exception:
            continue
    return ""


def iter_collection(root_dir: str, collection_type: str) -> Iterator[dict]:
    """
    Walk *root_dir* recursively, parse every document file, and yield
    dicts with keys 'docno', 'title', 'body'.
    """
    parser = _PARSERS.get(collection_type)
    if parser is None:
        raise ValueError(f"Unknown collection type: {collection_type!r}")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip DTD subdirectories
        dirnames[:] = [d for d in dirnames if d.upper() != "DTDS"]
        for fname in filenames:
            if _should_skip(fname):
                continue
            fpath = os.path.join(dirpath, fname)
            text  = _read_file(fpath)
            if not text:
                continue
            for raw in _split_docs(text):
                doc = parser(raw)
                if doc and doc["docno"]:
                    yield doc


def iter_all_collections(collections: list[tuple[str, str]]) -> Iterator[dict]:
    """Iterate over all (root_dir, collection_type) pairs from config.COLLECTIONS."""
    for root_dir, ctype in collections:
        if not os.path.isdir(root_dir):
            print(f"[WARN] Collection directory not found: {root_dir}")
            continue
        yield from iter_collection(root_dir, ctype)
