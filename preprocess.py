"""
preprocess.py — Text normalisation pipeline (Stage A1 / B1).

The SAME pipeline is applied to both documents and queries so that
stemmed query terms match stemmed index terms correctly.

Pipeline:
  1. Lowercase
  2. Strip HTML/SGML markup & PJG processing instructions
  3. Tokenise (unicode-aware word tokeniser)
  4. Remove punctuation-only tokens
  5. (Optional) Remove stopwords
  6. (Optional) Porter stemming

Returns: list of (surface_token, normalised_token, position)
so callers can track both the stemmed form used in the index and the
original position (needed for phrase/proximity matching).
"""

import re
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import config

# ---------------------------------------------------------------------------
# One-time NLTK resource downloads (safe to re-call; no-ops if present)
# ---------------------------------------------------------------------------
for _resource in ("punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger",
                  "averaged_perceptron_tagger_eng"):
    try:
        nltk.data.find(f"tokenizers/{_resource}")
    except LookupError:
        try:
            nltk.download(_resource, quiet=True)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Module-level singletons (initialised once)
# ---------------------------------------------------------------------------
_stemmer   = PorterStemmer()
_stopwords = set(stopwords.words("english"))

# Regex to strip XML/SGML tags and PJG instructions
_RE_TAG   = re.compile(r"<!--.*?-->|<[^>]+>", re.DOTALL)
_RE_SPACE = re.compile(r"\s+")

# Only keep tokens made entirely of letters (possibly hyphened compound words)
_RE_ALPHA = re.compile(r"^[a-z]+(?:-[a-z]+)*$")


def _strip_markup(text: str) -> str:
    """Remove all XML/SGML tags and HTML comments from *text*."""
    text = _RE_TAG.sub(" ", text)
    # Decode common SGML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&blank;", " ").replace("&nbsp;", " ")
    return _RE_SPACE.sub(" ", text).strip()


def normalise(text: str) -> list[tuple[str, str, int]]:
    """
    Normalise *text* and return a list of (surface, stemmed, position) triples.

    *surface*  — token as it appears after lowercasing (used for display)
    *stemmed*  — term stored / looked up in the index
    *position* — 0-based word offset in the token stream (used for phrase/proximity)
    """
    text = _strip_markup(text)
    text = text.lower()

    tokens = word_tokenize(text)
    result: list[tuple[str, str, int]] = []
    pos = 0                      # monotonic word counter

    for tok in tokens:
        # Keep only purely alphabetic (possibly hyphenated) tokens
        if not _RE_ALPHA.match(tok):
            pos += 1             # consume position even for skipped tokens
            continue

        if config.DO_REMOVE_STOPWORDS and tok in _stopwords:
            pos += 1
            continue

        stemmed = _stemmer.stem(tok) if config.DO_STEM else tok
        result.append((tok, stemmed, pos))
        pos += 1

    return result


def terms(text: str) -> list[str]:
    """Convenience wrapper: return just the stemmed terms (no positions)."""
    return [stemmed for _, stemmed, _ in normalise(text)]


def terms_with_positions(text: str) -> list[tuple[str, int]]:
    """Return (stemmed_term, position) pairs."""
    return [(stemmed, p) for _, stemmed, p in normalise(text)]
