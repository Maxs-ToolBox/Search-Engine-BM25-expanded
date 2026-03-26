"""
preprocess.py — Text normalisation pipeline (Stage A1 / B1).

The SAME pipeline is applied to both documents and queries so that
stemmed query terms match stemmed index terms correctly.

Pipeline:
  1. Lowercase
  2. Strip HTML/SGML markup & PJG processing instructions
  3. Tokenise with fast regex (replaces slow NLTK word_tokenize)
  4. Remove stopwords
  5. Porter stemming (via NLTK PorterStemmer)

Returns: list of (surface_token, normalised_token, position)
so callers can track both the stemmed form used in the index and the
original position (needed for phrase/proximity matching).
"""

import re
from functools import lru_cache
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import config

# ---------------------------------------------------------------------------
# One-time NLTK resource downloads (only needed for stopwords/wordnet/tagger)
# word_tokenize is no longer used so punkt is not required
# ---------------------------------------------------------------------------
for _resource in ("stopwords", "wordnet", "averaged_perceptron_tagger_eng"):
    try:
        nltk.data.find(f"corpora/{_resource}")
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

# Cache stemming results — the vocabulary is ~300K unique words but we process
# 100M+ tokens, so each word only gets stemmed once instead of thousands of times.
@lru_cache(maxsize=None)
def _stem(word: str) -> str:
    return _stemmer.stem(word)

# Regex to strip XML/SGML tags and HTML comments
_RE_TAG    = re.compile(r"<!--.*?-->|<[^>]+>", re.DOTALL)
_RE_SPACE  = re.compile(r"\s+")

# Fast tokeniser: extract runs of ASCII letters only.
# Handles hyphenated words by splitting on hyphens (e.g. "well-known" -> ["well","known"])
# This replaces NLTK word_tokenize and is ~8x faster.
_RE_WORDS  = re.compile(r"[a-z]+")


def _strip_markup(text: str) -> str:
    """Remove all XML/SGML tags and HTML comments from *text*."""
    text = _RE_TAG.sub(" ", text)
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
    text = _strip_markup(text).lower()

    result: list[tuple[str, str, int]] = []
    pos = 0

    for tok in _RE_WORDS.findall(text):
        if len(tok) < 2:          # skip single-letter tokens
            pos += 1
            continue

        if config.DO_REMOVE_STOPWORDS and tok in _stopwords:
            pos += 1
            continue

        stemmed = _stem(tok) if config.DO_STEM else tok
        result.append((tok, stemmed, pos))
        pos += 1

    return result


def terms(text: str) -> list[str]:
    """Convenience wrapper: return just the stemmed terms (no positions)."""
    return [stemmed for _, stemmed, _ in normalise(text)]


def terms_with_positions(text: str) -> list[tuple[str, int]]:
    """Return (stemmed_term, position) pairs."""
    return [(stemmed, p) for _, stemmed, p in normalise(text)]
