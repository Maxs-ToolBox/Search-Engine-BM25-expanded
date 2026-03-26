"""
setup_nltk.py — Download all NLTK data required by the search engine.
Run this once after installing dependencies:
    python setup_nltk.py
"""

import nltk

RESOURCES = [
    ("corpora", "stopwords"),
    ("corpora", "wordnet"),
    ("taggers", "averaged_perceptron_tagger_eng"),
]

for category, name in RESOURCES:
    print(f"Downloading {name} ... ", end="", flush=True)
    nltk.download(name, quiet=True)
    print("done")

print("\nAll NLTK resources ready.")
