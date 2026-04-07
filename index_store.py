"""
index_store.py — Dict-like wrapper around the SQLite inverted index.

Allows rank.py and query_expand.py to use .get(), `in`, and [] on the
on-disk index without loading the whole thing into RAM.
"""

import pickle
import sqlite3


class SQLiteIndex:
    """Read-only dict-like interface to the SQLite inverted index."""

    def __init__(self, path: str):
        self._conn = sqlite3.connect(path, check_same_thread=False)

    def get(self, term: str, default=None):
        row = self._conn.execute(
            "SELECT df, postings FROM idx WHERE term=?", (term,)
        ).fetchone()
        if row is None:
            return default
        return (row[0], pickle.loads(row[1]))

    def __contains__(self, term: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM idx WHERE term=?", (term,)
        ).fetchone() is not None

    def __getitem__(self, term: str):
        result = self.get(term)
        if result is None:
            raise KeyError(term)
        return result

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM idx").fetchone()[0]

    def close(self):
        self._conn.close()
