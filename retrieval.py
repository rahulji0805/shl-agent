"""
Lightweight retrieval over the SHL catalog using BM25 (no embedding model
download required — keeps this deployable on any free-tier host with zero
external model dependencies).
"""
import re
from rank_bm25 import BM25Okapi

from catalog import load_catalog

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class CatalogIndex:
    def __init__(self):
        self.items = load_catalog()
        self._corpus_tokens = [tokenize(it["corpus_text"]) for it in self.items]
        self.bm25 = BM25Okapi(self._corpus_tokens)
        self.by_name = {it["name"].lower(): it for it in self.items}

    def search(self, query: str, k: int = 10) -> list[dict]:
        if not query.strip():
            return []
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            range(len(self.items)), key=lambda i: scores[i], reverse=True
        )
        results = []
        for i in ranked[: k * 3]:  # overfetch, filter zero scores below
            if scores[i] <= 0:
                continue
            results.append(self.items[i])
            if len(results) >= k:
                break
        return results

    def find_by_name(self, name: str) -> dict | None:
        return self.by_name.get(name.strip().lower())

    def all_items(self) -> list[dict]:
        """Full catalog list — used by agent.py to reconstruct a prior
        shortlist from plain-text assistant replies (state tracking across
        the stateless API's turns)."""
        return self.items


_INDEX: CatalogIndex | None = None


def get_index() -> CatalogIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = CatalogIndex()
    return _INDEX


if __name__ == "__main__":
    idx = get_index()
    for q in ["Java developer stakeholders", "rust engineer", "safety dependability industrial"]:
        print("Q:", q)
        for r in idx.search(q, k=5):
            print("  -", r["name"], "|", r["test_type"])
        print()
