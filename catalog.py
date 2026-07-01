"""
Loads the scraped SHL product catalog and normalizes it into a flat,
code-consumable structure with a single `corpus_text` field used for
retrieval.
"""
import json
import re
from pathlib import Path
from typing import Any

CATALOG_PATH = Path(__file__).parent / "catalog.json"

# Map SHL "keys" (category tags) to the single-letter test_type codes used
# in the API response schema (matches the examples in the assignment doc).
KEY_TO_TYPE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


def _test_type(keys: list[str]) -> str:
    codes = []
    for k in keys:
        c = KEY_TO_TYPE.get(k)
        if c and c not in codes:
            codes.append(c)
    return ",".join(codes) if codes else "K"


def load_catalog() -> list[dict[str, Any]]:
    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    items = []
    for r in raw:
        name = r.get("name", "").strip()
        if not name:
            continue
        keys = r.get("keys", []) or []
        job_levels = r.get("job_levels", []) or []
        languages = r.get("languages", []) or []
        description = (r.get("description") or "").strip()
        duration = (r.get("duration") or "").strip()

        corpus = " ".join(
            [
                name,
                description,
                " ".join(keys),
                " ".join(job_levels),
                duration,
            ]
        )
        corpus = re.sub(r"\s+", " ", corpus).strip()

        items.append(
            {
                "entity_id": r.get("entity_id", ""),
                "name": name,
                "url": r.get("link", ""),
                "description": description,
                "job_levels": job_levels,
                "languages": languages,
                "duration": duration,
                "remote": r.get("remote", ""),
                "adaptive": r.get("adaptive", ""),
                "keys": keys,
                "test_type": _test_type(keys),
                "corpus_text": corpus,
            }
        )
    return items


if __name__ == "__main__":
    cat = load_catalog()
    print(f"Loaded {len(cat)} catalog items")
    print(cat[0])
