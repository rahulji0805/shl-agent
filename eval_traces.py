"""
Replays each sample conversation trace's USER turns against your locally
running /chat endpoint, so you can eyeball whether the agent's behavior
(clarify / recommend / refine / compare / refuse) looks sane before
deploying.

This does NOT compute Recall@10 (that needs the harness's hidden labels) —
it's just a fast visual sanity check.

Usage:
    export GROQ_API_KEY=...
    uvicorn main:app --port 8000 &
    python3 eval_traces.py --dir sample_conversations/GenAI_SampleConversations --url http://localhost:8000
"""
import argparse
import json
import re
from pathlib import Path

import requests


def extract_user_turns(md_text: str) -> list[str]:
    """Pull just the user message text out of each '**User**\n\n> ...' block."""
    pattern = re.compile(r"\*\*User\*\*\s*\n+>\s*(.+?)(?=\n\n\*\*Agent\*\*)", re.DOTALL)
    turns = []
    for m in pattern.finditer(md_text):
        text = m.group(1).strip()
        # strip leading '> ' continuation markers on wrapped quote lines
        text = "\n".join(line.lstrip("> ").rstrip() for line in text.splitlines())
        turns.append(text.strip())
    return turns


def run_trace(path: Path, base_url: str):
    print(f"\n{'=' * 70}\n{path.name}\n{'=' * 70}")
    user_turns = extract_user_turns(path.read_text(encoding="utf-8"))
    if not user_turns:
        print("  (no user turns found — check parsing)")
        return

    history = []
    for i, user_text in enumerate(user_turns, 1):
        history.append({"role": "user", "content": user_text})
        print(f"\n--- Turn {i} ---")
        print(f"USER: {user_text[:200]}")
        try:
            resp = requests.post(f"{base_url}/chat", json={"messages": history}, timeout=35)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  ERROR calling /chat: {e}")
            break

        print(f"AGENT: {data.get('reply', '')[:300]}")
        recs = data.get("recommendations", [])
        print(f"  recommendations: {len(recs)} item(s)" + (f" -> {[r['name'] for r in recs]}" if recs else ""))
        print(f"  end_of_conversation: {data.get('end_of_conversation')}")

        history.append({"role": "assistant", "content": data.get("reply", "")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="sample_conversations/GenAI_SampleConversations")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--only", default=None, help="run only this filename, e.g. C1.md")
    args = ap.parse_args()

    trace_dir = Path(args.dir)
    files = sorted(trace_dir.glob("*.md"))
    if args.only:
        files = [f for f in files if f.name == args.only]

    for f in files:
        run_trace(f, args.url)


if __name__ == "__main__":
    main()
