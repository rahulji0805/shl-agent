"""
Recall@10 evaluator for the SHL assessment recommender.

For each sample_conversations/*.md trace:
  1. Parse the ground-truth turns: each turn's user message, and the
     markdown table of recommended assessment names (if present in that
     turn). The "expected final shortlist" is the most recent table seen
     by the end of the conversation (later tables supersede earlier ones,
     matching how refine is supposed to work).
  2. Replay ONLY the user messages against the live agent (POST /chat),
     letting the real agent generate its own replies turn by turn (its own
     prior replies are fed back as assistant history, exactly like the
     real evaluator would).
  3. Compare the agent's final-turn `recommendations` against the trace's
     expected final shortlist by name (case-insensitive).
  4. Recall@10 for a trace = |expected ∩ actual| / |expected|.
     Mean Recall@10 = average across all traces.

Usage:
    python recall_eval.py --dir sample_conversations\\GenAI_SampleConversations --url https://your-app.up.railway.app
"""
import argparse
import re
import time
from pathlib import Path

import requests

TURN_RE = re.compile(r"### Turn \d+\n(.*?)(?=### Turn \d+|\Z)", re.DOTALL)
USER_RE = re.compile(r"\*\*User\*\*\s*\n\s*>\s*(.*?)(?=\n\s*\*\*Agent\*\*)", re.DOTALL)
TABLE_ROW_RE = re.compile(r"^\|\s*\d+\s*\|\s*([^|]+?)\s*\|", re.MULTILINE)


def parse_trace(path: Path):
    text = path.read_text(encoding="utf-8")
    turns = TURN_RE.findall(text)
    user_messages = []
    expected_final = []
    for turn_text in turns:
        m = USER_RE.search(turn_text)
        if m:
            # collapse multi-line blockquote continuations into one line
            msg = " ".join(
                line.strip().lstrip(">").strip()
                for line in m.group(1).strip().splitlines()
            ).strip()
            user_messages.append(msg)
        rows = TABLE_ROW_RE.findall(turn_text)
        if rows:
            expected_final = [r.strip() for r in rows]  # later tables supersede
    return user_messages, expected_final


def replay(base_url: str, user_messages: list[str], delay: float):
    history = []
    final_recs = []
    for i, um in enumerate(user_messages):
        history.append({"role": "user", "content": um})
        try:
            resp = requests.post(f"{base_url}/chat", json={"messages": history}, timeout=35)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    ERROR on turn {i+1}: {e}")
            continue
        reply_text = data.get("reply", "")
        history.append({"role": "assistant", "content": reply_text})
        recs = data.get("recommendations", []) or []
        if recs:
            final_recs = [r["name"] for r in recs]
        if delay:
            time.sleep(delay)
    return final_recs


def recall_at_10(expected: list[str], actual: list[str]) -> float:
    if not expected:
        return None
    exp_norm = {e.lower().strip() for e in expected}
    act_norm = {a.lower().strip() for a in actual[:10]}
    hit = len(exp_norm & act_norm)
    return hit / len(exp_norm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between calls (avoid rate limits)")
    args = ap.parse_args()

    base_url = args.url.rstrip("/")
    trace_dir = Path(args.dir)
    files = sorted(trace_dir.glob("*.md"))

    results = []
    for f in files:
        print(f"\n=== {f.name} ===")
        user_messages, expected_final = parse_trace(f)
        if not expected_final:
            print("  (no expected shortlist found in trace, skipping)")
            continue
        print(f"  expected ({len(expected_final)}): {expected_final}")
        actual_final = replay(base_url, user_messages, args.delay)
        print(f"  actual   ({len(actual_final)}): {actual_final}")
        r = recall_at_10(expected_final, actual_final)
        print(f"  Recall@10: {r:.2f}" if r is not None else "  Recall@10: n/a")
        if r is not None:
            results.append((f.name, r))

    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    for name, r in results:
        print(f"  {name}: {r:.2f}")
    if results:
        mean_r = sum(r for _, r in results) / len(results)
        print(f"\nMean Recall@10 across {len(results)} traces: {mean_r:.3f}")


if __name__ == "__main__":
    main()
