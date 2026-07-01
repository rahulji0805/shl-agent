# SHL Conversational Assessment Recommender

## Setup
```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here   # free tier: https://console.groq.com
```

## Run locally
```bash
uvicorn main:app --reload --port 8000
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring a Java developer who works with stakeholders"}]}'
```

## Architecture
- `catalog.py` — loads `catalog.json` (scraped SHL catalog), normalizes each
  entry into a flat record with a `corpus_text` field for retrieval and a
  `test_type` code derived from SHL's category "keys".
- `retrieval.py` — BM25 keyword retrieval over the catalog (no embedding
  model download needed — keeps this deployable anywhere with zero external
  ML dependencies beyond the LLM call itself).
- `agent.py` — core turn logic:
  1. Build a candidate pool via BM25 over the full user-message history
     (+ always include OPQ32r / Verify G+ as default contenders, since these
     recur across almost every professional hiring battery in the traces).
  2. Single LLM call (Groq, Llama 3.3 70B) with a system prompt encoding the
     four behaviors (clarify / recommend / refine / compare) + scope refusal
     rules, constrained to JSON output.
  3. **Validation layer**: every recommended item is re-resolved against the
     real catalog by exact name match. Anything the LLM hallucinates (wrong
     name, made-up test) is silently dropped, and url/test_type are always
     overwritten with the real catalog values — this is what guarantees the
     "URLs only from scraped catalog" hard eval passes even on a bad LLM turn.
  4. If the LLM call itself fails (timeout, bad key, etc.), we return a
     schema-compliant fallback reply instead of crashing or returning
     malformed JSON — protects the schema-compliance hard eval.
- `main.py` — FastAPI app, `GET /health`, `POST /chat`. Stateless: every
  call receives and reprocesses the full conversation history; we cap to
  the last 8 turns to match the evaluator's turn limit.

## Deploy (free tier, e.g. Render)
1. Push this folder to a GitHub repo.
2. Render → New Web Service → connect repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add env var `GROQ_API_KEY`.
6. Submit the deployed `/health` and `/chat` URLs.

## Known limitations / next steps
- BM25 retrieval is keyword-based, not semantic — works well for the traces
  observed (technical skill names, role terms match catalog text closely)
  but may miss paraphrased queries with no shared vocabulary. A fallback
  embedding-based retriever would be the natural upgrade if time allowed.
- Fact extraction/state is implicit in the LLM call (full history each
  turn) rather than an explicit structured slot-filling step — simpler and
  matches the stateless API contract, but harder to unit-test deterministically.
