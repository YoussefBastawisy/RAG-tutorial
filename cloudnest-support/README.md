# CloudNest Support — Workshop App

An end-to-end customer-support agent: RAG over a markdown knowledge base, a
LangGraph ReAct agent with two tools, and a Streamlit chat UI. Deployable
to Streamlit Community Cloud in under 5 minutes.

## Architecture

```
User → Streamlit chat UI
            │
            ▼
   LangGraph ReAct Agent  (Llama-3.1-8b on Groq)
        │           │
        ▼           ▼
  search_kb    create_ticket
        │
        ▼
  Chroma + MiniLM embeddings
        │
        ▼
  data/kb/*.md
```

## Local setup

```bash
git clone <this-repo>
cd cloudnest-support

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Get a free Groq key at https://console.groq.com
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and paste your key

streamlit run app.py
```

First launch downloads the embedding model (~90 MB) and may take ~30 seconds.

## Try these questions

- "What's the difference between Pro and Business plans?"
- "I forgot my password, how do I reset it?"
- "Can I get a refund on my annual plan after 6 months?"
- "My files aren't syncing — what should I do?"
- "I want to talk to a human about a billing problem." (triggers the ticket tool)

## Deploy to Streamlit Cloud

1. Push this repo to GitHub (public).
2. Go to https://share.streamlit.io and click "New app".
3. Pick the repo, set `app.py` as the main file.
4. Under "Advanced settings → Secrets", paste:
   ```
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click Deploy. First build takes ~3-5 min.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit UI, session state, tool-call rendering |
| `agent.py` | LangGraph agent + tool definitions |
| `rag.py` | KB loading, chunking, embeddings, vectorstore |
| `data/kb/*.md` | The fictional CloudNest knowledge base |

## Things to extend

- Swap `MemorySaver` for a SQLite checkpointer to persist conversations across restarts.
- Add a reranker (e.g. Cohere or `ms-marco-MiniLM`) between retrieval and LLM.
- Add hybrid search (BM25 + dense) via `EnsembleRetriever`.
- Add evaluation with RAGAS — generate a small Q/A set and measure faithfulness.
- Replace the mocked `create_support_ticket` with a real Zendesk/Intercom call.
