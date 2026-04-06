# Frontend UI (Next.js)

Minimal UI for the finance Agentic/RAG workflow:

- Left sidebar: history only
- Main area: question, conclusion, key textual evidence, key financial facts, filings
- Language toggle: `EN` / `中文` / `AUTO`
- Hidden advanced params (`top_k`, document count) to keep UI concise; default `top_k` follows `NEXT_PUBLIC_ASK_DEFAULT_TOP_K` (fallback 3), mirroring agent `ASK_DEFAULT_TOP_K`.

## Run

```bash
cd src/frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Backend URL

Frontend now calls a same-origin Next.js API route (`/api/ask/generate`) to avoid browser CORS.
Set backend upstream URL in `.env.local` if needed:

```bash
BACKEND_API_BASE_URL=http://127.0.0.1:8000
```
