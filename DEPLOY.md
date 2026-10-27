# Deploy to Streamlit Community Cloud

You do the clicking. It takes about 10 minutes and is free. The app is **public**: anyone
with the link can use it, so the spending cap and the Anthropic Console limit are what
protect your credit.

## 0. Before you start

- Revoke any API key that was ever pasted into a chat, and create a fresh one in the
  Anthropic Console (Settings → API Keys).
- In the Anthropic Console, set a **monthly spend limit** (e.g. $5). The app's daily cap
  resets whenever the app restarts, so this limit is the real backstop.

## 1. Create the app

1. Go to share.streamlit.io, sign in with GitHub, and click **Create app** → **Deploy a public app from GitHub**.
2. Repository `yashasvikesarwaniiitd/rag_agent`, branch `main`, main file `app.py`.
3. **Advanced settings:**
   - Python version **3.12**.
   - **Secrets**: paste the block below, with your new key in place of the placeholder:
     ```toml
     ANTHROPIC_API_KEY = "paste-your-new-key-here"
     RAG_LLM = "anthropic"
     RAG_ALLOW_UPLOADS = "off"
     RAG_DAILY_BUDGET_USD = "0.50"
     ```
     Keep the keys at the top level, with no `[section]` header. Streamlit exposes top-level
     secrets as environment variables, and that's how the app reads them.
4. Click **Deploy**. The first build installs CPU-only PyTorch and takes a few minutes. The
   first page load then downloads two small models (~220 MB) and builds the index from the
   committed vector cache, which takes about a minute more.

## 2. Smoke test (about ₹0.15)

1. The sidebar lists **5 documents and 3,823 chunks**, and there is **no upload box**.
2. Ask: *How often must states revalidate Medicaid provider enrollment?* You should get a
   written answer citing MACPAC, and the sidebar shows today's spend.
3. Ask: *What is the capital of France?* It should be declined with no model call.
4. Ask: *Should our plan approve Dr. Patel's credentialing application?* It should be
   declined.

## If something goes wrong

| Symptom | Fix |
|---|---|
| Only the "fake" model is offered | The `ANTHROPIC_API_KEY` secret is missing or misnamed. Fix it in Settings → Secrets; the app restarts on save |
| "Something went wrong (AuthenticationError)" | The key is wrong or revoked. Paste a new one into Secrets |
| Costs look high | Lower `RAG_DAILY_BUDGET_USD`, or set `RAG_LLM = "fake"` to stop all paid calls |
| Build fails on torch / nvidia | `uv.lock` must contain no `nvidia` packages. See `[tool.uv.sources]` in pyproject.toml |

## What is and isn't in the repo

- **In:** the 5 public-domain documents (CMS, eCFR, MACPAC) and `data/cache/embeddings.sqlite`
  (vectors only, so cold starts take ~1 min instead of ~7).
- **Out:** `.env`, the Chroma index (rebuilt at startup), and any other file in `data/raw`
  (allow-listed in `.gitignore`).
