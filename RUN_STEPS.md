# How to run this project on Windows

Five steps, about 10 minutes (most of it downloading packages).

## 1. Unzip and open in VS Code

Unzip to a short path **without spaces and outside OneDrive**, for example:

```
C:\Projects\agentic-ai-customer-support
```

Open that folder in VS Code (File > Open Folder), then open a terminal inside it with **Ctrl+`**. Make sure the terminal says **PowerShell**.

## 2. Create the virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

The prompt should now start with `(.venv)`.

If you get *"running scripts is disabled on this system"*, run this once, answer `Y`, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## 3. Install the packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This downloads about 2 GB, mostly PyTorch for the local embedding model. If you would rather skip that, see "Using OpenAI embeddings instead" at the bottom.

## 4. Add your OpenAI API key

```powershell
Copy-Item .env.example .env
notepad .env
```

Replace `sk-your-key-here` with your real key from https://platform.openai.com/api-keys, save, and close Notepad. `.env` is already in `.gitignore`, so it will never be committed.

## 5. Run

First, the free tests (no API calls):

```powershell
python -m pytest -v
```

**Expected:** `23 passed, 5 deselected`.

Then build the search index and check retrieval (first run downloads the ~90 MB embedding model):

```powershell
python -m src.rag
```

**Expected:** something like

```
Vector store built: 8 documents, 29 chunks
Embedding model: huggingface:sentence-transformers/all-MiniLM-L6-v2

Test query: What is your refund policy?
  refund_policy.txt #1 (distance 0.61): NovaCart Refund Policy Overview NovaCart offers refunds...
```

Then start the app:

```powershell
streamlit run app.py
```

Your browser opens http://localhost:8501. Turn on **Debug mode** in the sidebar and try the example query buttons. Press **Ctrl+C** in the terminal to stop.

## What you should see

| You type | What happens |
|---|---|
| What is your refund policy? | Debug shows Action: Knowledge base (RAG) and the source file |
| Where is order ORD1001? | Debug shows Tool `get_order_status`, input `ORD1001`, result Shipped |
| Where is my order? | The assistant asks for your order ID and calls no tool |
| My package arrived damaged. Create a support ticket. | A ticket ID like TCK1001, saved in `data/tickets.json` |
| Hello | A short friendly reply |

`docs/TESTING.md` has the full checklist, including the failure cases to try.

## Common problems

| Problem | Fix |
|---|---|
| `python` opens the Microsoft Store | Settings > Apps > Advanced app settings > App execution aliases, turn off python.exe and python3.exe |
| `Activate.ps1 cannot be loaded` | Run the `Set-ExecutionPolicy` command in step 2 |
| `OPENAI_API_KEY is not set` banner | `.env` is missing, or the key line still says `sk-your-key-here`. Restart the app after editing `.env`. |
| `The OpenAI API key is invalid` | Wrong or incomplete key. Create a new one. |
| `rate limit or quota was reached` | Add credit at platform.openai.com > Billing |
| `The model '...' is not available` | Put a model your account can use in `OPENAI_MODEL` in `.env` |
| Install fails on `torch` | Use the OpenAI embeddings option below |
| Answers look wrong after editing a policy file | Delete the `vector_store` folder and run `python -m src.rag` again |

## Using OpenAI embeddings instead of the local model

This avoids the 2 GB PyTorch download and costs a fraction of a cent for this knowledge base.

1. In `.env`, set `EMBEDDING_PROVIDER=openai`
2. In `requirements.txt`, delete the `langchain-huggingface` and `sentence-transformers` lines
3. Delete the `vector_store` folder, then run `python -m src.rag`

The app rebuilds the index automatically when the embedding model changes.

## Putting it on GitHub

```powershell
git init
git add .
git commit -m "Agentic AI customer support assistant"
git branch -M main
git remote add origin https://github.com/<your-username>/agentic-ai-customer-support.git
git push -u origin main
```

Before pushing, run `git status` and confirm **`.env` is not listed**. Add screenshots of the chat and the debug panel to a `screenshots/` folder and link them in the README.
