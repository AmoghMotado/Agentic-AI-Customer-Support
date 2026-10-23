# How to run this project on Windows

Five steps, about 10 minutes (most of it downloading packages).

## Upgrading from the single-business version

If the NovaCart-only version already runs on your machine, you don't need to set anything up again. No new packages are needed (SQLite comes with Python).

1. Unzip the new version into a **new** folder, for example `C:\Projects\agentic-ai-customer-support-v2`.
2. Copy your `.env` from the old folder into the new one. Optionally add the two new lines from `.env.example` (`DEFAULT_BUSINESS_ID`, `ADMIN_PASSWORD`); both have sensible defaults.
3. Reuse your existing virtual environment, or create a new one with steps 2 and 3 below. In Git Bash:

```bash
cd /c/Projects/agentic-ai-customer-support-v2
source /c/Projects/agentic-ai-customer-support/.venv/Scripts/activate   # your old .venv
python -m pytest -v                 # expect: 42 passed, 5 deselected
python -m streamlit run app.py
```

The first start creates `data/app.db` and loads NovaCart and GreenLeaf Grocery into it. Each business's search index is built the first time you chat with it.

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

**Expected:** `42 passed, 5 deselected`.

Then build each business's search index and check retrieval (first run downloads the ~90 MB embedding model):

```powershell
python -m src.rag
```

**Expected:** something like

```
NovaCart (novacart): 8 documents, 29 chunks, huggingface:sentence-transformers/all-MiniLM-L6-v2
Test query: What is your refund policy?
  refund_policy.txt #1 (distance 0.61): NovaCart Refund Policy Overview NovaCart offers refunds...

GreenLeaf Grocery (greenleaf): 3 documents, ...
Test query: What is your refund policy?
  refund_and_returns_policy.txt #1 (distance ...): GreenLeaf Grocery Refund and Returns Policy...
```

Then start the app:

```powershell
python -m streamlit run app.py
```

Your browser opens http://localhost:8501. Pick a business at the top of the sidebar (or open http://localhost:8501/?business=greenleaf), turn on **Debug mode** and try the example query buttons. Open **Admin** in the sidebar to add your own business. Press **Ctrl+C** in the terminal to stop.

## What you should see

| You type | What happens |
|---|---|
| What is your refund policy? | Debug shows Action: Knowledge base (RAG) and the source file |
| Where is order ORD1001? | Debug shows Tool `get_order_status`, input `ORD1001`, result Shipped |
| Where is my order? | The assistant asks for your order ID and calls no tool |
| My package arrived damaged. Create a support ticket. | A ticket ID like TCK1001, visible on the Admin page under Tickets |
| Hello | A short friendly reply |
| Switch to GreenLeaf, ask "What is your refund policy?" | GreenLeaf's 48-hour perishables policy, source `refund_and_returns_policy.txt` |

## Adding your own business

1. Open **Admin** in the sidebar and use **Add a business** (ID like `city-bakery`, name, one-line description).
2. In **Knowledge base**, upload `.txt` or `.md` files (policies, FAQs) and click **Save uploaded documents**.
3. Optionally, in **Customers & orders**, import a `customers.json` and then an `orders.json` (click **File format** for an example).
4. Open http://localhost:8501/?business=city-bakery and chat.

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
| Answers look wrong after changing documents | On the Admin page, click **Rebuild search index now** (or delete the `vector_store` folder) |
| I want the original demo data back | Stop the app, delete `data/app.db` and `vector_store`, start again |

## Using OpenAI embeddings instead of the local model

This avoids the 2 GB PyTorch download and costs a fraction of a cent for this knowledge base.

1. In `.env`, set `EMBEDDING_PROVIDER=openai`
2. In `requirements.txt`, delete the `langchain-huggingface` and `sentence-transformers` lines
3. Delete the `vector_store` folder, then run `python -m src.rag` (it rebuilds every business)

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
