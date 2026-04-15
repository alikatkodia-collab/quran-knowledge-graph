# Setup Guide — Quran Knowledge Graph

You've been given a `.dump` file containing the full Neo4j database and the project source code. Follow these steps to get it running on your machine.

---

## Step 1: Install Prerequisites

### Python 3.10+
Download from https://www.python.org/downloads/
During installation, check "Add Python to PATH".

### Neo4j Desktop
Download from https://neo4j.com/download/ (free, requires registration).

### Ollama (optional, for the free version)
Download from https://ollama.com/download
Then pull a model:
```bash
ollama pull qwen2.5:14b-instruct-q6_K
```

---

## Step 2: Set Up the Neo4j Database

1. Open **Neo4j Desktop**
2. Click **New** to create a new project
3. Inside the project, click **Add** (top right) -> **File** -> select the `.dump` file you were given
4. Click the **...** menu on the dump file -> **Create new DBMS from dump**
5. Set a password (remember it — you'll need it in Step 4)
6. Wait for the import to complete
7. **Start** the database

The database name needs to be `quran`. If Neo4j created it with a different name:
1. Click the database name to open **Neo4j Browser**
2. Run: `CREATE DATABASE quran IF NOT EXISTS`
3. Or just rename it in the DBMS settings

To verify it worked, open Neo4j Browser and run:
```
MATCH (v:Verse) RETURN count(v)
```
You should see **6234**.

---

## Step 3: Install Python Dependencies

Open a terminal in the project folder and run:

```bash
pip install anthropic neo4j fastapi uvicorn python-dotenv sentence-transformers scikit-learn nltk pyyaml numpy requests
```

---

## Step 4: Create Your .env File

In the project root folder, create a file called `.env` with this content:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=the_password_you_set_in_step_2
ANTHROPIC_API_KEY=sk-ant-api03-your_key_here
```

**For the Anthropic API key:**
- Go to https://console.anthropic.com/
- Create an account and add credits ($5 minimum)
- Go to API Keys -> Create Key
- Paste it in the `.env` file

**If you only want the free version** (no API key needed), you can skip the `ANTHROPIC_API_KEY` line entirely and just use `app_free.py` with Ollama.

---

## Step 5: Run It

Pick one:

```bash
# Default version (Claude Sonnet, ~$0.05-0.10 per question)
python app.py

# Cheapest API version (Claude Haiku, ~$0.01-0.03 per question)
python app_lite.py

# Full hallucination reduction (Claude Sonnet, ~$0.15-0.35 per question)
python app_full.py

# Free version (local Ollama, $0)
python app_free.py
```

Your browser will open automatically. If it doesn't, go to:
- `app.py` -> http://localhost:8081
- `app_full.py` -> http://localhost:8083
- `app_lite.py` -> http://localhost:8084
- `app_free.py` -> http://localhost:8085

---

## Step 6: Test It

Try asking:
- "What does the Quran say about patience?"
- "Show me the connection between mercy and forgiveness"
- "What is verse 2:255?"
- "What Arabic root connects the words for book and prescribed?"

You should see the AI calling tools (shown as expandable blocks), citing specific verses with `[surah:verse]` references you can hover over, and the 3D graph lighting up with connections.

---

## Troubleshooting

### "Neo4j unavailable" on startup
- Make sure Neo4j Desktop is open and the database is **Started** (green play button)
- Check that your password in `.env` matches what you set in Step 2

### "Credit balance too low" or API errors
- Go to https://console.anthropic.com/ and add credits
- Or use `python app_free.py` which doesn't need an API key

### "Cannot connect to Ollama" (free version only)
- Make sure Ollama is running: `ollama serve`
- Make sure you've pulled a model: `ollama pull qwen2.5:14b-instruct-q6_K`

### 3D graph shows but chat doesn't work
- Check the browser console (F12 -> Console) for errors
- Make sure both Neo4j and the Python server are running

### Port already in use
- Another program is using that port. Either close it or run with a different port:
  ```bash
  python app_free.py --port 9000
  ```

---

## What Each Version Costs

| Version | Command | Model | Cost per Question |
|---------|---------|-------|-------------------|
| Default | `python app.py` | Claude Sonnet | ~$0.05-0.10 |
| Full | `python app_full.py` | Claude Sonnet | ~$0.15-0.35 |
| Lite | `python app_lite.py` | Claude Haiku | ~$0.01-0.03 |
| Free | `python app_free.py` | Local (Ollama) | $0.00 |

The system has a built-in answer cache — repeated or similar questions cost less because previous answers are reused. The cache is shared across all versions.
