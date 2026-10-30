# ai-agent-web-support

An AI support agent that answers questions about a website using **RAG (Retrieval-Augmented Generation)**.

It crawls a website, stores the page content as vector embeddings in **ChromaDB**, and then uses **Google Gemini** to answer user questions based only on the most relevant pieces of that content.

By default it crawls `https://www.gbu.ac.in` and asks: *"what course are available in university"*.

---

## How it works

```
                ┌──────────────── Ingestion (runs once, when DB is empty) ────────────────┐
 Website  ──►   Scrape page  ──►  Split text into chunks  ──►  Embed (Gemini)  ──►  Store in ChromaDB
                    │                                                                   │
                    └── follow internal links (up to MAX_PAGES) ◄───────────────────────┘

                ┌──────────────────────────── Chat ───────────────────────────────────────┐
 Question ──►   Embed question  ──►  Find top-K similar chunks in ChromaDB  ──►  Gemini chat model  ──►  Answer
```

1. **Scrape** – `scrapeWebpage()` downloads a page with `axios`, parses it with `cheerio`, and extracts:
   - the *head* (page title + meta description),
   - the *body* text (scripts/styles removed, whitespace collapsed),
   - internal links (same site) and external links.
2. **Chunk** – `chunkText()` splits the body text into fixed-size chunks (`CHUNK_SIZE` characters).
3. **Embed** – `generateVectorEmbeddings()` converts each chunk into a vector with the `gemini-embedding-001` model.
4. **Store** – `insertIntoDB()` upserts each vector into the ChromaDB collection, along with the original text and URL as metadata.
5. **Crawl** – `ingest()` repeats steps 1–4 for every internal link, depth-first, until `MAX_PAGES` pages have been visited.
6. **Chat** – `chat()` embeds the question, retrieves the `TOP_K_RESULTS` closest chunks from ChromaDB, and sends them as context to the Gemini chat model, which writes the answer.

Ingestion is skipped on later runs if the collection already has data, so only the chat step runs.

---

## Tech stack

| Package | Purpose |
|---|---|
| [`axios`](https://www.npmjs.com/package/axios) | Fetch web pages |
| [`cheerio`](https://www.npmjs.com/package/cheerio) | Parse HTML and extract text/links |
| [`openai`](https://www.npmjs.com/package/openai) | SDK used to call Gemini through its **OpenAI-compatible** endpoint |
| [`chromadb`](https://www.npmjs.com/package/chromadb) | Client for the ChromaDB vector database |
| [`dotenv`](https://www.npmjs.com/package/dotenv) | Load the API key from `.env` |
| Docker | Runs the ChromaDB server locally |

---

## Project structure

```
ai-agent-web-support/
├── index.js            # All application code (scrape → embed → store → chat)
├── docker-compose.yml  # Starts ChromaDB on localhost:8000
├── package.json        # Dependencies (ES modules: "type": "module")
├── .env                # Your API key (not to be committed)
└── README.md
```

`index.js` is organised in numbered sections, top to bottom in the order the program uses them:

1. Imports
2. Configuration
3. Clients (Gemini via OpenAI SDK, ChromaDB)
4. Scraping
5. Text processing & embeddings
6. Vector database (ChromaDB)
7. Ingestion
8. Chat
9. Entry point (`main`)

---

## Prerequisites

- **Node.js 18+**
- **Docker** (for ChromaDB)
- A **Google Gemini API key** – get one from [Google AI Studio](https://aistudio.google.com/apikey)

---

## Setup & run

1. **Install dependencies**

   ```bash
   npm install
   ```

2. **Add your API key** – create a `.env` file in the project root:

   ```env
   OPENAI_API_KEY=your-gemini-api-key
   ```

   > The variable is named `OPENAI_API_KEY` because the OpenAI SDK is used, but the value must be a **Gemini** key — requests are sent to `https://generativelanguage.googleapis.com/v1beta/openai/`.

3. **Start ChromaDB**

   ```bash
   docker compose up -d
   ```

   ChromaDB will be available at `http://localhost:8000`.

4. **Run the agent**

   ```bash
   node index.js
   ```

### Example output

```
🫀 Chroma heartbeat: 1730000000000
📦 Collection "WEB_SCAPED_DATA_COLLECTION-1" has 0 records
🚀 Ingesting https://www.gbu.ac.in (max 10 pages)...
🌐 Scraping: https://www.gbu.ac.in
   ✔ head: 85 chars, body: 6120 chars, internal links: 142, external links: 18
   🧩 4 chunks to embed
   💾 Stored https://www.gbu.ac.in
...
✅ Ingestion done, 10 pages visited

❓ Question: what course are avillable in university
🔎 Retrieved 5 chunks from Chroma
🔗 Source URLs: https://www.gbu.ac.in/...
💬 Asking gemini-3.5-flash...
🤖 -> ...
```

---

## Configuration

All settings are constants at the top of `index.js` (section **2. Configuration**):

| Constant | Default | Description |
|---|---|---|
| `TARGET_URL` | `https://www.gbu.ac.in` | Website to crawl. Only links starting with this URL are followed. |
| `WEB_COLLECTION` | `WEB_SCAPED_DATA_COLLECTION-1` | ChromaDB collection name. |
| `CHAT_MODEL` | `gemini-3.5-flash` | Model that writes the answer. |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Model that creates vector embeddings. |
| `MAX_PAGES` | `10` | Maximum pages to crawl. |
| `CHUNK_SIZE` | `2000` | Characters per stored text chunk. |
| `TOP_K_RESULTS` | `5` | Number of chunks retrieved for each question. |

**To ask a different question**, edit the `chat('...')` call at the bottom of `main()`.

**To crawl a different website**, change `TARGET_URL` *and* `WEB_COLLECTION` (a new collection name forces a fresh crawl — otherwise the old site's data is reused).

---

## Re-crawling / resetting data

Ingestion only runs when the collection is empty. To re-crawl:

- change `WEB_COLLECTION` to a new name, **or**
- wipe ChromaDB's data by recreating the container:

  ```bash
  docker compose down && docker compose up -d
  ```

  (The compose file has no volume, so data is lost when the container is removed.)

---

## Troubleshooting

| Problem | Likely cause |
|---|---|
| `❌ Failed: ... fetch failed` / heartbeat error | ChromaDB isn't running — run `docker compose up -d`. |
| `401` / `403` errors | Missing or invalid Gemini key in `.env`. |
| `429` errors | Gemini rate limit hit — wait, or lower `MAX_PAGES`. |
| `✘ Error scraping ...` | That page timed out or blocked the request; it is skipped and crawling continues. |
| Answers are off-topic | The relevant pages weren't crawled — raise `MAX_PAGES` and re-crawl. |
