# AI-Powered Intelligent News Analytics & Agentic Search Platform

An intelligent news analytics system that ingests live RSS feeds, enriches
every article with NLP (entities, sentiment, keyphrases, embeddings), and
proactively surfaces emerging topics and shifting narratives — without
requiring the user to search for anything. Includes hybrid (keyword +
semantic) search with cross-encoder reranking, and an LLM-powered analyst
that can generate timelines and explain why a topic is trending.

> **Status:** This is an MVP built during an internship project. It runs
> fully locally on free-tier services. It is not production-hardened —
> see [Known Issues & Housekeeping](#known-issues--housekeeping) below for
> exactly what that means in practice.

---

## Features

- **Multi-source RSS ingestion** with URL-based deduplication and full-text
  extraction (not just RSS summaries).
- **NLP enrichment pipeline** — named entity recognition, sentiment
  classification, keyphrase extraction, and dense semantic embeddings,
  applied to every article automatically.
- **Hybrid search** — BM25 keyword search fused with dense-vector k-NN
  semantic search via reciprocal rank fusion, with optional cross-encoder
  reranking.
- **Proactive Intelligence Engine (PIE)** — unsupervised topic-emergence
  detection (density-based clustering, not a fixed topic list) and
  entity-level sentiment/narrative shift monitoring, synthesized into a
  natural-language feed by an LLM that is explicitly instructed to verify
  its own output is grounded in the retrieved articles before generating
  a summary.
- **Agentic analyst** — ask for a timeline of an entity/topic, or ask why
  something is trending, and a LangGraph-orchestrated workflow retrieves
  supporting data and synthesizes an answer.
- **React + TypeScript frontend** with four views: Intelligence Feed,
  Search, Trending, and Analyst.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Backend framework | FastAPI (Python 3.11) | Async-first, matches an I/O-bound workload |
| Frontend | React 18 + TypeScript + Vite | Fast dev server, typed components |
| Styling | Tailwind CSS | Utility-first, no separate CSS files |
| Relational store | PostgreSQL via Supabase (REST client) | Canonical article records |
| Search / analytics | Elasticsearch 8.12 | Native BM25 + dense_vector kNN in one engine |
| Message broker | Redis Streams | Lightweight, consumer-group semantics for the ingestion queue |
| NLP | spaCy (`en_core_web_sm`), DistilBERT (sentiment), sentence-transformers (`all-MiniLM-L6-v2`) | Small/fast models chosen to run on CPU-only hardware |
| Clustering | scikit-learn DBSCAN | Density-based — doesn't force a fixed number of topics |
| LLM inference | Groq API (`openai/gpt-oss-120b`) | Low-latency hosted inference |
| Agent orchestration | LangGraph | Bounded-turn graph workflows for Timeline/Trend Explanation |
| Content extraction | trafilatura | Strips boilerplate from arbitrary article HTML |
| Containerization | Docker (Redis + Elasticsearch only) | Consistent local dev services |

---

## Requirements

### Must already exist on your machine
- **Python 3.11+**
- **Node.js 18+** and npm
- **Docker Desktop** (running Redis and Elasticsearch as containers)
- **git**

### External accounts (free tier is enough)
- A [Supabase](https://supabase.com) account (PostgreSQL database)
- A [Groq](https://console.groq.com) account (LLM API key)

### Platform notes
- `torch==2.4.1` is a heavy download (several hundred MB). First `pip
  install` will take a while — this is expected, not a hang.
- All Python dependencies in `requirements.txt` install cleanly on
  macOS (including Apple Silicon), Linux, and Windows without needing
  platform-specific substitutions.
- No GPU is required or used. Model choices were deliberately picked to
  run acceptably on CPU-only hardware.

---

## Installation

### macOS

```bash
# Prerequisites (if not already installed)
brew install python@3.11 node git
# Install Docker Desktop from https://www.docker.com/products/docker-desktop/

git clone <your-repo-url>
cd <repo-folder>

# Backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cd ..

# Frontend
npm install
```

### Ubuntu / Linux

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip nodejs npm git
# Install Docker: https://docs.docker.com/engine/install/ubuntu/

git clone <your-repo-url>
cd <repo-folder>

cd backend
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cd ..

npm install
```

### Windows

Windows works, but running the backend inside **WSL2** (Windows Subsystem
for Linux) is noticeably smoother than native Windows for this stack —
Docker Desktop integrates with WSL2 directly. If you'd rather run natively:

```powershell
# Prerequisites: Python 3.11+, Node 18+, git, Docker Desktop — install from
# their official Windows installers.

git clone <your-repo-url>
cd <repo-folder>

cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cd ..

npm install
```

---

## Configuration

### 1. Create a Supabase project

1. Go to the [Supabase dashboard](https://supabase.com/dashboard) →
   **New Project**.
2. Choose a name and region, and set a database password —
   **save this password**, it's shown only once and you'll need it below.
3. Wait for provisioning to finish (~2 minutes).

### 2. Run the database migration

The schema is **not created automatically**. In the Supabase dashboard:

1. Go to **SQL Editor → New query**.
2. Open `supabase/migrations/20260702081437_001_articles_schema.sql` from
   this repo, copy its entire contents, paste into the query editor, and
   click **Run**.
3. Verify under **Table Editor** that `articles`, `article_entities`,
   `article_keyphrases`, and `tracked_entities` now exist.

Skipping this step is the most common cause of a confusing failure mode:
the backend will report a successful Supabase connection at startup, but
every actual database query will then fail with a "relation does not
exist" error, which looks like an unrelated bug.

### 3. Collect credentials

From **Settings → API** in your Supabase project:
- `Project URL` → `SUPABASE_URL`
- `anon public` key → `SUPABASE_ANON_KEY`
- `service_role secret` key (click the eye icon) → `SUPABASE_SERVICE_ROLE_KEY`

From **Settings → Database → Connection string**, select **Transaction
pooler** mode, copy the string, and substitute your database password from
Step 1 → `DATABASE_URL`.

From [console.groq.com/keys](https://console.groq.com/keys), create a key
→ `GROQ_API_KEY`.

### 4. Fill in environment files

```bash
cp backend/.env.example backend/.env
# edit backend/.env with the real values collected above

cp src/.env.example .env
# the default (http://localhost:8000) is correct unless you're running
# the backend on a different port or host
```

Never commit either `.env` file (both are already gitignored).

---

## Environment Variables

### `backend/.env`

| Variable | Required | Example | What it does |
|---|---|---|---|
| `SUPABASE_URL` | Yes | `https://abcxyz.supabase.co` | Base URL of your Supabase project |
| `SUPABASE_ANON_KEY` | Yes | `eyJhbGciOi...` | Public anon key, used for standard REST access |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | `eyJhbGciOi...` | Elevated key used server-side to bypass row-level security |
| `DATABASE_URL` | Yes | `postgresql://postgres.xxx:pw@aws-0-region.pooler.supabase.com:6543/postgres` | Direct Postgres connection string (transaction pooler mode) |
| `REDIS_URL` | Yes | `redis://localhost:6379` | Redis connection for the ingestion stream and PIE caching |
| `ELASTICSEARCH_URL` | Yes | `http://localhost:9200` | Elasticsearch connection for search and analytics |
| `GROQ_API_KEY` | Yes* | `gsk_...` | LLM inference for Intelligence Feed, Timeline, and Trend Explanation. Without it, those features fail or fall back to non-LLM output |
| `NEWSAPI_KEY` | No | — | Reserved for an additional news source; RSS ingestion does not require it |
| `ENVIRONMENT` | No | `development` | Informational; used in the health check response |
| `LOG_LEVEL` | No | `INFO` | structlog verbosity |

### Root `.env` (frontend)

| Variable | Required | Example | What it does |
|---|---|---|---|
| `VITE_API_URL` | No | `http://localhost:8000` | Backend base URL. Defaults to `http://localhost:8000` in code if unset, so this file can be empty for local development |

> **Note:** You may see `VITE_SUPABASE_URL` / `VITE_SUPABASE_ANON_KEY`
> referenced in older versions of this project's `.env`. The frontend does
> **not** actually use these — it talks exclusively to the FastAPI backend,
> which is the only thing that talks to Supabase. The `@supabase/supabase-js`
> package in `package.json` is an unused leftover dependency; it is safe to
> remove along with those two variables if you want to clean up further.

---

## Running the Project

Everything below assumes Docker Desktop is running. Order matters —
Elasticsearch needs ~20 seconds to become ready before the backend starts.

### 1. Start infrastructure

```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine

docker run -d --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" \
  docker.elastic.co/elasticsearch/elasticsearch:8.12.0

sleep 20
curl http://localhost:9200   # should return a JSON blob with cluster_name
```

If these containers already exist from a previous run, use
`docker start redis` / `docker start elasticsearch` instead of `docker run`.

### 2. Backend (Terminal 1)

```bash
cd backend
source venv/bin/activate        # Windows: venv\Scripts\activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Watch for `redis_connected`, `supabase_connected`, `elasticsearch_connected`,
then `application_started`, with no errors in between.

> **Avoid `--reload` for this project.** During development we found it can
> silently fail to pick up saved file changes on some setups, leaving an
> old process appearing to run new code. If an edit doesn't seem to take
> effect, fully kill the process (`lsof -ti :8000 | xargs kill -9` on
> macOS/Linux) and restart plain `uvicorn` rather than trusting `--reload`.

### 3. NLP Worker (Terminal 2)

```bash
cd backend
source venv/bin/activate
python -m app.workers.nlp_worker worker-1
```

spaCy and the sentiment model download automatically on first run if not
already cached — this can take a minute with **no console output**, which
looks like a hang but isn't.

**Do not `Ctrl+C` this process mid-batch.** Redis Streams tracks which
messages have been delivered to this worker; if killed before
acknowledging them and restarted, those messages sit in a "pending,
unacknowledged" state and are **not** automatically redelivered. The
worker will appear to hang waiting for new messages that will never
arrive, even though a real backlog exists. If this happens:
```bash
docker exec -it redis redis-cli DEL articles:raw
```
then re-run ingestion (step 5) to push fresh messages onto the stream.

### 4. Frontend (Terminal 3)

```bash
npm run dev
```
Open `http://localhost:5173`.

### 5. Ingest articles

```bash
curl -X POST http://localhost:8000/ingest/fetch
```

Watch Terminal 2 — it should immediately start processing the newly
fetched articles, roughly 5–10 seconds each on CPU-only hardware.

> `POST /ingest/process` also exists but is a **stub** — it only lists
> pending article IDs and does not actually process anything. Real
> processing happens automatically via the NLP worker consuming the Redis
> Stream that `/ingest/fetch` populates; you don't need to call
> `/ingest/process` at all under normal use.

### 6. Generate the Intelligence Feed

There is **no background scheduler** in this build — `apscheduler` is
listed in `requirements.txt` but is not wired up anywhere in the code, so
detection and feed generation must be triggered manually (or you can wire
up a scheduler yourself; see the report's Future Scope section):

```bash
curl -X POST http://localhost:8000/intelligence/emergence/detect
curl -X POST http://localhost:8000/intelligence/feed/generate
```

Then refresh the Intelligence Feed tab in the browser.

---

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── agents/          # LangGraph workflows (timeline, trend explanation)
│   │   │   └── workflows.py
│   │   ├── routers/         # FastAPI route definitions
│   │   │   ├── search.py         # /search/*
│   │   │   ├── intelligence.py   # /intelligence/*
│   │   │   ├── agent.py          # /agent/*
│   │   │   └── articles.py       # /articles/*
│   │   ├── services/
│   │   │   ├── fetcher.py            # RSS ingestion
│   │   │   ├── proactive_intelligence.py  # PIE: emergence detection, narrative monitoring, feed generation
│   │   │   ├── reranker.py           # cross-encoder reranking
│   │   │   └── temporal_analytics.py # trend/volume/spike computations
│   │   ├── utils/
│   │   │   ├── supabase_client.py      # actual DB client in use (REST-based)
│   │   │   ├── database_client.py      # unused asyncpg-based client — not connected anywhere, ignore
│   │   │   ├── elasticsearch_client.py
│   │   │   └── redis_client.py
│   │   ├── workers/
│   │   │   └── nlp_worker.py       # NER, sentiment, keyphrases, embeddings
│   │   ├── config.py       # all Settings / env var definitions
│   │   └── main.py         # app entrypoint, router registration, /health, /ingest/*
│   ├── requeue_pending.py  # one-off debugging utility, not part of the app — safe to delete
│   ├── .env.example
│   └── requirements.txt
├── src/                     # React frontend
│   ├── App.tsx              # all views live in this single file
│   ├── main.tsx
│   └── .env.example
├── supabase/
│   └── migrations/          # run these manually — see Configuration
├── .bolt/                   # Bolt.new IDE scaffold config — harmless, safe to delete
├── package.json
└── vite.config.ts
```

---

## Architecture

**Data flow, start to finish:**

1. **Ingestion** — `services/fetcher.py` polls a fixed list of RSS feeds
   (see `RSS_FEEDS` in that file), skips URLs already in the database,
   extracts full article text with `trafilatura`, inserts a canonical
   `pending`-status row into Postgres, and pushes a message onto the
   `articles:raw` Redis Stream.
2. **Enrichment** — `workers/nlp_worker.py`, running as an independent
   long-lived process under a Redis consumer group, dequeues each message
   and runs spaCy NER, DistilBERT sentiment classification, keyphrase
   extraction, and sentence-transformers embedding generation, then writes
   the enriched record to both Postgres (via the Supabase REST client) and
   Elasticsearch (as a retrieval-ready document), and acknowledges the
   message.
3. **Retrieval** — `routers/search.py` exposes hybrid search: a BM25 query
   and a dense-vector kNN query both run against Elasticsearch, fused via
   reciprocal rank fusion, with optional cross-encoder reranking on top.
4. **Proactive Intelligence** — `services/proactive_intelligence.py`
   contains three cooperating pieces:
   - `EmergenceDetector` — pulls recent articles from Elasticsearch,
     clusters their embeddings with DBSCAN (cosine metric), recursively
     re-splits any cluster larger than a size threshold, and compares
     cluster centroids against previously-seen ones (cached in Redis) to
     avoid re-reporting an already-known topic.
   - `NarrativeMonitor` — tracks configured entities' sentiment over a
     rolling window and flags shifts/reversals.
   - `FeedGenerator` — deduplicates and scores candidates from the two
     above, then calls the Groq LLM with an explicit grounding + cohesion
     instruction (the model is told to verify the supplied articles
     actually share a common event before writing a summary, and to
     decline otherwise) to produce the final feed item. Falls back to the
     lead article's own text, not a generic template, if the LLM is
     unavailable or declines.
5. **Agentic reasoning** — `agents/workflows.py` implements two bounded-turn
   LangGraph workflows (Timeline, Trend Explanation) that retrieve
   supporting data first and only then call the LLM to synthesize a
   narrative.
6. **Frontend** — `src/App.tsx` renders four views (Intelligence Feed,
   Search, Trending, Analyst), all talking exclusively to the FastAPI
   backend over `VITE_API_URL`.

**On the relational schema:** the `articles` table includes a
`vector(384)` column via the `pgvector` Postgres extension. In this build,
that column exists at the schema level but the operative vector search at
runtime goes through Elasticsearch's own `dense_vector` field, not
Postgres — don't be confused if you go looking for pgvector query code and
don't find it in the search path.

---

## API

Full interactive documentation is auto-generated at `http://localhost:8000/docs`
once the backend is running. Below are the routes grouped by router, verified
against the actual code.

### Search — `/search`
| Method | Path | Notes |
|---|---|---|
| POST | `/search` | Main hybrid search endpoint |
| GET | `/search/keyword` | BM25-only |
| GET | `/search/semantic` | Vector-only |
| GET | `/search/similar/{article_id}` | Find similar articles |

**Example — `POST /search`**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "semiconductor export controls", "size": 10, "use_reranking": true}'
```
Request body fields: `query` (string, required), `filters` (object,
optional), `size` (int, default 20), `use_reranking` (bool, default true).

Response shape:
```json
{
  "query": "semiconductor export controls",
  "results": [ { "...": "article fields with _score" } ],
  "total": 10,
  "latency_ms": 143.2,
  "used_reranking": true
}
```

### Intelligence — `/intelligence`
| Method | Path | Notes |
|---|---|---|
| GET | `/intelligence/feed` | Read the current cached feed |
| POST | `/intelligence/feed/generate` | Regenerate the feed from current candidates |
| POST | `/intelligence/emergence/detect` | Run topic emergence detection, populate candidates |
| POST | `/intelligence/narrative/monitor` | Run narrative shift monitoring |
| GET | `/intelligence/trending` | Trending entities by coverage velocity |
| GET | `/intelligence/digest` | Daily digest |
| POST | `/intelligence/digest/refresh` | Regenerate the daily digest |
| GET | `/intelligence/entity/{entity_name}/trend` | Coverage/sentiment trend for one entity |
| GET | `/intelligence/entity/{entity_name}/articles` | Articles mentioning an entity |
| GET | `/intelligence/topic/{topic}/trend` | Coverage trend for a free-text topic |
| GET | `/intelligence/volume` | Article volume over time |

Run `emergence/detect` and (optionally) `narrative/monitor` **before**
`feed/generate` — the generator only reads from candidates those two
endpoints produce; it does not detect anything itself.

### Agent — `/agent`
| Method | Path | Notes |
|---|---|---|
| POST | `/agent/timeline` | Chronological timeline for an entity/topic |
| POST | `/agent/explain-trend` | Natural-language explanation of why something is trending |
| POST | `/agent/ask` | General analyst query |

**Example — `POST /agent/timeline`**
```bash
curl -X POST http://localhost:8000/agent/timeline \
  -H "Content-Type: application/json" \
  -d '{"query": "timeline of the Kyiv drone strikes"}'
```
Request body: `{"query": "<free text>"}` for both `/timeline` and
`/explain-trend`.

### Articles — `/articles`
| Method | Path | Notes |
|---|---|---|
| GET | `/articles/{article_id}` | Fetch one article |
| GET | `/articles/` | List with filters |
| GET | `/articles/recent` | Most recently ingested |
| GET | `/articles/stats` | Corpus statistics |
| DELETE | `/articles/{article_id}` | Delete an article |

### Top-level
| Method | Path | Notes |
|---|---|---|
| GET | `/health` | Service connectivity status (redis/es/supabase) |
| GET | `/` | API metadata |
| POST | `/ingest/fetch` | Trigger RSS ingestion |
| POST | `/ingest/process` | **Stub — lists pending articles, does not process them** |

---

## Troubleshooting

**`Connection refused` on port 6379 or 9200**
Redis or Elasticsearch isn't running. `docker ps` to check; `docker start redis`
/ `docker start elasticsearch` to bring an existing container back up.

**Elasticsearch container exits immediately / won't start (Linux)**
```bash
sudo sysctl -w vm.max_map_count=262144
```
Elasticsearch requires this kernel setting on Linux hosts.

**`relation "articles" does not exist`**
You skipped the Supabase migration step. See Configuration → step 2.

**`ImportError` for a package that's clearly in `requirements.txt`**
Make sure the virtualenv is actually activated (`source venv/bin/activate`,
prompt should show `(venv)`) before running `pip install` or starting any
Python process. This bit us more than once during development.

**NLP worker seems frozen after starting**
Two possible causes:
- First run, silently downloading spaCy/DistilBERT models — wait a minute,
  this produces no console output.
- Genuine Redis Streams PEL deadlock from a previous unclean restart — see
  the note in Running the Project → step 3.

**Feed generation returns items but they look stale / unchanged**
The Proactive Intelligence Engine is fully deterministic given the same
input — DBSCAN and the LLM cohesion check will reliably produce the same
clusters from the same underlying articles. If output looks "stuck," you
likely need genuinely new articles (`POST /ingest/fetch`) rather than a
cache-clearing fix.

**Groq errors mentioning a decommissioned/unsupported model**
Groq periodically retires models. If you hit this, check
[console.groq.com/docs/models](https://console.groq.com/docs/models) for
currently supported models and update the `model=` string(s) in
`backend/app/agents/workflows.py` and
`backend/app/services/proactive_intelligence.py`. Note there is also a
`llama-3.1-8b-instant` reference in `workflows.py` that was not touched
during this project's last round of fixes — check that one too if agent
endpoints misbehave.

**Trending Entities all show 0% velocity**
Velocity is computed from articles published in the last 24 hours. A
one-time backlog fetch of older articles will legitimately show 0% until
you fetch again and let time pass.

**CORS errors from the frontend**
Shouldn't happen locally — CORS is currently wide open
(`allow_origins=["*"]` in `main.py`) for development convenience. If you
see a CORS error, double-check `VITE_API_URL` actually points at your
running backend.

**Port already in use**
```bash
lsof -ti :8000 | xargs kill -9   # backend
lsof -ti :5173 | xargs kill -9   # frontend
```

**Permission errors on macOS/Linux running Docker commands**
Ensure Docker Desktop is actually running (not just installed) and your
user has permission to use the Docker socket.

---

## Known Issues & Housekeeping

None of these block normal use, but are worth knowing about:

- `@supabase/supabase-js` in `package.json` and `database_client.py` in
  the backend are both dead/unused code, safe to remove.
- `apscheduler` in `requirements.txt` is installed but never imported or
  wired up anywhere.
- `backend/requeue_pending.py` is a one-off debugging script written
  during development, not part of the running application.
- `.bolt/` is leftover Bolt.new IDE scaffold configuration; harmless, safe
  to delete if you're not using Bolt.
- CORS is wide open (`allow_origins=["*"]`) — fine for local development,
  should be restricted before any real deployment.

---

## License

MIT License
