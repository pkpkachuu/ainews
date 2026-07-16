# Setup Notes — AI-Powered Intelligent News Analytics Platform

Follow these steps **in order**. Skipping the Supabase migration step is the
most common cause of confusing "it connects but nothing works" errors.

## Prerequisites
- Python 3.11+
- Node.js 18+
- Docker Desktop (for Redis and Elasticsearch)
- A free [Groq](https://console.groq.com) account (for the API key)
- A free [Supabase](https://supabase.com) account

---

## 1. Create a Supabase project

1. Go to [supabase.com/dashboard](https://supabase.com/dashboard) → **New Project**.
2. Pick a name and region, and set a database password — **write this
   password down**, you'll need it for `DATABASE_URL` below and it is only
   shown once.
3. Wait ~2 minutes for the project to finish provisioning.

## 2. Run the database migration

The table schema is **not** created automatically — you must run it once
against your new project.

1. Open the file `supabase/migrations/20260702081437_001_articles_schema.sql`
   in this repo.
2. In the Supabase dashboard, go to **SQL Editor → New query**.
3. Paste the entire contents of that file and click **Run**.
4. Confirm it succeeded — go to **Table Editor** and check that the
   `articles` table (and related tables) now exist.

If you skip this step, the backend will start up and report a successful
Supabase connection, but every actual query will fail with a
"relation does not exist" error — which looks like a different bug and
is easy to misdiagnose.

## 3. Collect your credentials

**Settings → API** in the Supabase dashboard:
- `Project URL` → this is `SUPABASE_URL`
- `anon public` key → this is `SUPABASE_ANON_KEY`
- `service_role secret` key (click the eye icon to reveal) → this is
  `SUPABASE_SERVICE_ROLE_KEY`

**Settings → Database → Connection string** → select **Transaction pooler**
mode → copy the connection string and substitute in the database password
you set in Step 1. This is `DATABASE_URL`.

**Groq API key**: [console.groq.com/keys](https://console.groq.com/keys) →
create a new key. This is `GROQ_API_KEY`.

## 4. Fill in environment files

**`backend/.env`** (copy from `backend/.env.example` first):
```
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
DATABASE_URL=postgresql://postgres.your-project-ref:YOUR-PASSWORD@aws-0-region.pooler.supabase.com:6543/postgres
GROQ_API_KEY=gsk_your-groq-key
REDIS_URL=redis://localhost:6379
ELASTICSEARCH_URL=http://localhost:9200
```

**Root `.env`** (frontend-facing, copy from `.env.example`):
```
VITE_SUPABASE_URL=https://your-project-ref.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key
```

Never commit either `.env` file — they should already be listed in
`.gitignore`.

## 5. Start Redis and Elasticsearch

```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine

docker run -d --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" \
  docker.elastic.co/elasticsearch/elasticsearch:8.12.0
```

Wait ~20 seconds, then confirm Elasticsearch is up:
```bash
curl http://localhost:9200
```
You should get back a JSON blob with a `cluster_name` field. If Redis or
Elasticsearch containers already exist from a previous run, use
`docker start redis` / `docker start elasticsearch` instead of `docker run`.

## 6. Backend setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

spaCy's model (`en_core_web_sm`) and the DistilBERT sentiment model download
automatically on first run if not already present. **This can take a minute
and produces no console output while downloading — it is not frozen.**

## 7. Start the backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Watch the startup log — you should see `redis_connected`,
`supabase_connected`, and `elasticsearch_connected` with no errors, followed
by `application_started`.

> Avoid `--reload` for this project. We found during development that it
> can silently fail to pick up file changes on some setups, making an
> old process appear to still be running the new code. If something you
> just edited doesn't seem to take effect, fully stop the process
> (`lsof -ti :8000 | xargs kill -9`) and restart plain `uvicorn` rather
> than trusting `--reload`.

## 8. Start the NLP worker

In a second terminal:
```bash
cd backend
source venv/bin/activate
python -m app.workers.nlp_worker worker-1
```

**Do not `Ctrl+C` this process in the middle of processing a batch of
articles.** Redis Streams tracks which messages have been delivered to this
worker; if you kill it mid-batch and restart, those in-flight messages sit
in a "pending, unacknowledged" state and will **not** be automatically
redelivered — the worker will look like it's hanging even though it's
actually just waiting for genuinely new messages that will never arrive.

If this happens: stop the worker, then run
```bash
docker exec -it redis redis-cli DEL articles:raw
```
and re-trigger ingestion (Step 10) to get fresh messages onto the stream.

## 9. Start the frontend

In a third terminal:
```bash
npm install
npm run dev
```
Open `http://localhost:5173`.

If the page loads blank with a browser console error mentioning
`ERR_BLOCKED_BY_CLIENT` on a file named `fingerprint.js`: this is a browser
ad-blocker/privacy-extension flagging an icon filename that happens to
contain the word "fingerprint" — it has nothing to do with tracking. Disable
your ad blocker for `localhost`, or check `chrome://extensions` for any
extension with "Allow in Incognito" enabled (those survive private
browsing windows).

## 10. Fetch articles

```bash
curl -X POST http://localhost:8000/ingest/fetch
```

Watch the NLP worker terminal — it should immediately start processing the
newly fetched articles. On CPU-only hardware, expect roughly 5–10 seconds
per article.

## 11. Generate the Intelligence Feed

The Proactive Intelligence Engine does **not** run on a background schedule
in this MVP — it must be triggered manually (or you can wire up a scheduler
yourself; see Future Scope in the report). After some articles have
finished processing:

```bash
curl -X POST http://localhost:8000/intelligence/emergence/detect
curl -X POST http://localhost:8000/intelligence/feed/generate
```

Then refresh the Intelligence Feed tab in the browser.

---

## Known non-bugs (don't waste time debugging these)

- **spaCy/DistilBERT "hanging" on first run** — it's downloading, not stuck.
- **Trending Entities showing 0% velocity** on a fresh dataset — velocity is
  based on articles published in the last 24 hours; a one-time backlog fetch
  of older articles will legitimately show 0% until you fetch again later.
- **Empty Intelligence Feed after a fresh fetch** — you must run Step 11
  manually; there is no automatic background trigger yet.
- **`requeue_pending.py`** in `backend/` is a one-off debugging utility, not
  part of the application. Safe to ignore or delete.
