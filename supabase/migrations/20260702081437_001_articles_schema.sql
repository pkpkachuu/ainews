/*
# Create articles schema for News Analytics Platform

1. New Tables
- `articles`: Canonical store for news articles (source of truth)
  - article_id (uuid, primary key)
  - url (text, unique) - original article URL
  - url_hash (char(64)) - SHA256 hash for fast deduplication
  - title (text) - article headline
  - body (text) - article content
  - summary (text) - generated summary
  - source (varchar(100)) - news source name
  - published_at (timestamptz) - publication timestamp
  - language (char(5), default 'en') - language code
  - category (varchar(50)) - topic category
  - sentiment_score (float) - sentiment polarity (-1 to +1)
  - sentiment_label (varchar(20)) - sentiment classification
  - word_count (integer) - article length
  - raw_storage_path (text) - path to raw content in object storage
  - processing_status (varchar(20), default 'pending') - processing state
  - embedding (vector(384)) - semantic embedding for article
  - indexed_at (timestamptz) - when indexed to Elasticsearch
  - created_at (timestamptz) - record creation timestamp

- `article_entities`: Named entities extracted from articles
  - id (serial, primary key)
  - article_id (uuid, foreign key to articles)
  - entity_text (varchar(200)) - entity name
  - entity_type (varchar(50)) - PERSON, ORG, GPE, etc.
  - confidence (float) - extraction confidence

- `article_keyphrases`: Key phrases extracted from articles
  - id (serial, primary key)
  - article_id (uuid, foreign key to articles)
  - keyphrase (varchar(200)) - extracted phrase
  - score (float) - relevance score

- `tracked_entities`: Entities to monitor for proactive intelligence
  - id (uuid, primary key)
  - entity_name (varchar(200), unique) - entity to track
  - entity_type (varchar(50)) - PERSON, ORG, GPE
  - date_added (timestamptz) - when tracking started
  - is_active (boolean) - whether currently tracked

2. Indexes
- idx_articles_url_hash: Fast deduplication lookup
- idx_articles_published_at: Date range queries
- idx_articles_category: Category filtering
- idx_articles_processing_status: Worker queue queries
- idx_article_entities_article_id: Entity-article joins
- idx_article_entities_text: Entity name lookups
- idx_article_keyphrases_article_id: Keyphrase-article joins
- idx_tracked_entities_active: Active entity filtering

3. Security
- Enable RLS on all tables.
- Single-tenant: allow anon + authenticated full CRUD access.
*/

-- Enable pgvector extension for embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Articles table (canonical store)
CREATE TABLE IF NOT EXISTS articles (
    article_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    url text UNIQUE NOT NULL,
    url_hash char(64) NOT NULL,
    title text,
    body text,
    summary text,
    source varchar(100),
    published_at timestamptz,
    language char(5) DEFAULT 'en',
    category varchar(50),
    sentiment_score float,
    sentiment_label varchar(20),
    word_count integer,
    raw_storage_path text,
    processing_status varchar(20) DEFAULT 'pending',
    embedding vector(384),
    indexed_at timestamptz,
    created_at timestamptz DEFAULT now()
);

-- Article entities table
CREATE TABLE IF NOT EXISTS article_entities (
    id serial PRIMARY KEY,
    article_id uuid NOT NULL REFERENCES articles(article_id) ON DELETE CASCADE,
    entity_text varchar(200),
    entity_type varchar(50),
    confidence float
);

-- Article keyphrases table
CREATE TABLE IF NOT EXISTS article_keyphrases (
    id serial PRIMARY KEY,
    article_id uuid NOT NULL REFERENCES articles(article_id) ON DELETE CASCADE,
    keyphrase varchar(200),
    score float
);

-- Tracked entities for proactive intelligence
CREATE TABLE IF NOT EXISTS tracked_entities (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_name varchar(200) UNIQUE NOT NULL,
    entity_type varchar(50),
    date_added timestamptz DEFAULT now(),
    is_active boolean DEFAULT true
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_articles_url_hash ON articles(url_hash);
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_category ON articles(category);
CREATE INDEX IF NOT EXISTS idx_articles_processing_status ON articles(processing_status);
CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);

CREATE INDEX IF NOT EXISTS idx_article_entities_article_id ON article_entities(article_id);
CREATE INDEX IF NOT EXISTS idx_article_entities_text ON article_entities(entity_text);
CREATE INDEX IF NOT EXISTS idx_article_entities_type ON article_entities(entity_type);

CREATE INDEX IF NOT EXISTS idx_article_keyphrases_article_id ON article_keyphrases(article_id);
CREATE INDEX IF NOT EXISTS idx_article_keyphrases_phrase ON article_keyphrases(keyphrase);

CREATE INDEX IF NOT EXISTS idx_tracked_entities_active ON tracked_entities(is_active);
CREATE INDEX IF NOT EXISTS idx_tracked_entities_name ON tracked_entities(entity_name);

-- Enable RLS on all tables
ALTER TABLE articles ENABLE ROW LEVEL SECURITY;
ALTER TABLE article_entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE article_keyphrases ENABLE ROW LEVEL SECURITY;
ALTER TABLE tracked_entities ENABLE ROW LEVEL SECURITY;

-- Articles policies (single-tenant, full access)
DROP POLICY IF EXISTS "anon_select_articles" ON articles;
CREATE POLICY "anon_select_articles" ON articles FOR SELECT
    TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "anon_insert_articles" ON articles;
CREATE POLICY "anon_insert_articles" ON articles FOR INSERT
    TO anon, authenticated WITH CHECK (true);

DROP POLICY IF EXISTS "anon_update_articles" ON articles;
CREATE POLICY "anon_update_articles" ON articles FOR UPDATE
    TO anon, authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "anon_delete_articles" ON articles;
CREATE POLICY "anon_delete_articles" ON articles FOR DELETE
    TO anon, authenticated USING (true);

-- Article entities policies
DROP POLICY IF EXISTS "anon_select_article_entities" ON article_entities;
CREATE POLICY "anon_select_article_entities" ON article_entities FOR SELECT
    TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "anon_insert_article_entities" ON article_entities;
CREATE POLICY "anon_insert_article_entities" ON article_entities FOR INSERT
    TO anon, authenticated WITH CHECK (true);

DROP POLICY IF EXISTS "anon_update_article_entities" ON article_entities;
CREATE POLICY "anon_update_article_entities" ON article_entities FOR UPDATE
    TO anon, authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "anon_delete_article_entities" ON article_entities;
CREATE POLICY "anon_delete_article_entities" ON article_entities FOR DELETE
    TO anon, authenticated USING (true);

-- Article keyphrases policies
DROP POLICY IF EXISTS "anon_select_article_keyphrases" ON article_keyphrases;
CREATE POLICY "anon_select_article_keyphrases" ON article_keyphrases FOR SELECT
    TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "anon_insert_article_keyphrases" ON article_keyphrases;
CREATE POLICY "anon_insert_article_keyphrases" ON article_keyphrases FOR INSERT
    TO anon, authenticated WITH CHECK (true);

DROP POLICY IF EXISTS "anon_update_article_keyphrases" ON article_keyphrases;
CREATE POLICY "anon_update_article_keyphrases" ON article_keyphrases FOR UPDATE
    TO anon, authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "anon_delete_article_keyphrases" ON article_keyphrases;
CREATE POLICY "anon_delete_article_keyphrases" ON article_keyphrases FOR DELETE
    TO anon, authenticated USING (true);

-- Tracked entities policies
DROP POLICY IF EXISTS "anon_select_tracked_entities" ON tracked_entities;
CREATE POLICY "anon_select_tracked_entities" ON tracked_entities FOR SELECT
    TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "anon_insert_tracked_entities" ON tracked_entities;
CREATE POLICY "anon_insert_tracked_entities" ON tracked_entities FOR INSERT
    TO anon, authenticated WITH CHECK (true);

DROP POLICY IF EXISTS "anon_update_tracked_entities" ON tracked_entities;
CREATE POLICY "anon_update_tracked_entities" ON tracked_entities FOR UPDATE
    TO anon, authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "anon_delete_tracked_entities" ON tracked_entities;
CREATE POLICY "anon_delete_tracked_entities" ON tracked_entities FOR DELETE
    TO anon, authenticated USING (true);