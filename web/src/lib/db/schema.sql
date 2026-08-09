-- Phase 4 schema: admin-authored blog/research posts.
-- Applied once via `npm run db:migrate` (see src/lib/db/migrate.ts).

CREATE TABLE IF NOT EXISTS posts (
  id            SERIAL PRIMARY KEY,
  slug          TEXT NOT NULL UNIQUE,
  title         TEXT NOT NULL,
  excerpt       TEXT NOT NULL DEFAULT '',
  content_md    TEXT NOT NULL DEFAULT '',
  category      TEXT NOT NULL DEFAULT 'Market Commentary',
  tags          TEXT[] NOT NULL DEFAULT '{}',
  author        TEXT NOT NULL DEFAULT 'Quant Research Team',
  status        TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'published')),
  published_at  TIMESTAMPTZ,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_posts_status_published_at ON posts (status, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_posts_slug ON posts (slug);
