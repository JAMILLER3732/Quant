import postgres from "postgres";

// Lazily create a single pooled connection, reused across requests in the
// same server process (standard for serverless/Next.js route handlers).
// Requires POSTGRES_URL (any standard Postgres connection string — Neon,
// Supabase, or otherwise) to be set; throws a clear error if it's missing
// rather than silently failing on first query.
let _sql: ReturnType<typeof postgres> | null = null;

export function sql() {
  if (!_sql) {
    const url = process.env.POSTGRES_URL;
    if (!url) {
      throw new Error(
        "POSTGRES_URL is not set. Add a Postgres connection string (Neon/Supabase) as an environment variable."
      );
    }
    _sql = postgres(url, { ssl: "require", max: 5, idle_timeout: 20 });
  }
  return _sql;
}
