// One-shot schema migration: applies src/lib/db/schema.sql against POSTGRES_URL.
// Run with: npm run db:migrate
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import postgres from "postgres";

const url = process.env.POSTGRES_URL;
if (!url) {
  console.error("POSTGRES_URL is not set. Export it or add it to .env.local, then rerun.");
  process.exit(1);
}

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const schemaPath = path.join(__dirname, "..", "src", "lib", "db", "schema.sql");
const schema = readFileSync(schemaPath, "utf-8");

const sql = postgres(url, { ssl: "require", max: 1 });

try {
  await sql.unsafe(schema);
  console.log("Migration applied successfully.");
} catch (err) {
  console.error("Migration failed:", err);
  process.exit(1);
} finally {
  await sql.end();
}
