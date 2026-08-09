// Client-safe duplicate of the slugify logic in lib/db/posts.ts (which also
// imports the `postgres` driver — not something we want in the browser
// bundle). Keep these two in sync if the slugify rule ever changes.
export function slugify(title: string): string {
  return title
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "")
    .slice(0, 80);
}
