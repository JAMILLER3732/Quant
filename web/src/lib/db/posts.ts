import { sql } from "./client";
export { slugify } from "@/lib/slugify-client";

export type Post = {
  id: number;
  slug: string;
  title: string;
  excerpt: string;
  content_md: string;
  category: string;
  tags: string[];
  author: string;
  status: "draft" | "published";
  published_at: string | null;
  created_at: string;
  updated_at: string;
};

export type PostInput = {
  slug: string;
  title: string;
  excerpt: string;
  content_md: string;
  category: string;
  tags: string[];
  author: string;
  status: "draft" | "published";
};

export async function listPublishedPosts(): Promise<Post[]> {
  const db = sql();
  return db<Post[]>`
    SELECT * FROM posts WHERE status = 'published' ORDER BY published_at DESC
  `;
}

export async function getPublishedPostBySlug(slug: string): Promise<Post | null> {
  const db = sql();
  const rows = await db<Post[]>`
    SELECT * FROM posts WHERE slug = ${slug} AND status = 'published' LIMIT 1
  `;
  return rows[0] ?? null;
}

export async function listAllPosts(): Promise<Post[]> {
  const db = sql();
  return db<Post[]>`SELECT * FROM posts ORDER BY updated_at DESC`;
}

export async function getPostById(id: number): Promise<Post | null> {
  const db = sql();
  const rows = await db<Post[]>`SELECT * FROM posts WHERE id = ${id} LIMIT 1`;
  return rows[0] ?? null;
}

export async function createPost(input: PostInput): Promise<Post> {
  const db = sql();
  const publishedAt = input.status === "published" ? new Date() : null;
  const rows = await db<Post[]>`
    INSERT INTO posts (slug, title, excerpt, content_md, category, tags, author, status, published_at)
    VALUES (${input.slug}, ${input.title}, ${input.excerpt}, ${input.content_md}, ${input.category},
            ${input.tags}, ${input.author}, ${input.status}, ${publishedAt})
    RETURNING *
  `;
  return rows[0];
}

export async function updatePost(id: number, input: PostInput): Promise<Post> {
  const db = sql();
  const existing = await getPostById(id);
  // Only stamp published_at the first time a post transitions into 'published'
  // (don't reset it on every subsequent edit of an already-published post).
  const publishedAt =
    input.status === "published" ? existing?.published_at ?? new Date() : null;
  const rows = await db<Post[]>`
    UPDATE posts SET
      slug = ${input.slug}, title = ${input.title}, excerpt = ${input.excerpt},
      content_md = ${input.content_md}, category = ${input.category}, tags = ${input.tags},
      author = ${input.author}, status = ${input.status}, published_at = ${publishedAt},
      updated_at = now()
    WHERE id = ${id}
    RETURNING *
  `;
  return rows[0];
}

export async function deletePost(id: number): Promise<void> {
  const db = sql();
  await db`DELETE FROM posts WHERE id = ${id}`;
}
