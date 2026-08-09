import { NextRequest, NextResponse } from "next/server";
import { requireAdmin } from "@/lib/require-admin";
import { createPost, listAllPosts, slugify, type PostInput } from "@/lib/db/posts";

export async function GET() {
  const unauthorized = await requireAdmin();
  if (unauthorized) return unauthorized;
  const posts = await listAllPosts();
  return NextResponse.json({ posts });
}

export async function POST(req: NextRequest) {
  const unauthorized = await requireAdmin();
  if (unauthorized) return unauthorized;

  const body = await req.json();
  const title = String(body.title ?? "").trim();
  if (!title) return NextResponse.json({ error: "Title is required." }, { status: 422 });

  const input: PostInput = {
    slug: (body.slug && String(body.slug).trim()) || slugify(title),
    title,
    excerpt: String(body.excerpt ?? ""),
    content_md: String(body.content_md ?? ""),
    category: String(body.category ?? "Market Commentary"),
    tags: Array.isArray(body.tags) ? body.tags.map(String) : [],
    author: String(body.author ?? "Quant Research Team"),
    status: body.status === "published" ? "published" : "draft",
  };

  try {
    const post = await createPost(input);
    return NextResponse.json({ post }, { status: 201 });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    const isDuplicateSlug = message.includes("duplicate key") && message.includes("slug");
    return NextResponse.json(
      { error: isDuplicateSlug ? `Slug '${input.slug}' is already in use.` : message },
      { status: isDuplicateSlug ? 409 : 500 }
    );
  }
}
