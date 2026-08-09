import { NextRequest, NextResponse } from "next/server";
import { requireAdmin } from "@/lib/require-admin";
import { deletePost, getPostById, slugify, updatePost, type PostInput } from "@/lib/db/posts";

type Params = { params: Promise<{ id: string }> };

export async function GET(_req: NextRequest, { params }: Params) {
  const unauthorized = await requireAdmin();
  if (unauthorized) return unauthorized;
  const { id } = await params;
  const post = await getPostById(Number(id));
  if (!post) return NextResponse.json({ error: "Post not found." }, { status: 404 });
  return NextResponse.json({ post });
}

export async function PUT(req: NextRequest, { params }: Params) {
  const unauthorized = await requireAdmin();
  if (unauthorized) return unauthorized;
  const { id } = await params;
  const postId = Number(id);

  const existing = await getPostById(postId);
  if (!existing) return NextResponse.json({ error: "Post not found." }, { status: 404 });

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
    const post = await updatePost(postId, input);
    return NextResponse.json({ post });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    const isDuplicateSlug = message.includes("duplicate key") && message.includes("slug");
    return NextResponse.json(
      { error: isDuplicateSlug ? `Slug '${input.slug}' is already in use.` : message },
      { status: isDuplicateSlug ? 409 : 500 }
    );
  }
}

export async function DELETE(_req: NextRequest, { params }: Params) {
  const unauthorized = await requireAdmin();
  if (unauthorized) return unauthorized;
  const { id } = await params;
  await deletePost(Number(id));
  return NextResponse.json({ ok: true });
}
