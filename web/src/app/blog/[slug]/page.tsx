import Link from "next/link";
import { notFound } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { getPublishedPostBySlug } from "@/lib/db/posts";

export const dynamic = "force-dynamic";

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const post = await getPublishedPostBySlug(slug);
  return { title: post ? `${post.title} — Quant Analytics Platform` : "Post not found" };
}

export default async function BlogPostPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const post = await getPublishedPostBySlug(slug);
  if (!post) notFound();

  return (
    <div className="mx-auto max-w-3xl px-4 sm:px-6 lg:px-8 py-12">
      <Link href="/blog" className="text-sm text-slate-400 hover:text-slate-200">← All posts</Link>
      <article className="prose prose-invert prose-slate max-w-none mt-4">
        <p className="text-xs uppercase tracking-wide text-emerald-400 mb-1">{post.category}</p>
        <h1 className="text-3xl font-semibold text-slate-50 mb-2">{post.title}</h1>
        <p className="text-sm text-slate-500 mb-8">
          By {post.author} ·{" "}
          {post.published_at && new Date(post.published_at).toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" })}
          {post.tags.length > 0 && ` · ${post.tags.join(", ")}`}
        </p>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{post.content_md}</ReactMarkdown>
      </article>
    </div>
  );
}
