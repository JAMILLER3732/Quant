import Link from "next/link";
import { listPublishedPosts } from "@/lib/db/posts";

export const dynamic = "force-dynamic";
export const metadata = { title: "Research & Insights — Quant Analytics Platform" };

export default async function BlogIndexPage() {
  const posts = await listPublishedPosts();

  return (
    <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 py-12">
      <p className="text-sm font-medium text-emerald-400 mb-2">Insights</p>
      <h1 className="text-3xl font-semibold text-slate-50 mb-8">Research & Market Commentary</h1>

      {posts.length === 0 ? (
        <p className="text-slate-400">No posts published yet — check back soon.</p>
      ) : (
        <div className="space-y-6">
          {posts.map((p) => (
            <Link key={p.id} href={`/blog/${p.slug}`} className="block rounded-xl border border-slate-800 bg-slate-900/40 p-6 hover:border-slate-600 transition-colors">
              <p className="text-xs uppercase tracking-wide text-emerald-400 mb-1">{p.category}</p>
              <h2 className="text-xl font-semibold text-slate-100 mb-2">{p.title}</h2>
              {p.excerpt && <p className="text-sm text-slate-400 mb-3">{p.excerpt}</p>}
              <p className="text-xs text-slate-500">
                {p.author} · {p.published_at ? new Date(p.published_at).toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" }) : ""}
                {p.tags.length > 0 && ` · ${p.tags.join(", ")}`}
              </p>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
