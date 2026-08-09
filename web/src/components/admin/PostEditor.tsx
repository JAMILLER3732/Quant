"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { slugify } from "@/lib/slugify-client";

export type PostFormState = {
  id?: number;
  title: string;
  slug: string;
  excerpt: string;
  content_md: string;
  category: string;
  tags: string; // comma-separated in the form, split on submit
  author: string;
  status: "draft" | "published";
};

const CATEGORIES = [
  "Quantitative Finance", "Portfolio Management", "Risk Management", "Asset Allocation",
  "Equities", "Fixed Income", "Derivatives", "Volatility", "Macro", "Economics",
  "Factor Investing", "Machine Learning", "Financial Engineering", "Market Microstructure",
  "Trading", "Market Commentary",
];

type Tab = "content" | "preview" | "details";

export default function PostEditor({ initial }: { initial: PostFormState }) {
  const router = useRouter();
  const [form, setForm] = useState<PostFormState>(initial);
  const [tab, setTab] = useState<Tab>("content");
  const [slugTouched, setSlugTouched] = useState(Boolean(initial.slug));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function update<K extends keyof PostFormState>(key: K, value: PostFormState[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  function handleTitleChange(title: string) {
    update("title", title);
    if (!slugTouched) update("slug", slugify(title));
  }

  async function save(status: "draft" | "published") {
    setSaving(true);
    setError(null);
    const payload = {
      ...form,
      status,
      tags: form.tags.split(",").map((t) => t.trim()).filter(Boolean),
    };
    try {
      const url = form.id ? `/api/admin/posts/${form.id}` : "/api/admin/posts";
      const method = form.id ? "PUT" : "POST";
      const res = await fetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = await res.json();
      if (!res.ok) {
        setError(body.error || "Save failed.");
        return;
      }
      router.push("/admin");
      router.refresh();
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex gap-2 border-b border-slate-800">
        {(["content", "preview", "details"] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium capitalize border-b-2 transition-colors ${
              tab === t ? "border-emerald-400 text-emerald-300" : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === "content" && (
        <div className="space-y-3">
          <input
            type="text"
            value={form.title}
            onChange={(e) => handleTitleChange(e.target.value)}
            placeholder="Post title"
            className="w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-lg font-semibold text-slate-100 focus:border-emerald-400 focus:outline-none"
          />
          <textarea
            value={form.content_md}
            onChange={(e) => update("content_md", e.target.value)}
            placeholder="Write in Markdown — headings, **bold**, tables, ```code```, and $LaTeX$ where supported."
            rows={24}
            className="w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-3 text-sm font-mono text-slate-100 focus:border-emerald-400 focus:outline-none resize-y"
          />
        </div>
      )}

      {tab === "preview" && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-8">
          <article className="prose prose-invert prose-slate max-w-none">
            <p className="text-xs uppercase tracking-wide text-emerald-400 mb-1">{form.category}</p>
            <h1 className="text-3xl font-semibold text-slate-50 mb-2">{form.title || "Untitled post"}</h1>
            <p className="text-sm text-slate-500 mb-6">
              By {form.author || "Quant Research Team"}
              {form.tags && ` · ${form.tags}`}
            </p>
            {form.excerpt && <p className="text-lg text-slate-300 mb-6 italic">{form.excerpt}</p>}
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {form.content_md || "*Nothing written yet — switch to the Content tab.*"}
            </ReactMarkdown>
          </article>
        </div>
      )}

      {tab === "details" && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 rounded-xl border border-slate-800 bg-slate-900/40 p-5">
          <Field label="URL slug">
            <input
              type="text"
              value={form.slug}
              onChange={(e) => {
                setSlugTouched(true);
                update("slug", e.target.value);
              }}
              className="input"
            />
          </Field>
          <Field label="Category">
            <select value={form.category} onChange={(e) => update("category", e.target.value)} className="input">
              {CATEGORIES.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </Field>
          <Field label="Author">
            <input type="text" value={form.author} onChange={(e) => update("author", e.target.value)} className="input" />
          </Field>
          <Field label="Tags (comma-separated)">
            <input type="text" value={form.tags} onChange={(e) => update("tags", e.target.value)} className="input" />
          </Field>
          <div className="sm:col-span-2">
            <Field label="Excerpt (shown on the post list)">
              <textarea value={form.excerpt} onChange={(e) => update("excerpt", e.target.value)} rows={3} className="input" />
            </Field>
          </div>
        </div>
      )}

      {error && <p className="text-sm text-red-400">{error}</p>}

      <div className="flex gap-3 pt-2">
        <button
          onClick={() => save("draft")}
          disabled={saving || !form.title}
          className="rounded-md border border-slate-700 hover:border-slate-500 disabled:opacity-50 px-4 py-2 text-sm font-medium text-slate-200 transition-colors"
        >
          Save Draft
        </button>
        <button
          onClick={() => save("published")}
          disabled={saving || !form.title}
          className="rounded-md bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 px-4 py-2 text-sm font-medium text-white transition-colors"
        >
          {form.status === "published" ? "Update & Keep Published" : "Publish"}
        </button>
      </div>

      <style jsx global>{`
        .input {
          width: 100%;
          border-radius: 0.375rem;
          border: 1px solid rgb(51 65 85);
          background: rgb(2 6 23);
          padding: 0.5rem 0.75rem;
          font-size: 0.875rem;
          color: rgb(241 245 249);
        }
        .input:focus { border-color: rgb(52 211 153); outline: none; }
      `}</style>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs font-medium text-slate-400">{label}</span>
      {children}
    </label>
  );
}
