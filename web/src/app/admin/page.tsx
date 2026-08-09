import Link from "next/link";
import { listAllPosts } from "@/lib/db/posts";
import PostRowActions from "@/components/admin/PostRowActions";

export const dynamic = "force-dynamic";

export default async function AdminDashboard() {
  const posts = await listAllPosts();

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-semibold text-slate-100">Posts</h1>
        <Link href="/admin/posts/new" className="rounded-md bg-emerald-600 hover:bg-emerald-500 px-4 py-2 text-sm font-medium text-white transition-colors">
          + New Post
        </Link>
      </div>

      {posts.length === 0 ? (
        <p className="text-slate-400 text-sm">No posts yet. Create your first one.</p>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-slate-800">
          <table className="min-w-full text-sm">
            <thead className="bg-slate-900 text-slate-300">
              <tr>
                <th className="px-3 py-2 text-left font-medium">Title</th>
                <th className="px-3 py-2 text-left font-medium">Category</th>
                <th className="px-3 py-2 text-left font-medium">Status</th>
                <th className="px-3 py-2 text-left font-medium">Updated</th>
                <th className="px-3 py-2 text-left font-medium">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {posts.map((p) => (
                <tr key={p.id} className="hover:bg-slate-900/50">
                  <td className="px-3 py-2 text-slate-200">{p.title}</td>
                  <td className="px-3 py-2 text-slate-400">{p.category}</td>
                  <td className="px-3 py-2">
                    <span className={`rounded-full px-2 py-0.5 text-xs ${p.status === "published" ? "bg-emerald-500/15 text-emerald-300" : "bg-slate-700/50 text-slate-300"}`}>
                      {p.status}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-slate-500">{new Date(p.updated_at).toLocaleDateString()}</td>
                  <td className="px-3 py-2">
                    <div className="flex gap-3">
                      <Link href={`/admin/posts/${p.id}/edit`} className="text-xs text-emerald-400 hover:text-emerald-300">Edit</Link>
                      <PostRowActions id={p.id} title={p.title} />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
