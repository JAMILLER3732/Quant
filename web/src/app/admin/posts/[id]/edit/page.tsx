import { notFound } from "next/navigation";
import { getPostById } from "@/lib/db/posts";
import PostEditor, { type PostFormState } from "@/components/admin/PostEditor";

export const dynamic = "force-dynamic";

export default async function EditPostPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const post = await getPostById(Number(id));
  if (!post) notFound();

  const initial: PostFormState = {
    id: post.id,
    title: post.title,
    slug: post.slug,
    excerpt: post.excerpt,
    content_md: post.content_md,
    category: post.category,
    tags: post.tags.join(", "),
    author: post.author,
    status: post.status,
  };

  return (
    <div>
      <h1 className="text-xl font-semibold text-slate-100 mb-4">Edit Post</h1>
      <PostEditor initial={initial} />
    </div>
  );
}
