import PostEditor, { type PostFormState } from "@/components/admin/PostEditor";

const EMPTY: PostFormState = {
  title: "", slug: "", excerpt: "", content_md: "", category: "Market Commentary",
  tags: "", author: "Quant Research Team", status: "draft",
};

export default function NewPostPage() {
  return (
    <div>
      <h1 className="text-xl font-semibold text-slate-100 mb-4">New Post</h1>
      <PostEditor initial={EMPTY} />
    </div>
  );
}
