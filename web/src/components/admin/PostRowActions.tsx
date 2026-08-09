"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

export default function PostRowActions({ id, title }: { id: number; title: string }) {
  const router = useRouter();
  const [busy, setBusy] = useState(false);

  async function handleDelete() {
    if (!confirm(`Delete "${title}"? This cannot be undone.`)) return;
    setBusy(true);
    try {
      await fetch(`/api/admin/posts/${id}`, { method: "DELETE" });
      router.refresh();
    } finally {
      setBusy(false);
    }
  }

  return (
    <button onClick={handleDelete} disabled={busy} className="text-xs text-red-400 hover:text-red-300 disabled:opacity-50">
      {busy ? "Deleting…" : "Delete"}
    </button>
  );
}
