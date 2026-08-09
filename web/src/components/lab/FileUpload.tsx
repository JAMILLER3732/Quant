"use client";

import { useCallback, useRef, useState } from "react";

export default function FileUpload({
  onFileSelected,
  busy,
}: {
  onFileSelected: (file: File) => void;
  busy: boolean;
}) {
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      onFileSelected(files[0]);
    },
    [onFileSelected]
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragActive(true);
      }}
      onDragLeave={() => setDragActive(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragActive(false);
        handleFiles(e.dataTransfer.files);
      }}
      onClick={() => inputRef.current?.click()}
      className={`cursor-pointer rounded-xl border-2 border-dashed p-10 text-center transition-colors ${
        dragActive ? "border-emerald-400 bg-emerald-400/5" : "border-slate-700 hover:border-slate-500"
      } ${busy ? "opacity-60 pointer-events-none" : ""}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.xlsx,.xls"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
      <p className="text-lg font-medium text-slate-100">
        {busy ? "Uploading & analyzing…" : "Drop a CSV or Excel file here, or click to browse"}
      </p>
      <p className="mt-2 text-sm text-slate-400">
        .csv, .xlsx, .xls — up to 25MB. Your data is parsed and validated by the Python engine before any
        calculation runs.
      </p>
    </div>
  );
}
