"use client";
import { useState } from "react";

export default function UploadPage() {
  const [status, setStatus] = useState("");
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  async function handleUpload(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const formData = new FormData(e.currentTarget as HTMLFormElement);
    setStatus("Uploading...");

    try {
      const res = await fetch(`${API_URL}/documents`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "upload failed");
      }

      const data = await res.json();
      setStatus("Uploaded! Doc ID: " + data.doc_id);
    } catch (err: any) {
      setStatus("Error: " + err.message);
    }
  }

  return (
    <div className="p-10 h-screen flex flex-col items-center justify-center">
      <h1 className="text-2xl font-bold mb-6">Upload a Document</h1>
      <form onSubmit={handleUpload} className="flex flex-col items-center gap-4">
        <input type="file" name="file" className="border p-2" required />
        <button
          type="submit"
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Upload
        </button>
      </form>
      <p className="mt-4">{status}</p>
    </div>
  );
}