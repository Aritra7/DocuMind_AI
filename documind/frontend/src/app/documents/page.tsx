"use client";

import { useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type DocumentItem = {
  id: number;
  filename: string;
  status: "READY" | "PROCESSING" | "FAILED";
};

export default function DocumentsPage() {
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadDocs() {
      try {
        const res = await fetch(`${API_URL}/documents`);
        if (!res.ok) {
          throw new Error(`Error fetching docs: ${res.statusText}`);
        }
        const data = await res.json();
        setDocs(data);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    loadDocs();
  }, []);

  if (loading) return <div className="p-8">Loading documents...</div>;
  if (error) return <div className="p-8 text-red-500">Failed: {error}</div>;

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-6">My Documents</h1>

      {docs.length === 0 ? (
        <p className="text-gray-500">No documents uploaded yet.</p>
      ) : (
        <ul className="space-y-4">
          {docs.map((doc) => (
            <li
              key={doc.id}
              className="border p-4 rounded shadow flex justify-between items-center"
            >
              {/* File info */}
              <div>
                <p className="font-semibold">{doc.filename}</p>
                <p className="text-sm">
                  Status:{" "}
                  <span
                    className={
                      doc.status === "READY"
                        ? "text-green-600"
                        : doc.status === "PROCESSING"
                        ? "text-yellow-600"
                        : "text-red-600"
                    }
                  >
                    {doc.status}
                  </span>
                </p>
              </div>

              {/* Action button */}
              {doc.status === "READY" ? (
                <a
                  href={`/documents/${doc.id}`}
                  className="px-3 py-2 bg-green-600 text-white rounded hover:bg-green-700"
                >
                  Open
                </a>
              ) : doc.status === "PROCESSING" ? (
                <span className="px-3 py-2 bg-yellow-200 text-yellow-800 rounded">
                  Processing...
                </span>
              ) : (
                <span className="px-3 py-2 bg-red-200 text-red-800 rounded">
                  Failed
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}