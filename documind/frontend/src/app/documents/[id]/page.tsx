"use client";

import { useParams } from "next/navigation";
import { useState, useEffect } from "react";
import dynamic from "next/dynamic";

// Dynamically import JUST the react-pdf components
const Document = dynamic(
  () => import("react-pdf").then((mod) => mod.Document),
  { ssr: false }
);
const Page = dynamic(
  () => import("react-pdf").then((mod) => mod.Page),
  { ssr: false }
);

export default function DocumentDetailPage() {
  const params = useParams();
  const docId = params?.id as string;
  const [doc, setDoc] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [numPages, setNumPages] = useState<number | null>(null);
  const [workerReady, setWorkerReady] = useState(false);

  // -----------------------------
  // PDF.js worker setup
  // -----------------------------
  useEffect(() => {
    (async () => {
      try {
        const { pdfjs } = await import("react-pdf");
        // ✅ Serve worker from your public folder
        pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.js";
        setWorkerReady(true);
      } catch (err) {
        console.error("Failed to set pdfjs workerSrc:", err);
      }
    })();
  }, []);

  // -----------------------------
  // Fetch document metadata
  // -----------------------------
  useEffect(() => {
    async function loadDoc() {
      try {
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/documents/${docId}`);
        if (res.ok) setDoc(await res.json());
      } catch (err) {
        console.error("Failed to load document metadata:", err);
      } finally {
        setLoading(false);
      }
    }
    if (docId) loadDoc();
  }, [docId]);

  // -----------------------------
  // Render
  // -----------------------------
  if (!workerReady) return <p>Initializing PDF worker...</p>;

  return (
    <div className="flex h-screen">
      <div className="flex-1 bg-gray-50 overflow-y-auto p-4">
        {loading ? (
          <p>Loading document...</p>
        ) : doc ? (
          <>
            <h2 className="text-lg font-bold mb-2">
              {doc.filename} ({doc.status})
            </h2>
            <Document
              file={`${process.env.NEXT_PUBLIC_API_URL}/files/${docId}`}
              onLoadSuccess={({ numPages }) => setNumPages(numPages)}
              loading={<p>Loading PDF…</p>}
              error={<p className="text-red-600">Failed to load PDF.</p>}
            >
              {numPages &&
                Array.from({ length: numPages }, (_, i) => (
                  <Page
                    key={`page_${i + 1}`}
                    pageNumber={i + 1}
                    renderAnnotationLayer={false}
                    renderTextLayer={false}
                  />
                ))}
            </Document>
          </>
        ) : (
          <p className="text-red-600">Document not found</p>
        )}
      </div>
    </div>
  );
}
