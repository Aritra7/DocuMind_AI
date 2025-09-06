const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function fetchDocuments() {
  const res = await fetch(`${API_BASE_URL}/documents`);
  if (!res.ok) {
    throw new Error("Failed to fetch documents");
  }
  return res.json();
}

export async function uploadDocument(formData: FormData) {
  const res = await fetch(`${API_BASE_URL}/documents`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    throw new Error("Upload failed");
  }
  return res.json();
}