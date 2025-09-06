"use client";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-12">
      <h1 className="text-4xl font-bold mb-6">DocuMind AI</h1>
      <p className="text-lg mb-8">Upload, explore, and query your documents!</p>

      <a
        href="/upload"
        className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-md"
      >
        Go to Upload Page →
      </a>
    </main>
  );
}