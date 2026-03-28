import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Finance Q&A UI",
  description: "Minimal UI for Agentic RAG Q&A"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
