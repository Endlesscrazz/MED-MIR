import type { Metadata } from 'next';
import './globals.css';
import { Header } from '@/components/Header';

export const metadata: Metadata = {
  title: 'Med-MIR | Medical Image Retrieval',
  description: 'A privacy-preserving, local-first medical image retrieval system using client-side AI inference.',
  keywords: ['medical imaging', 'image retrieval', 'AI', 'CLIP', 'chest x-ray', 'radiology'],
  authors: [{ name: 'Med-MIR Research Team' }],
  openGraph: {
    title: 'Med-MIR | Medical Image Retrieval',
    description: 'Search medical images using natural language with client-side AI.',
    type: 'website',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background font-sans antialiased">
        <div className="relative flex min-h-screen flex-col">
          <Header />
          <main className="flex-1">{children}</main>
          <footer className="border-t py-6 md:py-0">
            <div className="container flex flex-col items-center justify-between gap-4 md:h-16 md:flex-row">
              <p className="text-sm text-muted-foreground">
                Med-MIR — Privacy-preserving medical image retrieval
              </p>
              <p className="text-sm text-muted-foreground">
                All inference runs locally in your browser
              </p>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}
