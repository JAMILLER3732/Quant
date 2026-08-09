import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Quant Analytics Platform",
  description:
    "Upload portfolio/market data, select a quantitative-finance methodology, and get Python-calculated, mathematically correct charts and statistics.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-slate-950 text-slate-100">
        <header className="border-b border-slate-800 bg-slate-950/95 backdrop-blur sticky top-0 z-40">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
            <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight">
              <span className="inline-block h-2 w-2 rounded-full bg-emerald-400" />
              <span>Quant Analytics Platform</span>
            </Link>
            <nav className="flex items-center gap-6 text-sm text-slate-300">
              <Link href="/" className="hover:text-white transition-colors">
                Dashboard
              </Link>
              <Link href="/lab" className="hover:text-white transition-colors">
                Quant Lab
              </Link>
              <Link href="/blog" className="hover:text-white transition-colors">
                Insights
              </Link>
              <Link href="/admin" className="hover:text-white transition-colors text-slate-500">
                Admin
              </Link>
            </nav>
          </div>
        </header>
        <main className="flex-1">{children}</main>
        <footer className="border-t border-slate-800 py-6 text-center text-xs text-slate-500">
          Every chart and statistic on this platform is calculated from your uploaded data by the Python quant
          engine — nothing is fabricated or hard-coded.
        </footer>
      </body>
    </html>
  );
}
