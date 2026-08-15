"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ArrowUpRight, Sparkles } from "lucide-react";

export default function ChatWidget() {
  const pathname = usePathname();

  if (pathname === "/hagent" || pathname?.startsWith("/hagent/")) return null;

  return (
    <Link
      href="/hagent"
      aria-label="Mở HAgent workspace"
      className="fixed bottom-5 right-5 z-[60] inline-flex h-12 items-center gap-2 rounded-full border border-border bg-foreground px-4 text-sm font-semibold text-background shadow-lg transition-transform hover:-translate-y-0.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 motion-reduce:transform-none motion-reduce:transition-none"
    >
      <Sparkles className="size-4 text-amber-400" aria-hidden="true" />
      <span>HAgent</span>
      <ArrowUpRight className="size-4" aria-hidden="true" />
    </Link>
  );
}
