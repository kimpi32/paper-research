"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { foundationPapers } from "@/lib/foundations";
import { fields } from "@/lib/fields";
import { fieldColorStyles } from "@/lib/colors";

const eras = ["~1990s", "2010s-early", "2010s-mid", "2010s-late", "2020s"];
const eraLabels: Record<string, string> = {
  "~1990s": "~1990s",
  "2010s-early": "2010s 초",
  "2010s-mid": "2010s 중",
  "2010s-late": "2010s 후",
  "2020s": "2020s",
};

export function FoundationsSidebar() {
  const pathname = usePathname();
  const [openEras, setOpenEras] = useState<Set<string>>(() => {
    const initial = new Set<string>();
    const paper = foundationPapers.find((p) => pathname.includes(`/${p.id}`));
    if (paper) initial.add(paper.era);
    return initial;
  });
  const [mobileOpen, setMobileOpen] = useState(false);

  const toggleEra = (era: string) => {
    setOpenEras((prev) => {
      const next = new Set(prev);
      if (next.has(era)) next.delete(era);
      else next.add(era);
      return next;
    });
  };

  const sidebarContent = (
    <nav className="py-4 px-3 space-y-1 overflow-y-auto h-full">
      <Link
        href="/foundations"
        className="block px-3 py-2 rounded-lg text-sm font-bold mb-3 text-amber-600 hover:bg-gray-100 transition-colors"
      >
        Foundations
        <span className="block text-xs font-normal text-gray-400">
          역사적 랜드마크 논문
        </span>
      </Link>
      {eras.map((era) => {
        const papers = foundationPapers.filter((p) => p.era === era);
        if (papers.length === 0) return null;
        const isOpen = openEras.has(era);
        return (
          <div key={era}>
            <button
              onClick={() => toggleEra(era)}
              className="w-full flex items-center justify-between px-3 py-2 rounded-lg text-sm text-gray-600 hover:bg-gray-50 hover:text-gray-900 transition-colors"
            >
              <span>{eraLabels[era]}</span>
              <svg
                className={`w-4 h-4 text-gray-400 transition-transform ${isOpen ? "rotate-90" : ""}`}
                fill="none" viewBox="0 0 24 24" stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>
            {isOpen && (
              <div className="ml-4 mt-1 space-y-0.5 border-l-2 border-gray-100 pl-3">
                {papers.map((paper) => (
                  <Link
                    key={paper.id}
                    href={`/foundations/${paper.id}`}
                    onClick={() => setMobileOpen(false)}
                    className={`block px-2 py-1.5 text-xs rounded hover:bg-gray-50 ${
                      pathname.includes(`/${paper.id}`)
                        ? "font-medium text-gray-900 bg-gray-50"
                        : "text-gray-500"
                    }`}
                  >
                    {paper.title.length > 40
                      ? paper.title.substring(0, 40) + "..."
                      : paper.title}
                    <span className="text-gray-300 ml-1">({paper.year})</span>
                  </Link>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </nav>
  );

  return (
    <>
      <button
        onClick={() => setMobileOpen(!mobileOpen)}
        className="lg:hidden fixed bottom-4 right-4 z-50 bg-gray-900 text-white p-3 rounded-full shadow-lg"
        aria-label="Toggle sidebar"
      >
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          {mobileOpen ? (
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          ) : (
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          )}
        </svg>
      </button>
      {mobileOpen && (
        <div className="lg:hidden fixed inset-0 z-40 bg-black/30" onClick={() => setMobileOpen(false)} />
      )}
      <aside
        className={`fixed lg:sticky top-16 z-40 h-[calc(100vh-4rem)] w-64 bg-white border-r border-gray-200 transition-transform lg:translate-x-0 ${
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        {sidebarContent}
      </aside>
    </>
  );
}

export function FieldSidebar({ fieldId }: { fieldId: string }) {
  const pathname = usePathname();
  const field = fields.find((f) => f.id === fieldId);
  if (!field) return null;

  const sortedYears = [...field.years].sort((a, b) => b.year - a.year);
  const cs = fieldColorStyles[field.color] || fieldColorStyles.blue;

  const [openYears, setOpenYears] = useState<Set<number>>(() => {
    const initial = new Set<number>();
    // Auto-open the year whose paper is currently being viewed
    for (const yg of field.years) {
      for (const p of yg.papers) {
        if (pathname.includes(`/${p.id}`)) {
          initial.add(yg.year);
        }
      }
    }
    // If nothing matched (e.g. on the field index page), open all years
    if (initial.size === 0) {
      for (const yg of sortedYears) {
        initial.add(yg.year);
      }
    }
    return initial;
  });
  const [mobileOpen, setMobileOpen] = useState(false);

  const toggleYear = (year: number) => {
    setOpenYears((prev) => {
      const next = new Set(prev);
      if (next.has(year)) next.delete(year);
      else next.add(year);
      return next;
    });
  };

  const sidebarContent = (
    <nav className="py-4 px-3 space-y-1 overflow-y-auto h-full">
      <Link
        href={`/${field.id}`}
        className={`block px-3 py-2 rounded-lg text-sm font-bold mb-3 ${cs.accent} hover:bg-gray-100 transition-colors`}
      >
        {field.titleKo}
        <span className="block text-xs font-normal text-gray-400">
          {field.titleEn}
        </span>
      </Link>
      {sortedYears.map((yg) => {
        if (yg.papers.length === 0) return null;
        const isOpen = openYears.has(yg.year);
        return (
          <div key={yg.year}>
            <button
              onClick={() => toggleYear(yg.year)}
              className="w-full flex items-center justify-between px-3 py-2 rounded-lg text-sm text-gray-600 hover:bg-gray-50 hover:text-gray-900 transition-colors"
            >
              <span>
                {yg.year}
                <span className="text-gray-400 ml-1 text-xs">{yg.papers.length}편</span>
              </span>
              <svg
                className={`w-4 h-4 text-gray-400 transition-transform ${isOpen ? "rotate-90" : ""}`}
                fill="none" viewBox="0 0 24 24" stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>
            {isOpen && (
              <div className={`ml-4 mt-1 space-y-0.5 border-l-2 ${cs.sidebarBorder} pl-3`}>
                {yg.papers.map((paper) => (
                  <Link
                    key={paper.id}
                    href={`/${field.id}/${paper.year}/${paper.id}`}
                    onClick={() => setMobileOpen(false)}
                    className={`block px-2 py-1.5 text-xs rounded hover:bg-gray-50 ${
                      pathname.includes(`/${paper.id}`)
                        ? `font-medium text-gray-900 ${cs.sidebarActiveBg}`
                        : "text-gray-500"
                    }`}
                  >
                    {paper.title.length > 40
                      ? paper.title.substring(0, 40) + "..."
                      : paper.title}
                  </Link>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </nav>
  );

  return (
    <>
      <button
        onClick={() => setMobileOpen(!mobileOpen)}
        className="lg:hidden fixed bottom-4 right-4 z-50 bg-gray-900 text-white p-3 rounded-full shadow-lg"
        aria-label="사이드바 열기"
      >
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          {mobileOpen ? (
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          ) : (
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          )}
        </svg>
      </button>
      {mobileOpen && (
        <div className="lg:hidden fixed inset-0 z-40 bg-black/30" onClick={() => setMobileOpen(false)} />
      )}
      <aside
        className={`fixed lg:sticky top-16 z-40 h-[calc(100vh-4rem)] w-64 bg-white border-r border-gray-200 transition-transform lg:translate-x-0 ${
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        {sidebarContent}
      </aside>
    </>
  );
}
