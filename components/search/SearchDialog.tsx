"use client";

import { useState, useEffect, useRef, useMemo } from "react";
import Link from "next/link";
import { fields } from "@/lib/fields";
import { foundationPapers } from "@/lib/foundations";
import { Paper } from "@/lib/types";
import { VenueTag } from "@/components/ui/VenueTag";
import { AwardBadge } from "@/components/ui/AwardBadge";

interface SearchItem {
  paper: Paper;
  href: string;
  section: string;
}

function buildIndex(): SearchItem[] {
  const items: SearchItem[] = [];
  for (const p of foundationPapers) {
    items.push({ paper: p, href: `/foundations/${p.id}`, section: "Foundations" });
  }
  for (const field of fields) {
    for (const yg of field.years) {
      for (const p of yg.papers) {
        items.push({
          paper: p,
          href: `/${field.id}/${p.year}/${p.id}`,
          section: field.titleKo,
        });
      }
    }
  }
  return items;
}

export function SearchDialog() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [venueFilter, setVenueFilter] = useState<string | null>(null);
  const [awardFilter, setAwardFilter] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const allItems = useMemo(buildIndex, []);

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setOpen((v) => !v);
      }
      if (e.key === "Escape") setOpen(false);
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(() => {
    if (open) {
      setTimeout(() => inputRef.current?.focus(), 50);
    } else {
      setQuery("");
      setVenueFilter(null);
      setAwardFilter(false);
    }
  }, [open]);

  const results = useMemo(() => {
    let items = allItems;

    if (venueFilter) {
      items = items.filter((i) => i.paper.venueType === venueFilter);
    }
    if (awardFilter) {
      items = items.filter((i) => i.paper.award);
    }

    if (!query.trim()) return items.slice(0, 20);

    const q = query.toLowerCase();
    return items
      .filter((item) => {
        const p = item.paper;
        return (
          p.title.toLowerCase().includes(q) ||
          (p.titleKo && p.titleKo.toLowerCase().includes(q)) ||
          p.authors.some((a) => a.toLowerCase().includes(q)) ||
          p.venue.toLowerCase().includes(q) ||
          item.section.toLowerCase().includes(q)
        );
      })
      .slice(0, 20);
  }, [query, venueFilter, awardFilter, allItems]);

  const venues = ["neurips", "icml", "iclr", "cvpr", "iccv", "acl", "nature"];

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="flex items-center gap-2 px-3 py-1.5 text-sm text-gray-400 bg-gray-50 border border-gray-200 rounded-lg hover:border-gray-300 hover:text-gray-500 transition-colors"
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <span className="hidden sm:inline">논문 검색</span>
        <kbd className="hidden sm:inline-flex items-center gap-0.5 px-1.5 py-0.5 text-[10px] text-gray-400 bg-white border border-gray-200 rounded font-mono">
          <span className="text-xs">⌘</span>K
        </kbd>
      </button>
    );
  }

  return (
    <>
      <div className="fixed inset-0 z-50 bg-black/30 backdrop-blur-sm" onClick={() => setOpen(false)} />
      <div className="fixed inset-x-0 top-16 z-50 mx-auto max-w-2xl px-4">
        <div className="bg-white rounded-xl shadow-2xl border border-gray-200 overflow-hidden">
          {/* Search input */}
          <div className="flex items-center px-4 border-b border-gray-100">
            <svg className="w-5 h-5 text-gray-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="논문 제목, 저자, 학회 검색..."
              className="flex-1 px-3 py-3 text-sm text-gray-900 bg-transparent outline-none placeholder:text-gray-400"
            />
            <button
              onClick={() => setOpen(false)}
              className="text-xs text-gray-400 bg-gray-100 rounded px-1.5 py-0.5 hover:bg-gray-200"
            >
              ESC
            </button>
          </div>

          {/* Filters */}
          <div className="flex flex-wrap items-center gap-1.5 px-4 py-2 border-b border-gray-50 bg-gray-50/50">
            <span className="text-[10px] text-gray-400 uppercase tracking-wider mr-1">학회</span>
            {venues.map((v) => (
              <button
                key={v}
                onClick={() => setVenueFilter(venueFilter === v ? null : v)}
                className={`px-2 py-0.5 text-xs rounded-full border transition-colors ${
                  venueFilter === v
                    ? "bg-blue-50 border-blue-200 text-blue-700"
                    : "bg-white border-gray-200 text-gray-500 hover:border-gray-300"
                }`}
              >
                {v.toUpperCase()}
              </button>
            ))}
            <span className="mx-1 text-gray-200">|</span>
            <button
              onClick={() => setAwardFilter(!awardFilter)}
              className={`px-2 py-0.5 text-xs rounded-full border transition-colors ${
                awardFilter
                  ? "bg-amber-50 border-amber-200 text-amber-700"
                  : "bg-white border-gray-200 text-gray-500 hover:border-gray-300"
              }`}
            >
              Award
            </button>
          </div>

          {/* Results */}
          <div className="max-h-80 overflow-y-auto">
            {results.length === 0 ? (
              <div className="px-4 py-8 text-center text-sm text-gray-400">
                검색 결과가 없습니다
              </div>
            ) : (
              <ul>
                {results.map((item) => (
                  <li key={item.href}>
                    <Link
                      href={item.href}
                      onClick={() => setOpen(false)}
                      className="flex flex-col px-4 py-3 hover:bg-gray-50 transition-colors border-b border-gray-50 last:border-b-0"
                    >
                      <div className="flex items-center gap-2 mb-0.5">
                        <span className="text-[10px] text-gray-400 bg-gray-100 rounded px-1.5 py-0.5">
                          {item.section}
                        </span>
                        <VenueTag venue={item.paper.venue} venueType={item.paper.venueType} />
                        {item.paper.award && <AwardBadge award={item.paper.award} />}
                      </div>
                      <span className="text-sm font-medium text-gray-900 leading-snug">
                        {item.paper.title}
                      </span>
                      <span className="text-xs text-gray-400 mt-0.5">
                        {item.paper.authors.slice(0, 3).join(", ")}
                        {item.paper.authors.length > 3 && " et al."} ({item.paper.year})
                      </span>
                    </Link>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* Footer */}
          <div className="px-4 py-2 border-t border-gray-100 bg-gray-50/50 flex items-center justify-between">
            <span className="text-[10px] text-gray-400">
              {results.length}개 결과
            </span>
            <span className="text-[10px] text-gray-400">
              ↑↓ 이동 · Enter 선택 · ESC 닫기
            </span>
          </div>
        </div>
      </div>
    </>
  );
}
