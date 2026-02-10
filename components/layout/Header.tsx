"use client";

import Link from "next/link";
import { useState } from "react";
import { fields } from "@/lib/fields";
import { SearchDialog } from "@/components/search/SearchDialog";

export function Header() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [mobileFieldsOpen, setMobileFieldsOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link href="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
            <span className="text-2xl">&#x1F4D1;</span>
            <div>
              <h1 className="text-lg font-bold text-gray-900 leading-tight">
                AI Paper Research
              </h1>
              <p className="text-xs text-gray-500 -mt-0.5">AI 논문 조사 및 정리</p>
            </div>
          </Link>

          {/* Desktop nav */}
          <nav className="hidden md:flex items-center gap-6">
            <SearchDialog />
            <Link
              href="/foundations"
              className="text-sm font-semibold text-gray-800 hover:text-amber-600 transition-colors"
            >
              Foundations
            </Link>
            <div className="relative">
              <button
                onClick={() => setMenuOpen(!menuOpen)}
                className="text-sm font-medium text-gray-600 hover:text-gray-900 transition-colors flex items-center gap-1"
              >
                Fields
                <svg className={`w-3.5 h-3.5 transition-transform ${menuOpen ? "rotate-180" : ""}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              {menuOpen && (
                <>
                  <div className="fixed inset-0" onClick={() => setMenuOpen(false)} />
                  <div className="absolute top-full right-0 mt-2 w-64 bg-white border border-gray-200 rounded-xl shadow-lg py-2 max-h-96 overflow-y-auto">
                    {fields.map((field) => (
                      <Link
                        key={field.id}
                        href={`/${field.id}`}
                        onClick={() => setMenuOpen(false)}
                        className="block px-4 py-2 text-sm text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                      >
                        <span className="font-medium">{field.titleKo}</span>
                        <span className="text-gray-400 ml-2 text-xs">{field.titleEn}</span>
                      </Link>
                    ))}
                  </div>
                </>
              )}
            </div>
          </nav>

          {/* Mobile hamburger button */}
          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="md:hidden p-2 rounded-lg text-gray-600 hover:bg-gray-100 transition-colors"
            aria-label="메뉴 열기"
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              {mobileOpen ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              )}
            </svg>
          </button>
        </div>
      </div>

      {/* Mobile menu panel */}
      {mobileOpen && (
        <div className="md:hidden border-t border-gray-200 bg-white">
          <nav className="max-w-7xl mx-auto px-4 py-3 space-y-1">
            <Link
              href="/foundations"
              onClick={() => setMobileOpen(false)}
              className="block px-3 py-2.5 rounded-lg text-sm font-semibold text-gray-800 hover:bg-gray-50 transition-colors"
            >
              Foundations
            </Link>
            <div>
              <button
                onClick={() => setMobileFieldsOpen(!mobileFieldsOpen)}
                className="w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-sm font-medium text-gray-600 hover:bg-gray-50 transition-colors"
              >
                <span>Fields</span>
                <svg
                  className={`w-4 h-4 text-gray-400 transition-transform ${mobileFieldsOpen ? "rotate-180" : ""}`}
                  fill="none" viewBox="0 0 24 24" stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              {mobileFieldsOpen && (
                <div className="mt-1 ml-3 border-l-2 border-gray-100 pl-3 space-y-0.5 max-h-72 overflow-y-auto">
                  {fields.map((field) => (
                    <Link
                      key={field.id}
                      href={`/${field.id}`}
                      onClick={() => { setMobileOpen(false); setMobileFieldsOpen(false); }}
                      className="block px-3 py-2 rounded-lg text-sm text-gray-600 hover:bg-gray-50 hover:text-gray-900 transition-colors"
                    >
                      <span className="font-medium">{field.titleKo}</span>
                      <span className="text-gray-400 ml-2 text-xs">{field.titleEn}</span>
                    </Link>
                  ))}
                </div>
              )}
            </div>
          </nav>
        </div>
      )}
    </header>
  );
}
