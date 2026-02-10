import { ReactNode } from "react";

export function KeyIdea({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <div className="my-6 border-l-4 border-blue-500 bg-blue-50/50 rounded-r-lg p-5">
      <div className="font-semibold text-blue-800 mb-2 text-sm uppercase tracking-wide">
        <span className="inline-block bg-blue-100 text-blue-700 px-2 py-0.5 rounded text-xs mr-2">
          핵심 아이디어
        </span>
        {title}
      </div>
      <div className="text-gray-800 leading-relaxed">{children}</div>
    </div>
  );
}
