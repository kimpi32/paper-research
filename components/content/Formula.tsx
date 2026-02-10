import { ReactNode } from "react";

export function Formula({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <div className="my-6 border-l-4 border-violet-500 bg-violet-50/50 rounded-r-lg p-5">
      <div className="font-semibold text-violet-800 mb-2 text-sm uppercase tracking-wide">
        <span className="inline-block bg-violet-100 text-violet-700 px-2 py-0.5 rounded text-xs mr-2">
          수식
        </span>
        {title}
      </div>
      <div className="text-gray-800 leading-relaxed">{children}</div>
    </div>
  );
}
