import { ReactNode } from "react";

export function Impact({ children }: { children: ReactNode }) {
  return (
    <div className="my-6 border-l-4 border-amber-500 bg-amber-50/50 rounded-r-lg p-5">
      <div className="font-semibold text-amber-800 mb-2 text-sm uppercase tracking-wide">
        <span className="inline-block bg-amber-100 text-amber-700 px-2 py-0.5 rounded text-xs mr-2">
          임팩트
        </span>
      </div>
      <div className="text-gray-800 leading-relaxed">{children}</div>
    </div>
  );
}
