import Link from "next/link";
import { foundationToFieldLinks } from "@/lib/foundation-links";

export function FoundationFieldLinks({ paperId }: { paperId: string }) {
  const links = foundationToFieldLinks[paperId];
  if (!links || links.length === 0) return null;

  return (
    <div className="mt-10 pt-6 border-t border-gray-200">
      <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-4">
        이 논문의 영향을 받은 분야별 논문
      </h3>
      <div className="grid sm:grid-cols-2 gap-2">
        {links.map((link) => (
          <Link
            key={`${link.fieldId}-${link.paperId}`}
            href={`/${link.fieldId}/${link.year}/${link.paperId}`}
            className="flex items-center gap-2 px-3 py-2.5 rounded-lg border border-gray-100 hover:border-gray-300 hover:bg-gray-50 transition-colors group"
          >
            <span className="text-sm text-gray-800 group-hover:text-blue-600 transition-colors">
              {link.title}
            </span>
            <span className="text-xs text-gray-400 ml-auto shrink-0">
              {link.year}
            </span>
          </Link>
        ))}
      </div>
    </div>
  );
}
