import Link from "next/link";

interface RelatedPaper {
  title: string;
  href: string;
  relation: "prior" | "successor" | "related";
}

interface Props {
  papers: RelatedPaper[];
}

const relationLabel: Record<string, string> = {
  prior: "선행 연구",
  successor: "후속 연구",
  related: "관련 논문",
};

const relationColor: Record<string, string> = {
  prior: "text-blue-600",
  successor: "text-green-600",
  related: "text-gray-600",
};

export function RelatedPapers({ papers }: Props) {
  if (papers.length === 0) return null;

  const grouped = papers.reduce(
    (acc, p) => {
      if (!acc[p.relation]) acc[p.relation] = [];
      acc[p.relation].push(p);
      return acc;
    },
    {} as Record<string, RelatedPaper[]>,
  );

  return (
    <div className="mt-8 pt-6 border-t border-gray-200">
      <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-4">
        관련 논문
      </h3>
      {(["prior", "successor", "related"] as const).map((rel) =>
        grouped[rel] ? (
          <div key={rel} className="mb-3">
            <span className={`text-xs font-medium ${relationColor[rel]}`}>
              {relationLabel[rel]}
            </span>
            <ul className="mt-1 space-y-1">
              {grouped[rel].map((p) => (
                <li key={p.href}>
                  <Link
                    href={p.href}
                    className="text-sm text-gray-700 hover:text-blue-600 hover:underline"
                  >
                    {p.title}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        ) : null,
      )}
    </div>
  );
}
