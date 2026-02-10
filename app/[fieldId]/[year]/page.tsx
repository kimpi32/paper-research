import Link from "next/link";
import { fields } from "@/lib/fields";
import { Breadcrumbs } from "@/components/layout/Breadcrumbs";
import { PaperCard } from "@/components/content/PaperCard";
import { notFound } from "next/navigation";

export function generateStaticParams() {
  const params: { fieldId: string; year: string }[] = [];
  for (const field of fields) {
    for (const yg of field.years) {
      params.push({ fieldId: field.id, year: String(yg.year) });
    }
  }
  return params;
}

export default async function FieldYearPage({
  params,
}: {
  params: Promise<{ fieldId: string; year: string }>;
}) {
  const { fieldId, year } = await params;
  const field = fields.find((f) => f.id === fieldId);
  if (!field) return notFound();

  const yearNum = parseInt(year, 10);
  const yearGroup = field.years.find((yg) => yg.year === yearNum);
  if (!yearGroup) return notFound();

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <Breadcrumbs
        items={[
          { label: field.titleKo, href: `/${fieldId}` },
          { label: String(yearNum) },
        ]}
      />
      <h1 className="text-3xl font-bold text-gray-900 mb-1">
        {field.titleKo} — {yearNum}
      </h1>
      <p className="text-sm text-gray-400 mb-8">
        {yearGroup.papers.length}편의 논문
      </p>

      <div className="grid sm:grid-cols-2 gap-4">
        {yearGroup.papers.map((paper) => (
          <PaperCard
            key={paper.id}
            paper={paper}
            href={`/${fieldId}/${paper.year}/${paper.id}`}
          />
        ))}
      </div>

      <div className="mt-12 pt-8 border-t border-gray-100 flex justify-between">
        <Link
          href={`/${fieldId}`}
          className="text-sm text-gray-400 hover:text-gray-600"
        >
          &larr; {field.titleKo} 전체
        </Link>
      </div>
    </div>
  );
}
