import { fields } from "@/lib/fields";
import { Breadcrumbs } from "@/components/layout/Breadcrumbs";
import { notFound } from "next/navigation";

export function generateStaticParams() {
  return fields.map((field) => ({ fieldId: field.id }));
}

export default async function FieldPage({ params }: { params: Promise<{ fieldId: string }> }) {
  const { fieldId } = await params;
  const field = fields.find((f) => f.id === fieldId);
  if (!field) return notFound();

  const totalPapers = field.years.reduce((sum, y) => sum + y.papers.length, 0);

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <Breadcrumbs items={[{ label: field.titleKo }]} />
      <h1 className="text-3xl font-bold text-gray-900 mb-1">{field.titleKo}</h1>
      <p className="text-sm text-gray-400 mb-4">{field.titleEn}</p>
      <p className="text-gray-600 mb-10">{field.descriptionKo}</p>

      {totalPapers === 0 ? (
        <div className="text-center py-20">
          <div className="text-5xl mb-4 opacity-30">&#x1F4DD;</div>
          <p className="text-gray-400 text-lg mb-2">논문 준비 중</p>
          <p className="text-gray-300 text-sm">
            이 분야의 주요 논문들이 곧 추가될 예정입니다.
          </p>
        </div>
      ) : (
        field.years
          .sort((a, b) => b.year - a.year)
          .map((yearGroup) => (
            <section key={yearGroup.year} className="mb-10">
              <h2 className="text-lg font-semibold text-gray-700 mb-4 border-b border-gray-100 pb-2">
                {yearGroup.year}
              </h2>
              <div className="grid sm:grid-cols-2 gap-4">
                {yearGroup.papers.map((paper) => (
                  <div
                    key={paper.id}
                    className="border border-gray-200 rounded-xl p-5"
                  >
                    <p className="font-semibold text-gray-900 text-sm">
                      {paper.title}
                    </p>
                    <p className="text-xs text-gray-400 mt-1">
                      {paper.authors.slice(0, 3).join(", ")}
                      {paper.authors.length > 3 && " et al."} ({paper.year})
                    </p>
                  </div>
                ))}
              </div>
            </section>
          ))
      )}
    </div>
  );
}
