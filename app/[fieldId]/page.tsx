import Link from "next/link";
import { fields } from "@/lib/fields";
import { fieldColorStyles } from "@/lib/colors";
import { Breadcrumbs } from "@/components/layout/Breadcrumbs";
import { PaperCard } from "@/components/content/PaperCard";
import { notFound } from "next/navigation";

export function generateStaticParams() {
  return fields.map((field) => ({ fieldId: field.id }));
}

export default async function FieldPage({ params }: { params: Promise<{ fieldId: string }> }) {
  const { fieldId } = await params;
  const field = fields.find((f) => f.id === fieldId);
  if (!field) return notFound();

  const totalPapers = field.years.reduce((sum, y) => sum + y.papers.length, 0);
  const cs = fieldColorStyles[field.color] || fieldColorStyles.blue;
  const topBorder = cs.topBorder;
  const accentText = cs.accent;

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <Breadcrumbs items={[{ label: field.titleKo }]} />

      {/* Colored accent header */}
      <div className={`border-t-4 ${topBorder} rounded-t-lg pt-6 mb-8`}>
        <h1 className="text-3xl font-bold text-gray-900 mb-1">{field.titleKo}</h1>
        <p className={`text-sm ${accentText} font-medium mb-2`}>{field.titleEn}</p>
        <p className="text-gray-600 mb-2">{field.descriptionKo}</p>
        <p className="text-xs text-gray-400">{totalPapers}개 논문</p>
      </div>

      {totalPapers === 0 ? (
        <div className="text-center py-20">
          <p className="text-gray-400 text-lg mb-2">논문 준비 중</p>
          <p className="text-gray-300 text-sm">이 분야의 주요 논문들이 곧 추가될 예정입니다.</p>
        </div>
      ) : (
        field.years
          .sort((a, b) => b.year - a.year)
          .map((yearGroup) => (
            <section key={yearGroup.year} id={`year-${yearGroup.year}`} className="mb-10">
              <h2 className="text-lg font-semibold text-gray-700 mb-4 border-b border-gray-100 pb-2">
                <Link href={`/${fieldId}/${yearGroup.year}`} className="hover:text-blue-600 transition-colors">
                  {yearGroup.year}
                </Link>
                <span className="text-sm font-normal text-gray-400 ml-2">{yearGroup.papers.length}편</span>
              </h2>
              <div className="grid sm:grid-cols-2 gap-4">
                {yearGroup.papers.map((paper) => (
                  <PaperCard
                    key={paper.id}
                    paper={paper}
                    href={`/${fieldId}/${paper.year}/${paper.id}`}
                  />
                ))}
              </div>
            </section>
          ))
      )}

      <div className="mt-12 pt-8 border-t border-gray-100">
        <Link href="/" className="text-sm text-gray-400 hover:text-gray-600">
          &larr; 전체 분야 목록으로
        </Link>
      </div>
    </div>
  );
}
