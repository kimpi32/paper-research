import Link from "next/link";
import { fields } from "@/lib/fields";
import { fieldColorStyles } from "@/lib/colors";
import { Breadcrumbs } from "@/components/layout/Breadcrumbs";
import { PaperMeta } from "@/components/content/PaperMeta";
import { notFound } from "next/navigation";

export function generateStaticParams() {
  const params: { fieldId: string; year: string; paperId: string }[] = [];
  for (const field of fields) {
    for (const yg of field.years) {
      for (const paper of yg.papers) {
        params.push({ fieldId: field.id, year: String(paper.year), paperId: paper.id });
      }
    }
  }
  return params;
}

export default async function FieldPaperPage({
  params,
}: {
  params: Promise<{ fieldId: string; year: string; paperId: string }>;
}) {
  const { fieldId, paperId } = await params;
  const field = fields.find((f) => f.id === fieldId);
  if (!field) return notFound();

  let paper = null;
  for (const yg of field.years) {
    const found = yg.papers.find((p) => p.id === paperId);
    if (found) { paper = found; break; }
  }
  if (!paper) return notFound();

  const cs = fieldColorStyles[field.color] || fieldColorStyles.blue;
  const topBorder = cs.topBorder;
  const badgeStyle = cs.badge;

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <Breadcrumbs
        items={[
          { label: field.titleKo, href: `/${fieldId}` },
          { label: String(paper.year), href: `/${fieldId}` },
          { label: paper.title.length > 50 ? paper.title.substring(0, 50) + "..." : paper.title },
        ]}
      />

      {/* Field color accent badge */}
      <div className={`border-t-4 ${topBorder} rounded-t-lg pt-4 mb-4`}>
        <Link
          href={`/${fieldId}`}
          className={`inline-block text-xs font-medium px-2.5 py-1 rounded-full ${badgeStyle} hover:opacity-80 transition-opacity mb-4`}
        >
          {field.titleKo}
        </Link>
      </div>

      <PaperMeta
        title={paper.title}
        titleKo={paper.titleKo}
        authors={paper.authors}
        year={paper.year}
        venue={paper.venue}
        venueType={paper.venueType}
        arxivUrl={paper.arxivUrl}
        conferenceUrl={paper.conferenceUrl}
        award={paper.award}
        citations={paper.citations}
      />
      <div className="prose prose-gray max-w-none">
        <p className="text-gray-500 italic">이 논문의 상세 요약이 곧 추가될 예정입니다.</p>
        {paper.arxivUrl && (
          <p className="mt-4">
            <a
              href={paper.arxivUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:text-blue-800 underline underline-offset-2"
            >
              원문 보기 &rarr;
            </a>
          </p>
        )}
      </div>
    </div>
  );
}
