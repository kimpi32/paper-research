import Link from "next/link";
import { fields } from "@/lib/fields";
import { fieldColorStyles } from "@/lib/colors";
import { paperSummaries } from "@/lib/paper-summaries";
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
  const summary = paperSummaries[paperId];

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <Breadcrumbs
        items={[
          { label: field.titleKo, href: `/${fieldId}` },
          { label: String(paper.year), href: `/${fieldId}/${paper.year}` },
          { label: paper.title.length > 50 ? paper.title.substring(0, 50) + "..." : paper.title },
        ]}
      />

      <div className={`border-t-4 ${cs.topBorder} rounded-t-lg pt-4 mb-4`}>
        <Link
          href={`/${fieldId}`}
          className={`inline-block text-xs font-medium px-2.5 py-1 rounded-full ${cs.badge} hover:opacity-80 transition-opacity mb-4`}
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

      {summary ? (
        <div className="prose prose-gray max-w-none space-y-6">
          {/* 한줄 요약 */}
          <div className="bg-gray-50 rounded-lg px-5 py-4 border border-gray-100">
            <p className="text-base font-medium text-gray-800 leading-relaxed">{summary.tldr}</p>
          </div>

          {/* 배경 */}
          <section>
            <h2 className="text-lg font-bold text-gray-900 mb-2">배경</h2>
            <p className="text-gray-700 leading-relaxed whitespace-pre-line">{summary.background}</p>
          </section>

          {/* 핵심 아이디어 */}
          <section>
            <h2 className="text-lg font-bold text-gray-900 mb-2">핵심 아이디어</h2>
            <div className="bg-blue-50 border border-blue-100 rounded-lg px-5 py-4">
              <p className="text-gray-800 leading-relaxed whitespace-pre-line">{summary.keyIdea}</p>
            </div>
          </section>

          {/* 방법론 */}
          {summary.method && (
            <section>
              <h2 className="text-lg font-bold text-gray-900 mb-2">방법론</h2>
              <p className="text-gray-700 leading-relaxed whitespace-pre-line">{summary.method}</p>
            </section>
          )}

          {/* 실험 결과 */}
          {summary.results && (
            <section>
              <h2 className="text-lg font-bold text-gray-900 mb-2">주요 결과</h2>
              <p className="text-gray-700 leading-relaxed whitespace-pre-line">{summary.results}</p>
            </section>
          )}

          {/* 임팩트 */}
          <section>
            <h2 className="text-lg font-bold text-gray-900 mb-2">임팩트</h2>
            <div className="bg-amber-50 border border-amber-100 rounded-lg px-5 py-4">
              <p className="text-gray-800 leading-relaxed whitespace-pre-line">{summary.impact}</p>
            </div>
          </section>

          {/* 관련 Foundation 논문 */}
          {summary.relatedFoundations && summary.relatedFoundations.length > 0 && (
            <section className="pt-4 border-t border-gray-200">
              <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3">관련 Foundation 논문</h3>
              <div className="flex flex-wrap gap-2">
                {summary.relatedFoundations.map((fId) => (
                  <Link
                    key={fId}
                    href={`/foundations/${fId}`}
                    className="text-sm text-blue-600 hover:text-blue-800 bg-blue-50 px-3 py-1.5 rounded-lg hover:bg-blue-100 transition-colors"
                  >
                    {fId}
                  </Link>
                ))}
              </div>
            </section>
          )}

          {/* 관련 논문 */}
          {summary.relatedPapers && summary.relatedPapers.length > 0 && (
            <section className="pt-4 border-t border-gray-200">
              <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3">관련 논문</h3>
              <ul className="space-y-1">
                {summary.relatedPapers.map((rp) => (
                  <li key={rp.id}>
                    <Link
                      href={`/${rp.fieldId}`}
                      className="text-sm text-gray-700 hover:text-blue-600"
                    >
                      <span className="text-xs text-gray-400 mr-1">
                        {rp.relation === "prior" ? "선행" : rp.relation === "successor" ? "후속" : "관련"}
                      </span>
                      {rp.title}
                    </Link>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </div>
      ) : (
        <div className="prose prose-gray max-w-none">
          <p className="text-gray-500 italic">이 논문의 상세 요약이 곧 추가될 예정입니다.</p>
        </div>
      )}

      {/* Links */}
      <div className="mt-8 pt-6 border-t border-gray-200 flex flex-wrap gap-4">
        {paper.arxivUrl && (
          <a
            href={paper.arxivUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-blue-600 hover:text-blue-800 underline underline-offset-2"
          >
            원문 보기 &rarr;
          </a>
        )}
        <a
          href={`https://scholar.google.com/scholar?q=${encodeURIComponent(paper.title)}`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-sm text-gray-500 hover:text-gray-700 underline underline-offset-2"
        >
          Google Scholar &rarr;
        </a>
      </div>
    </div>
  );
}
