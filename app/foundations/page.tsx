import { foundationPapers } from "@/lib/foundations";
import { PaperCard } from "@/components/content/PaperCard";
import { Breadcrumbs } from "@/components/layout/Breadcrumbs";

const eraLabels: Record<string, string> = {
  "~1990s": "~ 1990s",
  "2010s-early": "2010s 초반",
  "2010s-mid": "2010s 중반",
  "2010s-late": "2010s 후반",
  "2020s": "2020s",
};

const eras = ["~1990s", "2010s-early", "2010s-mid", "2010s-late", "2020s"];

export default function FoundationsPage() {
  return (
    <div>
      <Breadcrumbs items={[{ label: "Foundations" }]} />
      <h1 className="text-3xl font-bold text-gray-900 mb-2">Foundations</h1>
      <p className="text-gray-500 mb-10">
        AI의 역사를 바꾼 랜드마크 논문들. 시대순으로 정리합니다.
      </p>
      {eras.map((era) => {
        const papers = foundationPapers.filter((p) => p.era === era);
        if (papers.length === 0) return null;
        return (
          <section key={era} className="mb-10">
            <h2 className="text-lg font-semibold text-gray-700 mb-4 border-b border-gray-100 pb-2">
              {eraLabels[era]}
            </h2>
            <div className="grid sm:grid-cols-2 gap-4">
              {papers.map((paper) => (
                <PaperCard
                  key={paper.id}
                  paper={paper}
                  href={`/foundations/${paper.id}`}
                />
              ))}
            </div>
          </section>
        );
      })}
    </div>
  );
}
