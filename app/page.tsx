import Link from "next/link";
import { fields } from "@/lib/fields";
import { foundationPapers } from "@/lib/foundations";
import { FieldCard } from "@/components/content/FieldCard";
import { PaperCard } from "@/components/content/PaperCard";

export default function Home() {
  const highlightPapers = foundationPapers.filter((p) =>
    ["transformer", "resnet", "alexnet", "gan", "bert", "gpt3"].includes(p.id)
  );

  return (
    <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
      <div className="text-center mb-16">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          AI Paper Research
        </h1>
        <p className="text-lg text-gray-500 max-w-2xl mx-auto">
          AI 분야의 주요 논문들을 <strong className="text-gray-700">요약</strong>,{" "}
          <strong className="text-gray-700">수식</strong>,{" "}
          <strong className="text-gray-700">임팩트</strong>와 함께 정리합니다.
        </p>
      </div>

      {/* Foundations Highlight */}
      <section className="mb-16">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Foundations</h2>
          <Link
            href="/foundations"
            className="text-sm text-amber-600 hover:text-amber-800 font-medium"
          >
            전체 보기 &rarr;
          </Link>
        </div>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {highlightPapers.map((paper) => (
            <PaperCard
              key={paper.id}
              paper={paper}
              href={`/foundations/${paper.id}`}
            />
          ))}
        </div>
      </section>

      {/* Fields */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Fields</h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {fields.map((field) => (
            <FieldCard key={field.id} field={field} />
          ))}
        </div>
      </section>
    </main>
  );
}
