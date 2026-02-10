import Link from "next/link";
import { Field } from "@/lib/types";
import { fieldColorStyles } from "@/lib/colors";

export function FieldCard({ field }: { field: Field }) {
  const style = fieldColorStyles[field.color] || fieldColorStyles.blue;
  const totalPapers = field.years.reduce((sum, y) => sum + y.papers.length, 0);

  return (
    <Link
      href={`/${field.id}`}
      className={`group block border-2 ${style.border} ${style.bg} rounded-2xl p-6 transition-all ${style.hover} hover:shadow-lg`}
    >
      <div className={`text-xs font-medium ${style.accent} mb-1`}>
        {field.titleEn}
      </div>
      <h2 className="text-xl font-bold text-gray-900 mb-2 group-hover:translate-x-1 transition-transform">
        {field.titleKo}
      </h2>
      <p className="text-sm text-gray-500 mb-3 leading-relaxed">
        {field.descriptionKo}
      </p>
      <div className="text-xs text-gray-400">
        {totalPapers > 0 ? `${totalPapers}개 논문` : "준비 중"}
      </div>
    </Link>
  );
}
