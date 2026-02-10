import Link from "next/link";
import { Field } from "@/lib/types";

const colorStyles: Record<string, { border: string; bg: string; hover: string; accent: string }> = {
  violet: { border: "border-violet-200", bg: "bg-violet-50/50", hover: "hover:border-violet-400 hover:shadow-violet-100", accent: "text-violet-600" },
  orange: { border: "border-orange-200", bg: "bg-orange-50/50", hover: "hover:border-orange-400 hover:shadow-orange-100", accent: "text-orange-600" },
  cyan: { border: "border-cyan-200", bg: "bg-cyan-50/50", hover: "hover:border-cyan-400 hover:shadow-cyan-100", accent: "text-cyan-600" },
  rose: { border: "border-rose-200", bg: "bg-rose-50/50", hover: "hover:border-rose-400 hover:shadow-rose-100", accent: "text-rose-600" },
  amber: { border: "border-amber-200", bg: "bg-amber-50/50", hover: "hover:border-amber-400 hover:shadow-amber-100", accent: "text-amber-600" },
  teal: { border: "border-teal-200", bg: "bg-teal-50/50", hover: "hover:border-teal-400 hover:shadow-teal-100", accent: "text-teal-600" },
  indigo: { border: "border-indigo-200", bg: "bg-indigo-50/50", hover: "hover:border-indigo-400 hover:shadow-indigo-100", accent: "text-indigo-600" },
  red: { border: "border-red-200", bg: "bg-red-50/50", hover: "hover:border-red-400 hover:shadow-red-100", accent: "text-red-600" },
  fuchsia: { border: "border-fuchsia-200", bg: "bg-fuchsia-50/50", hover: "hover:border-fuchsia-400 hover:shadow-fuchsia-100", accent: "text-fuchsia-600" },
  blue: { border: "border-blue-200", bg: "bg-blue-50/50", hover: "hover:border-blue-400 hover:shadow-blue-100", accent: "text-blue-600" },
  emerald: { border: "border-emerald-200", bg: "bg-emerald-50/50", hover: "hover:border-emerald-400 hover:shadow-emerald-100", accent: "text-emerald-600" },
  lime: { border: "border-lime-200", bg: "bg-lime-50/50", hover: "hover:border-lime-400 hover:shadow-lime-100", accent: "text-lime-600" },
  sky: { border: "border-sky-200", bg: "bg-sky-50/50", hover: "hover:border-sky-400 hover:shadow-sky-100", accent: "text-sky-600" },
  purple: { border: "border-purple-200", bg: "bg-purple-50/50", hover: "hover:border-purple-400 hover:shadow-purple-100", accent: "text-purple-600" },
  slate: { border: "border-slate-200", bg: "bg-slate-50/50", hover: "hover:border-slate-400 hover:shadow-slate-100", accent: "text-slate-600" },
};

export function FieldCard({ field }: { field: Field }) {
  const style = colorStyles[field.color] || colorStyles.blue;
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
