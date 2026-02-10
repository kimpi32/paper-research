import { AwardTag } from "@/lib/types";

const config: Record<AwardTag, { label: string; bg: string; text: string }> = {
  "best-paper": { label: "Best Paper", bg: "bg-yellow-100", text: "text-yellow-800" },
  "outstanding-paper": { label: "Outstanding", bg: "bg-yellow-50", text: "text-yellow-700" },
  oral: { label: "Oral", bg: "bg-red-50", text: "text-red-700" },
  spotlight: { label: "Spotlight", bg: "bg-orange-50", text: "text-orange-700" },
  poster: { label: "Poster", bg: "bg-gray-50", text: "text-gray-600" },
};

export function AwardBadge({ award }: { award: AwardTag }) {
  const c = config[award];
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs font-semibold ${c.bg} ${c.text}`}>
      {c.label}
    </span>
  );
}
