import { ContentStatus } from "@/lib/types";

const config: Record<ContentStatus, { label: string; dot: string; text: string }> = {
  skeleton: { label: "골격", dot: "bg-gray-300", text: "text-gray-400" },
  draft: { label: "초안", dot: "bg-amber-400", text: "text-amber-600" },
  complete: { label: "완성", dot: "bg-emerald-400", text: "text-emerald-600" },
};

export function StatusBadge({ status }: { status: ContentStatus }) {
  const c = config[status];
  return (
    <span className={`inline-flex items-center gap-1 text-xs ${c.text}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${c.dot}`} />
      {c.label}
    </span>
  );
}
