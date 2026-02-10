import { VenueType } from "@/lib/types";

const venueColors: Record<string, string> = {
  neurips: "bg-purple-100 text-purple-700",
  icml: "bg-blue-100 text-blue-700",
  iclr: "bg-teal-100 text-teal-700",
  cvpr: "bg-orange-100 text-orange-700",
  iccv: "bg-orange-50 text-orange-600",
  eccv: "bg-orange-50 text-orange-600",
  acl: "bg-emerald-100 text-emerald-700",
  emnlp: "bg-emerald-50 text-emerald-600",
  naacl: "bg-emerald-50 text-emerald-600",
  aaai: "bg-indigo-100 text-indigo-700",
  nature: "bg-red-100 text-red-700",
  science: "bg-red-50 text-red-600",
  jmlr: "bg-slate-100 text-slate-700",
  arxiv: "bg-gray-100 text-gray-600",
};

export function VenueTag({ venue, venueType }: { venue: string; venueType: VenueType }) {
  const color = venueColors[venueType] || "bg-gray-100 text-gray-600";
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${color}`}>
      {venue}
    </span>
  );
}
