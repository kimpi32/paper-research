import Link from "next/link";
import { Paper } from "@/lib/types";
import { AwardBadge } from "@/components/ui/AwardBadge";
import { VenueTag } from "@/components/ui/VenueTag";

interface Props {
  paper: Paper;
  href: string;
}

export function PaperCard({ paper, href }: Props) {
  return (
    <Link
      href={href}
      className="group block border border-gray-200 rounded-xl p-5 hover:border-gray-400 hover:shadow-md transition-all"
    >
      <div className="flex flex-wrap items-center gap-2 mb-2">
        <VenueTag venue={paper.venue} venueType={paper.venueType} />
        {paper.award && <AwardBadge award={paper.award} />}
        {paper.citations && (
          <span className="text-xs text-gray-400">{paper.citations}</span>
        )}
      </div>
      <h3 className="text-base font-semibold text-gray-900 mb-1 group-hover:text-blue-600 transition-colors leading-snug">
        {paper.title}
      </h3>
      {paper.titleKo && (
        <p className="text-sm text-gray-500 mb-2">{paper.titleKo}</p>
      )}
      <p className="text-xs text-gray-400">
        {paper.authors.slice(0, 3).join(", ")}
        {paper.authors.length > 3 && " et al."}
        {" "}({paper.year})
      </p>
    </Link>
  );
}
