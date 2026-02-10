import { AwardBadge } from "@/components/ui/AwardBadge";
import { VenueTag } from "@/components/ui/VenueTag";
import { AwardTag, VenueType } from "@/lib/types";

interface Props {
  title: string;
  titleKo?: string;
  authors: string[];
  year: number;
  venue: string;
  venueType: VenueType;
  arxivUrl?: string;
  conferenceUrl?: string;
  award?: AwardTag;
  citations?: string;
}

export function PaperMeta({
  title, titleKo, authors, year, venue, venueType,
  arxivUrl, conferenceUrl, award, citations,
}: Props) {
  return (
    <div className="mb-8 pb-6 border-b border-gray-200">
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <VenueTag venue={venue} venueType={venueType} />
        {award && <AwardBadge award={award} />}
        {citations && (
          <span className="text-xs text-gray-400">
            Citations: {citations}
          </span>
        )}
      </div>
      <h1 className="text-2xl font-bold text-gray-900 mb-1 leading-tight">
        {title}
      </h1>
      {titleKo && (
        <p className="text-base text-gray-500 mb-3">{titleKo}</p>
      )}
      <p className="text-sm text-gray-500 mb-3">
        {authors.join(", ")} ({year})
      </p>
      <div className="flex flex-wrap gap-3">
        {arxivUrl && (
          <a
            href={arxivUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-sm text-blue-600 hover:text-blue-800 underline underline-offset-2"
          >
            arXiv
          </a>
        )}
        {conferenceUrl && (
          <a
            href={conferenceUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-sm text-blue-600 hover:text-blue-800 underline underline-offset-2"
          >
            Paper Link
          </a>
        )}
      </div>
    </div>
  );
}
