export type ContentStatus = "skeleton" | "draft" | "complete";

export type AwardTag = "best-paper" | "outstanding-paper" | "oral" | "spotlight" | "poster";

export type PaperTag = "survey" | "benchmark" | "framework" | "theory";

export type VenueType =
  | "neurips" | "icml" | "iclr" | "cvpr" | "iccv" | "eccv"
  | "acl" | "emnlp" | "naacl" | "aaai" | "icra" | "interspeech"
  | "jmlr" | "nature" | "science" | "arxiv" | "other";

export interface Paper {
  id: string;
  title: string;
  titleKo?: string;
  authors: string[];
  year: number;
  venue: string;
  venueType: VenueType;
  arxivUrl?: string;
  conferenceUrl?: string;
  award?: AwardTag;
  tags: PaperTag[];
  status: ContentStatus;
  citations?: string;
}

export interface YearGroup {
  year: number;
  papers: Paper[];
}

export interface Field {
  id: string;
  titleKo: string;
  titleEn: string;
  descriptionKo: string;
  descriptionEn: string;
  color: string;
  years: YearGroup[];
}

export interface FoundationPaper extends Paper {
  era: string;
}
