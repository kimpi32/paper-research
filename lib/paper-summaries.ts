import { nlpLlmSummaries } from "./summaries-nlp-llm";
import { cvGenSummaries } from "./summaries-cv-gen";
import { rlMultiSummaries } from "./summaries-rl-multi";
import { restSummaries } from "./summaries-rest";

export interface PaperSummary {
  tldr: string;
  background: string;
  keyIdea: string;
  method?: string;
  results?: string;
  impact: string;
  relatedFoundations?: string[];
  relatedPapers?: { id: string; fieldId: string; title: string; relation: "prior" | "successor" | "related" }[];
}

export const paperSummaries: Record<string, PaperSummary> = {
  ...nlpLlmSummaries,
  ...cvGenSummaries,
  ...rlMultiSummaries,
  ...restSummaries,
};
