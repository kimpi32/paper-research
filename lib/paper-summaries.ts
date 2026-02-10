import { nlpLlmSummaries } from "./summaries-nlp-llm";
import { cvGenSummaries } from "./summaries-cv-gen";
import { rlMultiSummaries } from "./summaries-rl-multi";
import { restSummaries } from "./summaries-rest";
import { newNlpLlmSummaries } from "./summaries-new-nlp-llm";
import { newCvGenSummaries } from "./summaries-new-cv-gen";
import { newRlRobotWorldSummaries } from "./summaries-new-rl-robot-world";
import { newMultiAudioReprSummaries } from "./summaries-new-multi-audio-repr";
import { newSafetyOptSummaries } from "./summaries-new-safety-opt";
import { newEffGraphSummaries } from "./summaries-new-eff-graph";
import { newSciSummaries } from "./summaries-new-sci";

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
  ...newNlpLlmSummaries,
  ...newCvGenSummaries,
  ...newRlRobotWorldSummaries,
  ...newMultiAudioReprSummaries,
  ...newSafetyOptSummaries,
  ...newEffGraphSummaries,
  ...newSciSummaries,
};
