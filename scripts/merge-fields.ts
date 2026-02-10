/**
 * Merge script: Adds new papers from fields-new-*.ts into fields.ts
 * Run with: npx tsx scripts/merge-fields.ts
 */
import { fields } from "../lib/fields";
import { newPapers as nlpLlm } from "../lib/fields-new-nlp-llm";
import { newPapers as cvGen } from "../lib/fields-new-cv-gen";
import { newPapers as rlRobotWorld } from "../lib/fields-new-rl-robot-world";
import { newPapers as multiAudioRepr } from "../lib/fields-new-multi-audio-repr";
import { newPapers as safetyOptEffGraphSci } from "../lib/fields-new-safety-opt-eff-graph-sci";
import type { Paper, YearGroup } from "../lib/types";

// Combine all new papers by field
const allNewPapers: Record<string, Paper[]> = {};
for (const source of [nlpLlm, cvGen, rlRobotWorld, multiAudioRepr, safetyOptEffGraphSci]) {
  for (const [fieldId, papers] of Object.entries(source)) {
    if (!allNewPapers[fieldId]) allNewPapers[fieldId] = [];
    allNewPapers[fieldId].push(...papers);
  }
}

// Merge new papers into fields
for (const field of fields) {
  const newPapers = allNewPapers[field.id];
  if (!newPapers || newPapers.length === 0) continue;

  // Get existing paper IDs to avoid duplicates
  const existingIds = new Set<string>();
  for (const yg of field.years) {
    for (const p of yg.papers) {
      existingIds.add(p.id);
    }
  }

  // Group new papers by year
  const byYear: Record<number, Paper[]> = {};
  for (const paper of newPapers) {
    if (existingIds.has(paper.id)) {
      console.log(`  SKIP duplicate: ${paper.id} already in ${field.id}`);
      continue;
    }
    if (!byYear[paper.year]) byYear[paper.year] = [];
    byYear[paper.year].push(paper);
  }

  // Merge into existing year groups or create new ones
  for (const [yearStr, papers] of Object.entries(byYear)) {
    const year = parseInt(yearStr);
    const existingYg = field.years.find(yg => yg.year === year);
    if (existingYg) {
      existingYg.papers.push(...papers);
    } else {
      field.years.push({ year, papers });
    }
  }

  // Sort year groups
  field.years.sort((a, b) => a.year - b.year);

  console.log(`${field.id}: ${newPapers.length} new papers merged`);
}

// Generate output
function paperToString(p: Paper, indent: string): string {
  const parts: string[] = [];
  parts.push(`${indent}  id: ${JSON.stringify(p.id)}`);
  parts.push(`${indent}  title: ${JSON.stringify(p.title)}`);
  if (p.titleKo) parts.push(`${indent}  titleKo: ${JSON.stringify(p.titleKo)}`);
  parts.push(`${indent}  authors: ${JSON.stringify(p.authors)}`);
  parts.push(`${indent}  year: ${p.year}`);
  parts.push(`${indent}  venue: ${JSON.stringify(p.venue)}`);
  parts.push(`${indent}  venueType: ${JSON.stringify(p.venueType)}`);
  if (p.award) parts.push(`${indent}  award: ${JSON.stringify(p.award)}`);
  if (p.arxivUrl) parts.push(`${indent}  arxivUrl: ${JSON.stringify(p.arxivUrl)}`);
  if (p.conferenceUrl) parts.push(`${indent}  conferenceUrl: ${JSON.stringify(p.conferenceUrl)}`);
  parts.push(`${indent}  tags: ${JSON.stringify(p.tags)}`);
  parts.push(`${indent}  status: ${JSON.stringify(p.status)}`);
  if (p.citations) parts.push(`${indent}  citations: ${JSON.stringify(p.citations)}`);
  return `${indent}{\n${parts.join(",\n")},\n${indent}}`;
}

let output = `import { Field } from "./types";\n\nexport const fields: Field[] = [\n`;

for (const field of fields) {
  output += `  {\n`;
  output += `    id: ${JSON.stringify(field.id)},\n`;
  output += `    titleKo: ${JSON.stringify(field.titleKo)},\n`;
  output += `    titleEn: ${JSON.stringify(field.titleEn)},\n`;
  output += `    descriptionKo: ${JSON.stringify(field.descriptionKo)},\n`;
  output += `    descriptionEn: ${JSON.stringify(field.descriptionEn)},\n`;
  output += `    color: ${JSON.stringify(field.color)},\n`;
  output += `    years: [\n`;
  for (const yg of field.years) {
    output += `      {\n`;
    output += `        year: ${yg.year},\n`;
    output += `        papers: [\n`;
    for (const p of yg.papers) {
      output += paperToString(p, "          ") + ",\n";
    }
    output += `        ],\n`;
    output += `      },\n`;
  }
  output += `    ],\n`;
  output += `  },\n`;
}
output += `];\n`;

// Write to file
import { writeFileSync } from "fs";
writeFileSync("lib/fields.ts", output);
console.log("\n✓ Written merged fields.ts");

// Count papers
let totalPapers = 0;
for (const field of fields) {
  for (const yg of field.years) {
    totalPapers += yg.papers.length;
  }
}
console.log(`Total papers in fields.ts: ${totalPapers}`);
