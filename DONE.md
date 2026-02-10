# DONE - 완료된 작업

## 2025-02-10

### 프로젝트 초기 세팅
- [x] GitHub 저장소 생성 (kimpi32/paper-research, public)
- [x] PLAN.md 기획 문서 작성
- [x] Next.js 15 + TypeScript 5 + Tailwind CSS 4 프로젝트 초기화
- [x] MDX + KaTeX (remark-math + rehype-katex) 설정
- [x] next.config.mjs (basePath: "/paper", output: "export", trailingSlash)
- [x] .gitignore, tsconfig.json, postcss.config.mjs

### 데이터 & 타입 정의
- [x] `lib/types.ts` — Paper, Field, YearGroup, FoundationPaper, AwardTag, VenueType 등
- [x] `lib/fields.ts` — 15개 AI 분야 정의 (색상, 설명 포함)
- [x] `lib/foundations.ts` — 20개 Foundation 논문 메타데이터

### 컴포넌트 구현
- [x] **Layout**: Header (분야 드롭다운), Breadcrumbs, FoundationsSidebar (시대별 트리)
- [x] **Content**: PaperMeta, KeyIdea, Formula, Impact, PaperCard, FieldCard
- [x] **UI**: AwardBadge, VenueTag, StatusBadge
- [x] `mdx-components.tsx` — MDX 스타일 매핑 (h1~h3, p, ul, table, code 등)

### 페이지 구현
- [x] 홈 페이지 — Foundations 하이라이트 6편 + 15개 분야 카드 그리드
- [x] `/foundations` — 시대별 그룹핑 목록 + 사이드바
- [x] `/foundations/[paperId]` — 15편의 상세 MDX 페이지
- [x] `/[fieldId]` — 15개 분야 placeholder 페이지 (generateStaticParams)

### Foundation 논문 요약 작성 (15편)
각 논문마다 한줄 요약, 배경, 핵심 아이디어, 수식(KaTeX), 실험 결과, 임팩트 포함.

| 시대 | 작성 완료 논문 |
|------|---------------|
| ~1990s | Backpropagation (1986), Universal Approximation Theorem (1989), LSTM (1997) |
| 2010s 초 | AlexNet (2012), GAN (2014), Adam (2014), Dropout (2014), Attention Mechanism (2014) |
| 2010s 중 | Batch Normalization (2015), ResNet (2015) |
| 2010s 후 | Transformer (2017), BERT (2018) |
| 2020s | GPT-3 (2020), ViT (2020), DDPM (2020), Scaling Laws (2020) |

### 배포
- [x] Static export 빌드 (9MB, 36페이지)
- [x] EC2 서버 배포 — Apache Alias /paper (메모리 사용량 0MB)
- [x] Apache RewriteEngine 설정 (clean URL + .html fallback)
- [x] 전체 페이지 200 OK 확인
