# AI Paper Research - 기획 문서

## 1. 프로젝트 개요

AI 분야의 주요 논문들을 조사·요약·정리하여 웹으로 제공하는 사이트.
각 분야별, 연도별로 논문을 정리하고 학회 태그(oral, spotlight, best paper 등)를 달아 관리한다.

## 2. 기술 스택

| 항목 | 기술 |
|------|------|
| 프레임워크 | Next.js 15 (App Router) |
| 언어 | TypeScript 5 |
| 스타일링 | Tailwind CSS 4 |
| 콘텐츠 | MDX (@next/mdx) |
| 수식 렌더링 | remark-math + rehype-katex |
| 폰트 | Noto Sans KR (Google Fonts) |
| 빌드 | Static Export (`output: "export"`) |
| Base Path | `/research` |

## 3. 콘텐츠 구조

### 3.1 전체 구조

```
AI Paper Research
├── Foundations                  ← 역사적 랜드마크 논문 (시대를 바꾼 논문들)
└── Fields (분야별)             ← 분야 → 연도 → 논문
    ├── NLP
    │   ├── 2024
    │   │   ├── paper-a  (oral)
    │   │   └── paper-b  (best paper)
    │   ├── 2023
    │   └── ...
    ├── Computer Vision
    └── ...
```

### 3.2 Foundations (역사적 논문)

AI의 흐름을 바꾼 랜드마크 논문들. 분야 구분 없이 시대순으로 정리.

| 연대 | 대표 논문 |
|------|-----------|
| ~1990s | Perceptron (1958), Backpropagation (1986), Universal Approximation (1989), LeNet (1998), LSTM (1997) |
| 2010s 초 | AlexNet (2012), VAE (2013), GAN (2014), Adam (2014), Seq2Seq (2014), Dropout (2014) |
| 2010s 중 | Batch Norm (2015), ResNet (2015), Attention Mechanism (2015) |
| 2010s 후 | Attention Is All You Need (2017), BERT (2018), GPT (2018), GPT-2 (2019) |
| 2020s | GPT-3 (2020), ViT (2020), DDPM (2020), Scaling Laws (2020), CLIP (2021), DALL-E (2021), InstructGPT (2022), GPT-4 (2023) |

### 3.3 AI 분야 분류

| ID | 분야 | 색상 | 다루는 논문 주제 |
|----|------|------|-----------------|
| `nlp` | 자연어처리 (NLP) | violet | 언어모델, 번역, QA, 요약, 토크나이저 |
| `cv` | 컴퓨터 비전 (CV) | orange | 분류, 검출, 분할, 3D, 비디오 |
| `generative` | 생성 모델 | cyan | GAN, VAE, Diffusion, Flow, 이미지/오디오 생성 |
| `rl` | 강화학습 (RL) | rose | MDP, 정책 경사, Actor-Critic, MARL, 게임 |
| `llm` | 대규모 언어모델 (LLM) | amber | 스케일링, RLHF, 프롬프팅, RAG, 에이전트, 추론 |
| `multimodal` | 멀티모달 | teal | 비전-언어, 오디오-텍스트, 통합 모델 |
| `graph` | 그래프 ML | indigo | GNN, Knowledge Graph, 분자 그래프 |
| `robotics` | 로보틱스 | red | 조작, 내비게이션, Sim2Real, 체화 에이전트 |
| `safety` | AI 안전성·정렬 | fuchsia | 정렬, 해석 가능성, 레드팀, 거버넌스 |
| `optimization` | 최적화·학습이론 | blue | SGD, Adam, 수렴, 일반화, 손실 함수 |
| `representation` | 표현 학습 | emerald | 자기지도학습, 대조학습, 사전학습, 전이학습 |
| `science` | AI for Science | lime | 단백질 접힘, 신약, 기후, 수학 증명 |
| `efficient` | 경량화·효율화 | sky | 양자화, 프루닝, 증류, NAS, 추론 최적화 |
| `world-models` | 월드 모델 | purple | 비디오 예측, 시뮬레이션, 내부 세계 모델 |
| `audio` | 음성·오디오 | slate | ASR, TTS, 음악 생성, 오디오 이해 |

## 4. 학회 & 태그 시스템

### 4.1 주요 학회

| 학회 | 분야 |
|------|------|
| NeurIPS | ML 전반 |
| ICML | ML 전반 |
| ICLR | 표현 학습 / ML 전반 |
| CVPR | 컴퓨터 비전 |
| ICCV | 컴퓨터 비전 |
| ECCV | 컴퓨터 비전 |
| ACL | 자연어처리 |
| EMNLP | 자연어처리 |
| NAACL | 자연어처리 |
| AAAI | AI 전반 |
| ICRA | 로보틱스 |
| INTERSPEECH | 음성 |

### 4.2 논문 태그

```
수상/발표 태그:  best-paper | outstanding-paper | oral | spotlight | poster
논문 유형 태그:  survey | benchmark | framework | theory
```

### 4.3 학회 데이터 수집

학회 논문 목록은 공개 소스에서 참조 가능:
- **OpenReview** (ICLR, NeurIPS, ICML) — 공개 API 존재, accepted paper 목록 + 리뷰 열람 가능
- **학회 공식 사이트** — accepted paper list 공개됨
- **arXiv** — 대부분의 논문이 사전 공개
- **Semantic Scholar / DBLP** — 메타데이터 API 제공

> 자동 크롤링보다는 수동 큐레이션 + API 참조 혼합 방식.
> 모든 논문을 긁어오는 게 아니라 분야별 핵심 논문을 선별하여 정리.

## 5. 페이지 구조 (3단계 위계)

```
/research                              ← 홈 (Foundations 하이라이트 + 분야 카드)
/research/foundations                  ← Foundations 전체 (시대순)
/research/foundations/[paperId]        ← 개별 Foundation 논문
/research/[fieldId]                    ← 분야 개요 (연도별 논문 목록)
/research/[fieldId]/[year]             ← 해당 분야의 특정 연도 논문들
/research/[fieldId]/[year]/[paperId]   ← 개별 논문 상세
```

## 6. 논문 페이지 구성

각 논문 MDX 페이지에 포함되는 내용:

### 필수 항목
1. **메타 정보** — 제목, 저자, 연도, 학회, 태그 (oral/spotlight/best paper)
2. **논문 링크** — arXiv, 학회 공식 링크 (conference proceedings)
3. **한줄 요약** — 이 논문이 뭔지 1~2문장
4. **논문 설명** — 문제 정의, 핵심 방법론, 주요 결과 요약
5. **임팩트** — 후속 연구에 미친 영향, 인용 수, 현재 관점에서의 평가

### 선택 항목 (논문에 따라)
- **핵심 수식** — KaTeX로 렌더링
- **관련 논문** — 선행/후속 논문 링크

## 7. 데이터 구조 (TypeScript)

```typescript
type ContentStatus = "skeleton" | "draft" | "complete"

type AwardTag = "best-paper" | "outstanding-paper" | "oral" | "spotlight" | "poster"
type PaperTag = "survey" | "benchmark" | "framework" | "theory"

interface Paper {
  id: string
  title: string
  titleKo?: string
  authors: string[]
  year: number
  venue: string              // "NeurIPS 2024", "ICML 2023", "arXiv" 등
  venueType: string          // "neurips" | "icml" | "iclr" | "cvpr" | "arxiv" ...
  arxivUrl?: string
  conferenceUrl?: string     // 학회 공식 proceedings 링크
  award?: AwardTag
  tags: PaperTag[]
  status: ContentStatus
}

interface YearGroup {
  year: number
  papers: Paper[]
}

interface Field {
  id: string
  titleKo: string
  titleEn: string
  descriptionKo: string
  descriptionEn: string
  color: string
  years: YearGroup[]
}

interface FoundationPaper extends Paper {
  era: string               // "~1990s" | "2010s-early" | "2010s-mid" | "2010s-late" | "2020s"
}
```

## 8. 컴포넌트 설계

### Layout
- `Header.tsx` — 상단 네비 (Foundations + 분야 목록)
- `Sidebar.tsx` — 분야 내 연도/논문 트리
- `Breadcrumbs.tsx` — 경로

### Content
- `PaperMeta.tsx` — 메타 카드 (제목, 저자, 연도, 학회, 링크, 태그)
- `PaperSummary.tsx` — 한줄 요약 박스
- `KeyIdea.tsx` — 핵심 아이디어 박스
- `Formula.tsx` — 수식 박스
- `Impact.tsx` — 임팩트 박스
- `RelatedPapers.tsx` — 관련 논문 링크
- `PaperCard.tsx` — 논문 카드 (목록용, 태그 표시)
- `FieldCard.tsx` — 분야 카드 (홈 화면용)
- `YearSection.tsx` — 연도별 섹션 구분

### UI
- `StatusBadge.tsx` — skeleton / draft / complete
- `AwardBadge.tsx` — best-paper / oral / spotlight 배지
- `VenueTag.tsx` — 학회 태그 (NeurIPS, ICML 등)
- `Tag.tsx` — 일반 태그

## 9. 디렉토리 구조

```
/research
├── app/
│   ├── layout.tsx
│   ├── page.tsx                             ← 홈
│   ├── globals.css
│   ├── foundations/
│   │   ├── page.tsx                         ← Foundations 목록
│   │   └── [paperId]/
│   │       └── page.mdx                     ← 개별 논문
│   └── [fieldId]/
│       ├── layout.tsx
│       ├── page.tsx                          ← 분야 개요 (연도별)
│       └── [year]/
│           ├── page.tsx                      ← 연도별 논문 목록
│           └── [paperId]/
│               └── page.mdx                  ← 개별 논문
├── components/
│   ├── content/
│   ├── layout/
│   └── ui/
├── lib/
│   ├── types.ts
│   ├── fields.ts                            ← 분야·연도·논문 데이터
│   └── foundations.ts                       ← Foundation 논문 데이터
├── public/
├── mdx-components.tsx
├── next.config.mjs
├── tsconfig.json
├── postcss.config.mjs
└── package.json
```

## 10. 구현 순서

### Phase 1 — 프로젝트 세팅
- [ ] Next.js 15 + TypeScript + Tailwind CSS 4 초기화
- [ ] MDX + KaTeX 설정
- [ ] next.config.mjs (basePath: "/research", output: "export")
- [ ] 기본 레이아웃, 글로벌 스타일

### Phase 2 — 데이터 & 타입
- [ ] `lib/types.ts` — Paper, Field, YearGroup 등
- [ ] `lib/fields.ts` — 15개 분야 + 연도별 논문 메타
- [ ] `lib/foundations.ts` — Foundation 논문 데이터

### Phase 3 — 레이아웃 & 페이지
- [ ] Header, Sidebar, Breadcrumbs
- [ ] 홈 페이지 (Foundations 하이라이트 + 분야 카드)
- [ ] Foundations 목록 & 상세
- [ ] 분야 → 연도 → 논문 3단계 페이지

### Phase 4 — 콘텐츠 컴포넌트 & MDX
- [ ] PaperMeta, PaperSummary, KeyIdea, Formula, Impact
- [ ] PaperCard, FieldCard, YearSection
- [ ] AwardBadge, VenueTag, StatusBadge, Tag
- [ ] mdx-components.tsx

### Phase 5 — 논문 요약 작성
- [ ] Foundation 논문 요약 (우선)
- [ ] 분야별 주요 논문 요약 (최근 연도부터)
- [ ] 지속적 추가
