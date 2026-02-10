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
- [x] `/foundations/[paperId]` — 16편의 상세 MDX 페이지
- [x] `/[fieldId]` — 15개 분야 논문 목록 (연도별 그룹핑)

### Foundation 논문 요약 작성 (16편)
각 논문마다 한줄 요약, 배경, 핵심 아이디어, 수식(KaTeX), 실험 결과, 임팩트 포함.

| 시대 | 작성 완료 논문 |
|------|---------------|
| ~1990s | Backpropagation (1986), Universal Approximation Theorem (1989), LSTM (1997), **LeNet (1998)** |
| 2010s 초 | AlexNet (2012), **VAE (2013)**, GAN (2014), Adam (2014), **Seq2Seq (2014)**, Dropout (2014), Attention Mechanism (2014) |
| 2010s 중 | Batch Normalization (2015), ResNet (2015) |
| 2010s 후 | Transformer (2017), BERT (2018), **GPT-1 (2018)** |
| 2020s | GPT-3 (2020), ViT (2020), DDPM (2020), Scaling Laws (2020) |

### 배포
- [x] Static export 빌드 (9MB, 36페이지)
- [x] EC2 서버 배포 — Apache Alias /paper (메모리 사용량 0MB)
- [x] Apache RewriteEngine 설정 (clean URL + .html fallback)
- [x] 전체 페이지 200 OK 확인

---

## 2026-02-10 (P0~P3 일괄 작업)

### P0: Foundation 논문 보충
- [x] LeNet (1998) MDX 요약 작성 — CNN의 시초, 문서 인식
- [x] VAE (2013) MDX 요약 작성 — 변분 오토인코더, ELBO, 재매개변수화 트릭
- [x] Seq2Seq (2014) MDX 요약 작성 — 인코더-디코더 LSTM 아키텍처
- [x] GPT-1 (2018) MDX 요약 작성 — 생성적 사전학습의 첫 성공

### P0+P3: 15개 분야 논문 데이터 추가
- [x] `lib/fields.ts` — 15개 분야에 총 53편의 논문 메타데이터 등록
- [x] NLP: Word2Vec, ELMo, T5, XLNet, InstructGPT (5편)
- [x] CV: YOLO, U-Net, Mask R-CNN, DETR, SAM (5편)
- [x] 생성 모델: WGAN, StyleGAN, LDM, DALL-E 2, Flow Matching (5편)
- [x] 강화학습: DQN, AlphaGo, PPO, MuZero (4편)
- [x] LLM: Chinchilla, CoT, LLaMA, GPT-4, RAG (5편)
- [x] 멀티모달: CLIP, Flamingo, LLaVA (3편)
- [x] 그래프 ML: GCN, GAT, GraphSAGE (3편)
- [x] 로보틱스: SayCan, RT-2 (2편)
- [x] AI 안전성: Constitutional AI, RLHF, Interpretability (3편)
- [x] 최적화: AdamW, Lottery Ticket, Lion (3편)
- [x] 표현 학습: SimCLR, BYOL, DINO, MAE (4편)
- [x] AI for Science: AlphaFold2, AlphaGeometry (2편)
- [x] 경량화: Distillation, LoRA, QLoRA (3편)
- [x] 월드 모델: World Models, Sora, Genie (3편)
- [x] 음성: WaveNet, Whisper, VALL-E (3편)

### P1: 구조 개선
- [x] `/[fieldId]/[year]` 연도별 논문 목록 페이지 생성
- [x] `/[fieldId]/[year]/[paperId]` 분야별 개별 논문 상세 페이지 생성
- [x] `FieldSidebar` 컴포넌트 — 분야별 연도 → 논문 트리 사이드바 (색상 반영)
- [x] `app/[fieldId]/layout.tsx` — 분야 페이지에 사이드바 레이아웃 적용
- [x] `RelatedPapers` 컴포넌트 구현 (선행/후속/관련 논문 링크)
- [x] 연도 제목 클릭 시 연도 페이지로 링크

### P2: UX 개선
- [x] 모바일 햄버거 메뉴 (Header에 모바일 토글 + Fields 아코디언)
- [x] 모바일 사이드바 (FAB 버튼으로 열림/닫힘)
- [x] 홈 페이지 "최근 추가된 논문" 섹션 (최신 6편)
- [x] `SearchDialog` — ⌘K 글로벌 검색 (논문 제목, 저자, 학회)
- [x] 학회별 필터 (NeurIPS, ICML, ICLR, CVPR, ICCV, ACL, Nature)
- [x] Award 필터 (best-paper, outstanding, oral, spotlight)
- [x] 분야별 색상 상세 페이지 반영 (border-top accent + 분야 배지)
- [x] `lib/colors.ts` — 공유 색상 스타일 맵 (15색상, topBorder/badge/sidebar 등)
- [x] 반응형 레이아웃 (모바일 사이드바 + 데스크톱 sticky sidebar)

### 빌드
- [x] Static export 성공 — 20개 Foundation + 53개 분야 논문 + 43개 연도 페이지
