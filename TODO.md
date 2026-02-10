# TODO - 우선순위별 작업 목록

## P0 — 바로 해야 할 것

### 누락된 Foundation 논문 보충
메타데이터는 있지만 MDX 요약이 없는 논문 5편:
- [x] **LeNet** (1998) — CNN의 시초, 문서 인식
- [x] **VAE** (2013) — 변분 오토인코더, 생성 모델의 한 축
- [x] **Seq2Seq** (2014) — 인코더-디코더 패러다임의 시작
- [x] **GPT-1** (2018) — 생성적 사전학습의 첫 성공
- [ ] **GPT** 시리즈 정리 (GPT-1 → GPT-2 → GPT-3 → GPT-4 흐름)

### 분야별 핵심 논문 추가 (빈 분야 채우기)
15개 분야 전부 핵심 논문 데이터 추가 완료 (분야당 2~5편):
- [x] **NLP** — Word2Vec, ELMo, T5, XLNet, InstructGPT
- [x] **CV** — YOLO, U-Net, Mask R-CNN, DETR, SAM
- [x] **LLM** — GPT-4, LLaMA, Chinchilla, Chain-of-Thought, RAG
- [x] **생성 모델** — WGAN, StyleGAN, LDM/Stable Diffusion, DALL-E 2, Flow Matching
- [x] **강화학습** — DQN, AlphaGo, PPO, MuZero
- [x] **멀티모달** — CLIP, Flamingo, LLaVA
- [x] **그래프 ML** — GCN, GAT, GraphSAGE
- [x] **로보틱스** — SayCan, RT-2
- [x] **AI 안전성** — Constitutional AI, RLHF, Interpretability
- [x] **최적화** — AdamW, Lottery Ticket, Lion
- [x] **표현 학습** — SimCLR, BYOL, DINO, MAE
- [x] **AI for Science** — AlphaFold2, AlphaGeometry
- [x] **경량화** — Distillation, LoRA, QLoRA
- [x] **월드 모델** — World Models (Ha), Sora, Genie
- [x] **음성** — WaveNet, Whisper, VALL-E

## P1 — 구조 개선

### 연도별 라우팅
3단계 위계 구현 완료:
- [x] `/[fieldId]/[year]` — 연도별 논문 목록 페이지
- [x] `/[fieldId]/[year]/[paperId]` — 분야별 개별 논문 상세
- [x] 분야 사이드바 (연도 → 논문 트리)

### 논문 페이지 개선
- [x] 관련 논문(Related Papers) 컴포넌트 구현 (미연결 — 데이터 필요)
- [ ] Foundation 논문에서 분야별 논문으로의 연결 (예: Transformer → NLP/LLM 분야)
- [ ] 논문 간 계보도(lineage) 시각화 (선택)

## P2 — UX 개선

### 네비게이션
- [x] 모바일 헤더 메뉴 (햄버거)
- [x] 분야 페이지에 사이드바 추가 (FieldSidebar)
- [x] 홈 화면에서 최근 추가된 논문 섹션

### 검색 & 필터
- [x] 논문 검색 기능 (제목, 저자, 학회 검색 — ⌘K)
- [x] 학회별 필터 (NeurIPS, ICML, ICLR, CVPR 등)
- [x] Award 필터 (best-paper, oral, spotlight)
- [ ] 연도 범위 필터

### 디자인
- [ ] 다크 모드
- [x] 분야별 색상이 카드뿐 아니라 상세 페이지에도 반영
- [x] 반응형 레이아웃 (모바일 사이드바 + 햄버거 메뉴)

## P3 — 콘텐츠 확장

### 분야별 논문 데이터
15개 분야 전부 메타데이터 완료. 상세 MDX 요약은 미작성:
- [x] 메타데이터 (53편) — fields.ts에 등록 완료
- [ ] 각 논문별 상세 MDX 요약 작성 (현재 placeholder)

### 학회별 정리 (2024~2025)
- [ ] NeurIPS 2024 주요 논문 (oral/spotlight)
- [ ] ICML 2024 주요 논문
- [ ] ICLR 2025 주요 논문
- [ ] CVPR 2024 주요 논문

## P4 — 인프라 & 자동화

- [ ] 배포 자동화 (GitHub Actions → build → scp → 서버 반영)
- [ ] 서버 재배포 (EC2 서버 접속 불가 상태 — IP 확인 필요)
- [ ] OpenReview API 연동 — ICLR/NeurIPS/ICML accepted paper 목록 자동 수집
- [ ] Semantic Scholar API — 인용 수 자동 업데이트
- [ ] 한/영 전환 (i18n)
- [ ] SEO 메타 태그 (og:image, description 등)
- [ ] RSS 피드
