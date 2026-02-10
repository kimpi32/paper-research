# TODO - 우선순위별 작업 목록

## P0 — 바로 해야 할 것

### 누락된 Foundation 논문 보충
메타데이터는 있지만 MDX 요약이 없는 논문 5편:
- [ ] **LeNet** (1998) — CNN의 시초, 문서 인식
- [ ] **VAE** (2013) — 변분 오토인코더, 생성 모델의 한 축
- [ ] **Seq2Seq** (2014) — 인코더-디코더 패러다임의 시작
- [ ] **GPT-1** (2018) — 생성적 사전학습의 첫 성공
- [ ] **GPT** 시리즈 정리 (GPT-1 → GPT-2 → GPT-3 → GPT-4 흐름)

### 분야별 핵심 논문 추가 (빈 분야 채우기)
현재 15개 분야 전부 "논문 준비 중" 상태. 분야당 최소 3~5편씩 추가:
- [ ] **NLP** — Word2Vec, ELMo, T5, ChatGPT/InstructGPT
- [ ] **CV** — YOLO, U-Net, Mask R-CNN, DETR, SAM
- [ ] **LLM** — GPT-4, LLaMA, Chinchilla, Chain-of-Thought, RLHF
- [ ] **생성 모델** — StyleGAN, Stable Diffusion/LDM, DALL-E 2, Flow Matching
- [ ] **강화학습** — DQN, AlphaGo, PPO, MuZero

## P1 — 구조 개선

### 연도별 라우팅
현재 `/[fieldId]` 한 단계인데, 계획대로 3단계 위계 구현:
- [ ] `/[fieldId]/[year]` — 연도별 논문 목록 페이지
- [ ] `/[fieldId]/[year]/[paperId]` — 분야별 개별 논문 상세
- [ ] 분야 사이드바 (연도 → 논문 트리)

### 논문 페이지 개선
- [ ] 관련 논문(Related Papers) 컴포넌트 — 선행/후속 논문 상호 링크
- [ ] Foundation 논문에서 분야별 논문으로의 연결 (예: Transformer → NLP/LLM 분야)
- [ ] 논문 간 계보도(lineage) 시각화 (선택)

## P2 — UX 개선

### 네비게이션
- [ ] 모바일 헤더 메뉴 (햄버거)
- [ ] 분야 페이지에 사이드바 추가
- [ ] 홈 화면에서 최근 추가된 논문 섹션

### 검색 & 필터
- [ ] 논문 검색 기능 (제목, 저자, 키워드)
- [ ] 학회별 필터 (NeurIPS, ICML, ICLR, CVPR 등)
- [ ] 태그별 필터 (best-paper, oral, spotlight)
- [ ] 연도 범위 필터

### 디자인
- [ ] 다크 모드
- [ ] 분야별 색상이 카드뿐 아니라 상세 페이지에도 반영
- [ ] 반응형 레이아웃 점검 (모바일/태블릿)

## P3 — 콘텐츠 확장

### 나머지 분야 논문 채우기
- [ ] 멀티모달 — CLIP, Flamingo, GPT-4V, LLaVA
- [ ] 그래프 ML — GCN, GAT, GraphSAGE
- [ ] 로보틱스 — RT-2, SayCan, Inner Monologue
- [ ] AI 안전성 — Constitutional AI, RLHF, Interpretability (Anthropic)
- [ ] 최적화 — SGD convergence, AdamW, Lion, Sophia
- [ ] 표현 학습 — SimCLR, BYOL, MAE, DINO
- [ ] AI for Science — AlphaFold, AlphaGeometry
- [ ] 경량화 — LoRA, QLoRA, Distillation, Pruning
- [ ] 월드 모델 — Sora, Genie, World Models (Ha & Schmidhuber)
- [ ] 음성 — Whisper, WaveNet, Tacotron, VALL-E

### 학회별 정리 (2024~2025)
- [ ] NeurIPS 2024 주요 논문 (oral/spotlight)
- [ ] ICML 2024 주요 논문
- [ ] ICLR 2025 주요 논문
- [ ] CVPR 2024 주요 논문

## P4 — 인프라 & 자동화

- [ ] 배포 자동화 (GitHub Actions → build → scp → 서버 반영)
- [ ] OpenReview API 연동 — ICLR/NeurIPS/ICML accepted paper 목록 자동 수집
- [ ] Semantic Scholar API — 인용 수 자동 업데이트
- [ ] 한/영 전환 (i18n)
- [ ] SEO 메타 태그 (og:image, description 등)
- [ ] RSS 피드
