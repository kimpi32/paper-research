import { Field } from "./types";

export const fields: Field[] = [
  {
    id: "nlp",
    titleKo: "자연어처리",
    titleEn: "Natural Language Processing",
    descriptionKo: "언어모델, 번역, QA, 요약, 토크나이저 등 텍스트 이해와 생성에 관한 연구",
    descriptionEn: "Research on text understanding and generation",
    color: "violet",
    years: [
      {
        year: 2013,
        papers: [
          { id: "word2vec", title: "Efficient Estimation of Word Representations in Vector Space", titleKo: "벡터 공간에서의 효율적 단어 표현 추정", authors: ["Tomas Mikolov", "Kai Chen", "Greg Corrado", "Jeffrey Dean"], year: 2013, venue: "ICLR 2013 Workshop", venueType: "iclr", arxivUrl: "https://arxiv.org/abs/1301.3781", tags: [], status: "complete", citations: "40,000+" },
        ],
      },
      {
        year: 2018,
        papers: [
          { id: "elmo", title: "Deep contextualized word representations", titleKo: "심층 문맥화 단어 표현", authors: ["Matthew E. Peters", "Mark Neumann", "Mohit Iyyer", "et al."], year: 2018, venue: "NAACL 2018", venueType: "naacl", award: "best-paper", arxivUrl: "https://arxiv.org/abs/1802.05365", tags: [], status: "complete", citations: "15,000+" },
        ],
      },
      {
        year: 2019,
        papers: [
          { id: "t5", title: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", titleKo: "통합 텍스트-투-텍스트 트랜스포머를 통한 전이학습의 한계 탐구", authors: ["Colin Raffel", "Noam Shazeer", "Adam Roberts", "et al."], year: 2019, venue: "JMLR 2020", venueType: "jmlr", arxivUrl: "https://arxiv.org/abs/1910.10683", tags: [], status: "complete", citations: "18,000+" },
          { id: "xlnet", title: "XLNet: Generalized Autoregressive Pretraining for Language Understanding", titleKo: "XLNet: 언어 이해를 위한 일반화된 자기회귀 사전학습", authors: ["Zhilin Yang", "Zihang Dai", "Yiming Yang", "et al."], year: 2019, venue: "NeurIPS 2019", venueType: "neurips", arxivUrl: "https://arxiv.org/abs/1906.08237", tags: [], status: "complete", citations: "8,000+" },
        ],
      },
      {
        year: 2022,
        papers: [
          { id: "instructgpt", title: "Training language models to follow instructions with human feedback", titleKo: "인간 피드백으로 지시를 따르도록 언어 모델 학습", authors: ["Long Ouyang", "Jeff Wu", "Xu Jiang", "et al."], year: 2022, venue: "NeurIPS 2022", venueType: "neurips", arxivUrl: "https://arxiv.org/abs/2203.02155", tags: [], status: "complete", citations: "8,000+" },
        ],
      },
    ],
  },
  {
    id: "cv",
    titleKo: "컴퓨터 비전",
    titleEn: "Computer Vision",
    descriptionKo: "분류, 검출, 분할, 3D 비전, 비디오 분석 등 시각 정보 처리 연구",
    descriptionEn: "Research on visual information processing",
    color: "orange",
    years: [
      {
        year: 2015,
        papers: [
          { id: "yolo", title: "You Only Look Once: Unified, Real-Time Object Detection", titleKo: "한 번만 보면 된다: 통합 실시간 객체 검출", authors: ["Joseph Redmon", "Santosh Divvala", "Ross Girshick", "Ali Farhadi"], year: 2015, venue: "CVPR 2016", venueType: "cvpr", arxivUrl: "https://arxiv.org/abs/1506.02640", tags: [], status: "complete", citations: "35,000+" },
          { id: "unet", title: "U-Net: Convolutional Networks for Biomedical Image Segmentation", titleKo: "U-Net: 의료 영상 분할을 위한 합성곱 네트워크", authors: ["Olaf Ronneberger", "Philipp Fischer", "Thomas Brox"], year: 2015, venue: "MICCAI 2015", venueType: "other", arxivUrl: "https://arxiv.org/abs/1505.04597", tags: [], status: "complete", citations: "70,000+" },
        ],
      },
      {
        year: 2017,
        papers: [
          { id: "mask-rcnn", title: "Mask R-CNN", titleKo: "Mask R-CNN", authors: ["Kaiming He", "Georgia Gkioxari", "Piotr Dollár", "Ross Girshick"], year: 2017, venue: "ICCV 2017", venueType: "iccv", award: "best-paper", arxivUrl: "https://arxiv.org/abs/1703.06870", tags: [], status: "complete", citations: "20,000+" },
        ],
      },
      {
        year: 2020,
        papers: [
          { id: "detr", title: "End-to-End Object Detection with Transformers", titleKo: "트랜스포머를 이용한 엔드투엔드 객체 검출", authors: ["Nicolas Carion", "Francisco Massa", "Gabriel Synnaeve", "et al."], year: 2020, venue: "ECCV 2020", venueType: "eccv", arxivUrl: "https://arxiv.org/abs/2005.12872", tags: [], status: "complete", citations: "10,000+" },
        ],
      },
      {
        year: 2023,
        papers: [
          { id: "sam", title: "Segment Anything", titleKo: "무엇이든 분할하기", authors: ["Alexander Kirillov", "Eric Mintun", "Nikhila Ravi", "et al."], year: 2023, venue: "ICCV 2023", venueType: "iccv", arxivUrl: "https://arxiv.org/abs/2304.02643", tags: [], status: "complete", citations: "5,000+" },
        ],
      },
    ],
  },
  {
    id: "generative",
    titleKo: "생성 모델",
    titleEn: "Generative Models",
    descriptionKo: "GAN, VAE, Diffusion, Flow 등 데이터 생성에 관한 연구",
    descriptionEn: "Research on data generation models",
    color: "cyan",
    years: [
      {
        year: 2017,
        papers: [
          { id: "wgan", title: "Wasserstein GAN", titleKo: "바서슈타인 GAN", authors: ["Martin Arjovsky", "Soumith Chintala", "Léon Bottou"], year: 2017, venue: "ICML 2017", venueType: "icml", arxivUrl: "https://arxiv.org/abs/1701.07875", tags: ["theory"], status: "complete", citations: "12,000+" },
        ],
      },
      {
        year: 2019,
        papers: [
          { id: "stylegan", title: "A Style-Based Generator Architecture for Generative Adversarial Networks", titleKo: "생성적 적대 신경망을 위한 스타일 기반 생성기 아키텍처", authors: ["Tero Karras", "Samuli Laine", "Timo Aila"], year: 2019, venue: "CVPR 2019", venueType: "cvpr", arxivUrl: "https://arxiv.org/abs/1812.04948", tags: [], status: "complete", citations: "12,000+" },
        ],
      },
      {
        year: 2022,
        papers: [
          { id: "ldm", title: "High-Resolution Image Synthesis with Latent Diffusion Models", titleKo: "잠재 확산 모델을 이용한 고해상도 이미지 합성", authors: ["Robin Rombach", "Andreas Blattmann", "Dominik Lorenz", "et al."], year: 2022, venue: "CVPR 2022", venueType: "cvpr", award: "oral", arxivUrl: "https://arxiv.org/abs/2112.10752", tags: [], status: "complete", citations: "12,000+" },
          { id: "dalle2", title: "Hierarchical Text-Conditional Image Generation with CLIP Latents", titleKo: "CLIP 잠재 변수를 이용한 계층적 텍스트 조건 이미지 생성", authors: ["Aditya Ramesh", "Prafulla Dhariwal", "Alex Nichol", "et al."], year: 2022, venue: "arXiv", venueType: "arxiv", arxivUrl: "https://arxiv.org/abs/2204.06125", tags: [], status: "complete", citations: "5,000+" },
        ],
      },
      {
        year: 2023,
        papers: [
          { id: "flow-matching", title: "Flow Matching for Generative Modeling", titleKo: "생성 모델링을 위한 플로우 매칭", authors: ["Yaron Lipman", "Ricky T. Q. Chen", "Heli Ben-Hamu", "Maximilian Nickel"], year: 2023, venue: "ICLR 2023", venueType: "iclr", arxivUrl: "https://arxiv.org/abs/2210.02747", tags: ["theory"], status: "complete", citations: "1,000+" },
        ],
      },
    ],
  },
  {
    id: "rl",
    titleKo: "강화학습",
    titleEn: "Reinforcement Learning",
    descriptionKo: "MDP, 정책 경사, Actor-Critic, MARL, 게임 AI 등",
    descriptionEn: "Research on learning from interaction and rewards",
    color: "rose",
    years: [
      {
        year: 2013,
        papers: [
          { id: "dqn", title: "Playing Atari with Deep Reinforcement Learning", titleKo: "심층 강화학습으로 Atari 게임하기", authors: ["Volodymyr Mnih", "Kavukcuoglu", "Silver", "et al."], year: 2013, venue: "NeurIPS 2013 Workshop", venueType: "neurips", arxivUrl: "https://arxiv.org/abs/1312.5602", tags: [], status: "complete", citations: "15,000+" },
        ],
      },
      {
        year: 2016,
        papers: [
          { id: "alphago", title: "Mastering the game of Go with deep neural networks and tree search", titleKo: "심층 신경망과 트리 탐색으로 바둑 마스터하기", authors: ["David Silver", "Aja Huang", "Chris J. Maddison", "et al."], year: 2016, venue: "Nature", venueType: "nature", arxivUrl: "https://doi.org/10.1038/nature16961", tags: [], status: "complete", citations: "18,000+" },
        ],
      },
      {
        year: 2017,
        papers: [
          { id: "ppo", title: "Proximal Policy Optimization Algorithms", titleKo: "근접 정책 최적화 알고리즘", authors: ["John Schulman", "Filip Wolski", "Prafulla Dhariwal", "et al."], year: 2017, venue: "arXiv", venueType: "arxiv", arxivUrl: "https://arxiv.org/abs/1707.06347", tags: [], status: "complete", citations: "15,000+" },
        ],
      },
      {
        year: 2019,
        papers: [
          { id: "muzero", title: "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model", titleKo: "학습된 모델로 계획하여 Atari, 바둑, 체스, 장기 마스터하기", authors: ["Julian Schrittwieser", "Ioannis Antonoglou", "Thomas Hubert", "et al."], year: 2019, venue: "Nature 2020", venueType: "nature", arxivUrl: "https://arxiv.org/abs/1911.08265", tags: [], status: "complete", citations: "3,000+" },
        ],
      },
    ],
  },
  {
    id: "llm",
    titleKo: "대규모 언어모델",
    titleEn: "Large Language Models",
    descriptionKo: "스케일링, RLHF, 프롬프팅, RAG, 에이전트, 추론 등",
    descriptionEn: "Research on scaling and capabilities of language models",
    color: "amber",
    years: [
      {
        year: 2022,
        papers: [
          { id: "chinchilla", title: "Training Compute-Optimal Large Language Models", titleKo: "계산 최적 대규모 언어 모델 학습", authors: ["Jordan Hoffmann", "Sebastian Borgeaud", "Arthur Mensch", "et al."], year: 2022, venue: "NeurIPS 2022", venueType: "neurips", arxivUrl: "https://arxiv.org/abs/2203.15556", tags: [], status: "complete", citations: "3,000+" },
          { id: "cot", title: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", titleKo: "연쇄 사고 프롬프팅은 대규모 언어 모델에서 추론을 이끌어낸다", authors: ["Jason Wei", "Xuezhi Wang", "Dale Schuurmans", "et al."], year: 2022, venue: "NeurIPS 2022", venueType: "neurips", award: "outstanding-paper", arxivUrl: "https://arxiv.org/abs/2201.11903", tags: [], status: "complete", citations: "5,000+" },
        ],
      },
      {
        year: 2023,
        papers: [
          { id: "llama", title: "LLaMA: Open and Efficient Foundation Language Models", titleKo: "LLaMA: 개방적이고 효율적인 기초 언어 모델", authors: ["Hugo Touvron", "Thibaut Lavril", "Gautier Izacard", "et al."], year: 2023, venue: "arXiv", venueType: "arxiv", arxivUrl: "https://arxiv.org/abs/2302.13971", tags: [], status: "complete", citations: "8,000+" },
          { id: "gpt4", title: "GPT-4 Technical Report", titleKo: "GPT-4 기술 보고서", authors: ["OpenAI"], year: 2023, venue: "arXiv", venueType: "arxiv", arxivUrl: "https://arxiv.org/abs/2303.08774", tags: [], status: "complete", citations: "5,000+" },
          { id: "rag", title: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", titleKo: "지식 집약적 NLP 태스크를 위한 검색 증강 생성", authors: ["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus", "et al."], year: 2020, venue: "NeurIPS 2020", venueType: "neurips", arxivUrl: "https://arxiv.org/abs/2005.11401", tags: [], status: "complete", citations: "4,000+" },
        ],
      },
    ],
  },
  {
    id: "multimodal",
    titleKo: "멀티모달",
    titleEn: "Multimodal AI",
    descriptionKo: "비전-언어, 오디오-텍스트, 통합 모델 연구",
    descriptionEn: "Research on multi-modal understanding and generation",
    color: "teal",
    years: [
      {
        year: 2021,
        papers: [
          { id: "clip", title: "Learning Transferable Visual Models From Natural Language Supervision", titleKo: "자연어 감독으로 전이 가능한 시각 모델 학습", authors: ["Alec Radford", "Jong Wook Kim", "Chris Hallacy", "et al."], year: 2021, venue: "ICML 2021", venueType: "icml", arxivUrl: "https://arxiv.org/abs/2103.00020", tags: [], status: "complete", citations: "20,000+" },
        ],
      },
      {
        year: 2022,
        papers: [
          { id: "flamingo", title: "Flamingo: a Visual Language Model for Few-Shot Learning", titleKo: "Flamingo: 퓨샷 학습을 위한 시각 언어 모델", authors: ["Jean-Baptiste Alayrac", "Jeff Donahue", "Pauline Luc", "et al."], year: 2022, venue: "NeurIPS 2022", venueType: "neurips", arxivUrl: "https://arxiv.org/abs/2204.14198", tags: [], status: "complete", citations: "3,000+" },
        ],
      },
      {
        year: 2023,
        papers: [
          { id: "llava", title: "Visual Instruction Tuning", titleKo: "시각적 지시 튜닝", authors: ["Haotian Liu", "Chunyuan Li", "Qingyang Wu", "Yong Jae Lee"], year: 2023, venue: "NeurIPS 2023", venueType: "neurips", award: "oral", arxivUrl: "https://arxiv.org/abs/2304.08485", tags: [], status: "complete", citations: "3,000+" },
        ],
      },
    ],
  },
  {
    id: "graph",
    titleKo: "그래프 ML",
    titleEn: "Graph Machine Learning",
    descriptionKo: "GNN, Knowledge Graph, 분자 그래프 등 그래프 구조 학습",
    descriptionEn: "Research on learning from graph-structured data",
    color: "indigo",
    years: [
      {
        year: 2016,
        papers: [
          { id: "gcn", title: "Semi-Supervised Classification with Graph Convolutional Networks", titleKo: "그래프 합성곱 네트워크를 이용한 준지도 분류", authors: ["Thomas N. Kipf", "Max Welling"], year: 2016, venue: "ICLR 2017", venueType: "iclr", arxivUrl: "https://arxiv.org/abs/1609.02907", tags: [], status: "complete", citations: "25,000+" },
        ],
      },
      {
        year: 2017,
        papers: [
          { id: "gat", title: "Graph Attention Networks", titleKo: "그래프 어텐션 네트워크", authors: ["Petar Veličković", "Guillem Cucurull", "Arantxa Casanova", "et al."], year: 2017, venue: "ICLR 2018", venueType: "iclr", arxivUrl: "https://arxiv.org/abs/1710.10903", tags: [], status: "complete", citations: "15,000+" },
          { id: "graphsage", title: "Inductive Representation Learning on Large Graphs", titleKo: "대규모 그래프에서의 귀납적 표현 학습", authors: ["William L. Hamilton", "Rex Ying", "Jure Leskovec"], year: 2017, venue: "NeurIPS 2017", venueType: "neurips", arxivUrl: "https://arxiv.org/abs/1706.02216", tags: [], status: "complete", citations: "10,000+" },
        ],
      },
    ],
  },
  {
    id: "robotics",
    titleKo: "로보틱스",
    titleEn: "Robotics & Embodied AI",
    descriptionKo: "조작, 내비게이션, Sim2Real, 체화 에이전트 연구",
    descriptionEn: "Research on robotic manipulation, navigation, and embodied agents",
    color: "red",
    years: [
      {
        year: 2022,
        papers: [
          { id: "saycan", title: "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances", titleKo: "내가 하는 대로 해, 말하는 대로가 아니라: 로봇 어포던스에서의 언어 접지", authors: ["Michael Ahn", "Anthony Brohan", "Noah Brown", "et al."], year: 2022, venue: "arXiv", venueType: "arxiv", arxivUrl: "https://arxiv.org/abs/2204.01691", tags: [], status: "complete", citations: "1,500+" },
        ],
      },
      {
        year: 2023,
        papers: [
          { id: "rt2", title: "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control", titleKo: "RT-2: 비전-언어-행동 모델이 웹 지식을 로봇 제어로 전이", authors: ["Anthony Brohan", "Noah Brown", "Justice Carbajal", "et al."], year: 2023, venue: "arXiv", venueType: "arxiv", arxivUrl: "https://arxiv.org/abs/2307.15818", tags: [], status: "complete", citations: "500+" },
        ],
      },
    ],
  },
  {
    id: "safety",
    titleKo: "AI 안전성·정렬",
    titleEn: "AI Safety & Alignment",
    descriptionKo: "정렬, 해석 가능성, 레드팀, 거버넌스 연구",
    descriptionEn: "Research on making AI systems safe and aligned",
    color: "fuchsia",
    years: [
      {
        year: 2022,
        papers: [
          { id: "constitutional-ai", title: "Constitutional AI: Harmlessness from AI Feedback", titleKo: "헌법적 AI: AI 피드백을 통한 무해성", authors: ["Yuntao Bai", "Saurav Kadavath", "Sandipan Kundu", "et al."], year: 2022, venue: "arXiv", venueType: "arxiv", arxivUrl: "https://arxiv.org/abs/2212.08073", tags: [], status: "complete", citations: "1,500+" },
          { id: "rlhf", title: "Training a Helpful and Harmless Assistant with RLHF", titleKo: "RLHF로 도움이 되고 무해한 어시스턴트 학습", authors: ["Yuntao Bai", "Andy Jones", "Kamal Ndousse", "et al."], year: 2022, venue: "arXiv", venueType: "arxiv", arxivUrl: "https://arxiv.org/abs/2204.05862", tags: [], status: "complete", citations: "2,000+" },
        ],
      },
      {
        year: 2023,
        papers: [
          { id: "interpretability-circuits", title: "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning", titleKo: "단의성을 향하여: 사전 학습으로 언어 모델 분해", authors: ["Trenton Bricken", "Adly Templeton", "Joshua Batson", "et al."], year: 2023, venue: "Anthropic Research", venueType: "other", conferenceUrl: "https://transformer-circuits.pub/2023/monosemantic-features", tags: [], status: "complete", citations: "500+" },
        ],
      },
    ],
  },
  {
    id: "optimization",
    titleKo: "최적화·학습이론",
    titleEn: "Optimization & Learning Theory",
    descriptionKo: "SGD, Adam, 수렴 이론, 일반화, 손실 함수 연구",
    descriptionEn: "Research on optimization methods and learning theory",
    color: "blue",
    years: [
      {
        year: 2017,
        papers: [
          { id: "adamw", title: "Decoupled Weight Decay Regularization", titleKo: "분리된 가중치 감쇠 정규화", authors: ["Ilya Loshchilov", "Frank Hutter"], year: 2017, venue: "ICLR 2019", venueType: "iclr", arxivUrl: "https://arxiv.org/abs/1711.05101", tags: [], status: "complete", citations: "10,000+" },
        ],
      },
      {
        year: 2019,
        papers: [
          { id: "lottery-ticket", title: "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks", titleKo: "복권 가설: 희소하고 학습 가능한 신경망 찾기", authors: ["Jonathan Frankle", "Michael Carlin"], year: 2019, venue: "ICLR 2019", venueType: "iclr", award: "best-paper", arxivUrl: "https://arxiv.org/abs/1803.03635", tags: ["theory"], status: "complete", citations: "5,000+" },
        ],
      },
      {
        year: 2023,
        papers: [
          { id: "lion", title: "Symbolic Discovery of Optimization Algorithms", titleKo: "최적화 알고리즘의 기호적 발견", authors: ["Xiangning Chen", "Chen Liang", "Da Huang", "et al."], year: 2023, venue: "NeurIPS 2023", venueType: "neurips", arxivUrl: "https://arxiv.org/abs/2302.06675", tags: [], status: "complete", citations: "500+" },
        ],
      },
    ],
  },
  {
    id: "representation",
    titleKo: "표현 학습",
    titleEn: "Representation Learning",
    descriptionKo: "자기지도학습, 대조학습, 사전학습, 전이학습 연구",
    descriptionEn: "Research on learning useful representations",
    color: "emerald",
    years: [
      {
        year: 2020,
        papers: [
          { id: "simclr", title: "A Simple Framework for Contrastive Learning of Visual Representations", titleKo: "시각 표현의 대조 학습을 위한 간단한 프레임워크", authors: ["Ting Chen", "Simon Kornblith", "Mohammad Norouzi", "Geoffrey Hinton"], year: 2020, venue: "ICML 2020", venueType: "icml", arxivUrl: "https://arxiv.org/abs/2002.05709", tags: [], status: "complete", citations: "12,000+" },
          { id: "byol", title: "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning", titleKo: "스스로의 잠재 표현을 부트스트랩하라", authors: ["Jean-Bastien Grill", "Florian Strub", "Florent Altché", "et al."], year: 2020, venue: "NeurIPS 2020", venueType: "neurips", arxivUrl: "https://arxiv.org/abs/2006.07733", tags: [], status: "complete", citations: "5,000+" },
        ],
      },
      {
        year: 2021,
        papers: [
          { id: "dino", title: "Emerging Properties in Self-Supervised Vision Transformers", titleKo: "자기지도 비전 트랜스포머에서 나타나는 특성", authors: ["Mathilde Caron", "Hugo Touvron", "Ishan Misra", "et al."], year: 2021, venue: "ICCV 2021", venueType: "iccv", arxivUrl: "https://arxiv.org/abs/2104.14294", tags: [], status: "complete", citations: "5,000+" },
        ],
      },
      {
        year: 2022,
        papers: [
          { id: "mae", title: "Masked Autoencoders Are Scalable Vision Learners", titleKo: "마스크 오토인코더는 확장 가능한 비전 학습기이다", authors: ["Kaiming He", "Xinlei Chen", "Saining Xie", "et al."], year: 2022, venue: "CVPR 2022", venueType: "cvpr", arxivUrl: "https://arxiv.org/abs/2111.06377", tags: [], status: "complete", citations: "5,000+" },
        ],
      },
    ],
  },
  {
    id: "science",
    titleKo: "AI for Science",
    titleEn: "AI for Science",
    descriptionKo: "단백질 접힘, 신약, 기후, 수학 증명 등 과학 연구에 AI 적용",
    descriptionEn: "Applying AI to scientific research",
    color: "lime",
    years: [
      {
        year: 2021,
        papers: [
          { id: "alphafold2", title: "Highly accurate protein structure prediction with AlphaFold", titleKo: "AlphaFold를 이용한 고정밀 단백질 구조 예측", authors: ["John Jumper", "Richard Evans", "Alexander Pritzel", "et al."], year: 2021, venue: "Nature", venueType: "nature", arxivUrl: "https://doi.org/10.1038/s41586-021-03819-2", tags: [], status: "complete", citations: "25,000+" },
        ],
      },
      {
        year: 2024,
        papers: [
          { id: "alphageometry", title: "Solving olympiad geometry without human demonstrations", titleKo: "인간 시연 없이 올림피아드 기하 문제 풀기", authors: ["Trieu H. Trinh", "Yuhuai Wu", "Quoc V. Le", "et al."], year: 2024, venue: "Nature", venueType: "nature", arxivUrl: "https://doi.org/10.1038/s41586-023-06747-5", tags: [], status: "complete", citations: "500+" },
        ],
      },
    ],
  },
  {
    id: "efficient",
    titleKo: "경량화·효율화",
    titleEn: "Efficient AI",
    descriptionKo: "양자화, 프루닝, 증류, NAS, 추론 최적화 연구",
    descriptionEn: "Research on making AI models smaller and faster",
    color: "sky",
    years: [
      {
        year: 2015,
        papers: [
          { id: "distillation", title: "Distilling the Knowledge in a Neural Network", titleKo: "신경망의 지식 증류", authors: ["Geoffrey Hinton", "Oriol Vinyals", "Jeff Dean"], year: 2015, venue: "NeurIPS 2014 Workshop", venueType: "neurips", arxivUrl: "https://arxiv.org/abs/1503.02531", tags: [], status: "complete", citations: "15,000+" },
        ],
      },
      {
        year: 2021,
        papers: [
          { id: "lora", title: "LoRA: Low-Rank Adaptation of Large Language Models", titleKo: "LoRA: 대규모 언어 모델의 저순위 적응", authors: ["Edward J. Hu", "Yelong Shen", "Phillip Wallis", "et al."], year: 2021, venue: "ICLR 2022", venueType: "iclr", arxivUrl: "https://arxiv.org/abs/2106.09685", tags: [], status: "complete", citations: "8,000+" },
        ],
      },
      {
        year: 2023,
        papers: [
          { id: "qlora", title: "QLoRA: Efficient Finetuning of Quantized LLMs", titleKo: "QLoRA: 양자화된 LLM의 효율적 미세 조정", authors: ["Tim Dettmers", "Artidoro Pagnoni", "Ari Holtzman", "Luke Zettlemoyer"], year: 2023, venue: "NeurIPS 2023", venueType: "neurips", arxivUrl: "https://arxiv.org/abs/2305.14314", tags: [], status: "complete", citations: "3,000+" },
        ],
      },
    ],
  },
  {
    id: "world-models",
    titleKo: "월드 모델",
    titleEn: "World Models",
    descriptionKo: "비디오 예측, 시뮬레이션, 내부 세계 모델 연구",
    descriptionEn: "Research on learning world representations",
    color: "purple",
    years: [
      {
        year: 2018,
        papers: [
          { id: "world-models-ha", title: "World Models", titleKo: "월드 모델", authors: ["David Ha", "Jürgen Schmidhuber"], year: 2018, venue: "NeurIPS 2018", venueType: "neurips", arxivUrl: "https://arxiv.org/abs/1803.10122", tags: [], status: "complete", citations: "3,000+" },
        ],
      },
      {
        year: 2024,
        papers: [
          { id: "sora", title: "Video generation models as world simulators", titleKo: "세계 시뮬레이터로서의 비디오 생성 모델", authors: ["OpenAI"], year: 2024, venue: "OpenAI Technical Report", venueType: "other", conferenceUrl: "https://openai.com/research/video-generation-models-as-world-simulators", tags: [], status: "complete", citations: "N/A" },
          { id: "genie", title: "Genie: Generative Interactive Environments", titleKo: "Genie: 생성적 상호작용 환경", authors: ["Jake Bruce", "Michael Dennis", "Ashley Edwards", "et al."], year: 2024, venue: "ICML 2024", venueType: "icml", arxivUrl: "https://arxiv.org/abs/2402.15391", tags: [], status: "complete", citations: "200+" },
        ],
      },
    ],
  },
  {
    id: "audio",
    titleKo: "음성·오디오",
    titleEn: "Audio & Speech",
    descriptionKo: "ASR, TTS, 음악 생성, 오디오 이해 연구",
    descriptionEn: "Research on audio and speech processing",
    color: "slate",
    years: [
      {
        year: 2016,
        papers: [
          { id: "wavenet", title: "WaveNet: A Generative Model for Raw Audio", titleKo: "WaveNet: 원시 오디오를 위한 생성 모델", authors: ["Aäron van den Oord", "Sander Dieleman", "Heiga Zen", "et al."], year: 2016, venue: "arXiv", venueType: "arxiv", arxivUrl: "https://arxiv.org/abs/1609.03499", tags: [], status: "complete", citations: "10,000+" },
        ],
      },
      {
        year: 2022,
        papers: [
          { id: "whisper", title: "Robust Speech Recognition via Large-Scale Weak Supervision", titleKo: "대규모 약한 감독을 통한 강건한 음성 인식", authors: ["Alec Radford", "Jong Wook Kim", "Tao Xu", "et al."], year: 2022, venue: "ICML 2023", venueType: "icml", arxivUrl: "https://arxiv.org/abs/2212.04356", tags: [], status: "complete", citations: "5,000+" },
        ],
      },
      {
        year: 2023,
        papers: [
          { id: "vall-e", title: "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers", titleKo: "신경 코덱 언어 모델은 제로샷 텍스트 음성 합성기이다", authors: ["Chengyi Wang", "Sanyuan Chen", "Yu Wu", "et al."], year: 2023, venue: "arXiv", venueType: "arxiv", arxivUrl: "https://arxiv.org/abs/2301.02111", tags: [], status: "complete", citations: "1,000+" },
        ],
      },
    ],
  },
];
