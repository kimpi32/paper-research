import type { PaperSummary } from "./paper-summaries";

export const nlpLlmSummaries: Record<string, PaperSummary> = {
  "word2vec": {
    tldr: "단어를 저차원 밀집 벡터로 표현하는 효율적인 학습 방법(CBOW, Skip-gram)을 제안하여, 단어 간 의미적 관계를 벡터 연산으로 포착할 수 있음을 보여주었다.",
    background: "전통적인 NLP에서는 단어를 원-핫 인코딩으로 표현했으나, 이는 단어 간 유사도를 반영하지 못하고 차원이 어휘 크기에 비례하여 비효율적이었다. 신경망 기반 언어 모델(NNLM)이 분산 표현을 학습할 수 있음이 알려졌으나, 계산 비용이 매우 높아 대규모 코퍼스에 적용하기 어려웠다.",
    keyIdea: "Mikolov 등은 은닉층을 제거하거나 단순화한 두 가지 아키텍처를 제안했다. CBOW(Continuous Bag-of-Words)는 주변 단어들로부터 중심 단어를 예측하고, Skip-gram은 반대로 중심 단어로부터 주변 단어를 예측한다. 이러한 단순한 구조 덕분에 수십억 단어 규모의 코퍼스에서도 효율적으로 학습이 가능했다. 학습된 벡터는 'king - man + woman = queen'과 같은 유추 관계를 벡터 산술로 표현할 수 있는 놀라운 성질을 보여주었다. Negative sampling과 subsampling 등의 학습 기법도 함께 제안되어 학습 효율을 크게 높였다.",
    method: "CBOW 모델은 문맥 윈도우 내의 주변 단어 벡터를 평균하여 중심 단어를 예측하며, Skip-gram 모델은 중심 단어로부터 각 주변 단어의 출현 확률을 최대화한다. Hierarchical softmax와 negative sampling을 통해 전체 어휘에 대한 softmax 계산을 회피하여 학습 속도를 획기적으로 개선했다.",
    results: "Google 유추 테스트셋에서 Skip-gram 모델이 기존 방법 대비 월등한 성능을 달성했다. 특히 의미적 유추(semantic analogy)와 구문적 유추(syntactic analogy) 모두에서 우수한 결과를 보였으며, 더 큰 코퍼스와 높은 벡터 차원에서 성능이 향상되는 경향을 확인했다.",
    impact: "Word2Vec은 NLP 분야에서 사전 학습된 단어 임베딩 활용을 대중화한 혁명적 연구이다. 이후 GloVe, FastText 등 후속 임베딩 연구의 토대가 되었고, 분산 표현이라는 개념은 ELMo, BERT 등 문맥화 임베딩으로 발전하는 출발점이 되었다. 현대 NLP의 전이 학습 패러다임을 여는 데 결정적 역할을 했다.",
    relatedFoundations: ["backpropagation"],
    relatedPapers: [
      { id: "elmo", fieldId: "nlp", title: "Deep contextualized word representations", relation: "successor" },
      { id: "t5", fieldId: "nlp", title: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", relation: "successor" },
    ],
  },

  "elmo": {
    tldr: "사전 학습된 양방향 LSTM 언어 모델의 내부 표현을 결합하여, 문맥에 따라 달라지는 단어 임베딩(contextualized word embedding)을 생성하는 ELMo를 제안했다.",
    background: "Word2Vec이나 GloVe 같은 정적 단어 임베딩은 동음이의어나 다의어를 구분하지 못하는 근본적 한계가 있었다. 예를 들어 'bank'라는 단어가 은행인지 강둑인지에 관계없이 동일한 벡터로 표현되었다. 문맥에 따라 단어의 의미가 달라진다는 사실을 반영하는 새로운 표현 방식이 필요했다.",
    keyIdea: "ELMo(Embeddings from Language Models)는 대규모 코퍼스에서 양방향 LSTM 언어 모델을 사전 학습한 뒤, 모델의 각 층에서 추출한 표현을 가중 합산하여 문맥화된 단어 벡터를 생성한다. 핵심 통찰은 언어 모델의 서로 다른 층이 서로 다른 유형의 정보를 포착한다는 것이다. 하위 층은 구문적(syntactic) 정보를, 상위 층은 의미적(semantic) 정보를 더 많이 담고 있다. 다운스트림 태스크에 따라 각 층의 가중치를 학습함으로써 태스크에 최적화된 표현을 얻을 수 있다.",
    method: "2층 양방향 LSTM을 사용하여 순방향과 역방향 언어 모델을 동시에 학습한다. 입력 토큰에 대해 문자(character) 수준 CNN으로 초기 임베딩을 생성한 뒤, 양방향 LSTM 각 층의 출력을 태스크별 학습 가능한 가중치로 선형 결합한다. 이렇게 생성된 ELMo 벡터를 기존 모델의 입력에 추가(concatenate)하는 방식으로 활용한다.",
    results: "6개의 NLP 벤치마크(질의응답, 텍스트 함의, 의미역 결정, 공참조 해결, 개체명 인식, 감성 분석)에서 ELMo를 추가하면 기존 최고 성능 모델 대비 평균 상대 오차 감소율이 크게 향상되었다. 특히 SQuAD 질의응답에서는 절대 성능이 크게 개선되었다.",
    impact: "ELMo는 '사전 학습 후 미세 조정' 패러다임의 핵심 선구자로서, 문맥화 임베딩이라는 새로운 방향을 개척했다. 이 아이디어는 곧바로 BERT, GPT 등 Transformer 기반 사전 학습 모델로 이어졌으며, 현대 NLP의 근간이 되는 전이 학습 방법론을 확립하는 데 핵심적 역할을 했다.",
    relatedFoundations: ["lstm"],
    relatedPapers: [
      { id: "word2vec", fieldId: "nlp", title: "Efficient Estimation of Word Representations in Vector Space", relation: "prior" },
      { id: "bert", fieldId: "foundations", title: "BERT: Pre-training of Deep Bidirectional Transformers", relation: "successor" },
      { id: "xlnet", fieldId: "nlp", title: "XLNet: Generalized Autoregressive Pretraining", relation: "successor" },
    ],
  },

  "t5": {
    tldr: "모든 NLP 태스크를 텍스트-투-텍스트(text-to-text) 형식으로 통일하고, 대규모 비교 연구를 통해 전이 학습의 최적 전략을 체계적으로 탐색한 연구이다.",
    background: "BERT, GPT 등의 사전 학습 모델이 NLP에서 큰 성공을 거두었으나, 사전 학습 목표, 아키텍처, 데이터셋 크기, 미세 조정 방법 등 수많은 설계 선택지가 존재했다. 각 연구가 서로 다른 설정을 사용하여 어떤 요소가 실제로 중요한지 비교하기 어려운 상황이었다.",
    keyIdea: "T5(Text-to-Text Transfer Transformer)는 분류, 번역, 요약, 질의응답 등 모든 NLP 태스크를 '입력 텍스트 → 출력 텍스트' 형식으로 변환하는 통합 프레임워크를 제안한다. 이를 통해 동일한 모델, 동일한 학습 절차, 동일한 손실 함수를 모든 태스크에 적용할 수 있다. 또한 C4(Colossal Clean Crawled Corpus)라는 대규모 정제 데이터셋을 구축하고, 아키텍처(인코더-디코더 vs 디코더 전용), 사전 학습 목표(언어 모델링, 마스킹 등), 데이터셋 크기, 학습 전략 등을 체계적으로 비교 실험했다. 이 방대한 비교 연구 자체가 논문의 핵심 기여이다.",
    method: "인코더-디코더 Transformer 아키텍처를 기반으로, 입력에 태스크 접두어(예: 'translate English to German:', 'summarize:')를 붙여 태스크를 구분한다. 사전 학습 목표로는 span corruption(연속 토큰을 마스킹하고 복원)이 가장 효과적임을 확인했다. 모델 크기를 Small부터 11B 파라미터까지 다양하게 실험했다.",
    results: "T5-11B 모델은 GLUE, SuperGLUE, SQuAD, WMT 번역 등 다수의 벤치마크에서 당시 최고 성능을 달성했다. 특히 SuperGLUE에서 인간 수준의 성능에 근접한 결과를 보였다. 비교 연구를 통해 인코더-디코더 구조, span corruption 사전 학습, 충분한 데이터와 모델 크기의 중요성을 실증적으로 입증했다.",
    impact: "T5는 텍스트-투-텍스트 패러다임을 통해 NLP 태스크의 통합적 처리 방식을 제시했으며, 이는 이후 GPT-3의 프롬프트 기반 접근과 함께 현대 LLM의 범용적 활용 방식에 큰 영향을 미쳤다. C4 데이터셋은 후속 연구에서 널리 활용되었고, 체계적 비교 연구는 전이 학습 연구의 모범적 방법론을 제시했다.",
    relatedFoundations: ["transformer", "bert"],
    relatedPapers: [
      { id: "bert", fieldId: "foundations", title: "BERT: Pre-training of Deep Bidirectional Transformers", relation: "prior" },
      { id: "xlnet", fieldId: "nlp", title: "XLNet: Generalized Autoregressive Pretraining", relation: "related" },
    ],
  },

  "xlnet": {
    tldr: "BERT의 마스킹 기반 사전 학습의 한계를 극복하기 위해 순열 언어 모델링(permutation language modeling)을 제안하여, 자기 회귀 모델의 장점과 양방향 문맥 활용을 동시에 달성했다.",
    background: "BERT는 마스크드 언어 모델(MLM)을 통해 양방향 문맥을 활용하여 큰 성공을 거두었으나, 사전 학습 시 사용하는 [MASK] 토큰이 미세 조정 시에는 등장하지 않아 사전 학습-미세 조정 간 불일치(pretrain-finetune discrepancy)가 발생했다. 또한 마스킹된 토큰들이 서로 독립이라고 가정하여 토큰 간 상관관계를 무시하는 문제도 있었다.",
    keyIdea: "XLNet은 순열 언어 모델링이라는 새로운 사전 학습 목표를 도입한다. 입력 시퀀스의 모든 가능한 순서(permutation)를 고려하여 자기 회귀 방식으로 학습함으로써, [MASK] 토큰 없이도 양방향 문맥 정보를 활용할 수 있다. 구체적으로 각 학습 단계에서 토큰 순서의 무작위 순열을 샘플링하고, 해당 순열에 따른 조건부 확률을 자기 회귀적으로 분해한다. 이를 통해 BERT의 독립성 가정 없이 모든 토큰 간 의존성을 모델링할 수 있다. 또한 Transformer-XL의 세그먼트 재귀 메커니즘을 통합하여 긴 문맥 처리 능력을 향상시켰다.",
    method: "Two-stream self-attention 메커니즘을 도입하여, 예측 대상 토큰의 위치 정보만 사용하는 query stream과 내용 정보를 포함하는 content stream을 분리한다. Transformer-XL의 상대 위치 인코딩과 세그먼트 수준 재귀를 채택하여 긴 시퀀스에 대한 의존성을 효과적으로 포착한다.",
    results: "XLNet은 20개의 NLP 벤치마크에서 BERT를 능가하는 성능을 달성했다. 특히 SQuAD 2.0, RACE 독해 이해, GLUE 벤치마크 등에서 유의미한 성능 향상을 보였다. 긴 문서를 다루는 태스크에서 Transformer-XL 구조의 이점이 특히 두드러졌다.",
    impact: "XLNet은 자기 회귀 모델과 자기 인코딩 모델의 장점을 결합하는 새로운 방향을 제시했다. 순열 언어 모델링이라는 사전 학습 패러다임은 이후 연구에 영감을 주었으며, 사전 학습 목표 설계에서 마스킹 방식의 대안을 탐구하는 계기가 되었다.",
    relatedFoundations: ["transformer", "bert"],
    relatedPapers: [
      { id: "bert", fieldId: "foundations", title: "BERT: Pre-training of Deep Bidirectional Transformers", relation: "prior" },
      { id: "t5", fieldId: "nlp", title: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", relation: "related" },
    ],
  },

  "instructgpt": {
    tldr: "인간 피드백을 활용한 강화 학습(RLHF)으로 GPT-3를 미세 조정하여, 사용자의 의도를 더 정확히 따르면서 유해한 출력을 줄이는 InstructGPT를 개발했다.",
    background: "GPT-3와 같은 대규모 언어 모델은 프롬프트에 따라 다양한 태스크를 수행할 수 있었으나, 사용자의 의도와 다른 응답을 생성하거나, 거짓 정보를 만들어내거나, 유해한 콘텐츠를 출력하는 문제가 빈번했다. 이는 언어 모델의 학습 목표(다음 토큰 예측)가 '사용자에게 유용하고 안전한 응답 생성'이라는 실제 목표와 괴리가 있기 때문이었다.",
    keyIdea: "InstructGPT는 세 단계의 학습 과정을 통해 언어 모델을 인간의 의도에 정렬(align)한다. 첫째, 인간이 작성한 시연 데이터로 지도 학습 미세 조정(SFT)을 수행한다. 둘째, 모델이 생성한 여러 응답에 대해 인간 평가자가 순위를 매긴 비교 데이터로 보상 모델(RM)을 학습한다. 셋째, 학습된 보상 모델을 사용하여 PPO(Proximal Policy Optimization) 알고리즘으로 정책을 최적화한다. 이 과정에서 1.3B 파라미터의 InstructGPT가 175B의 GPT-3보다 인간 평가에서 선호되는 놀라운 결과를 보여주었다.",
    method: "총 40명의 계약 레이블러가 참여하여 시연 데이터와 비교 데이터를 생성했다. SFT 단계에서 약 13,000개의 프롬프트-응답 쌍으로 학습하고, RM 단계에서 약 33,000개의 비교 데이터로 6B 보상 모델을 학습했다. PPO 단계에서는 31,000개의 프롬프트를 사용하여 정책을 최적화하되, 원래 언어 모델 분포에서 너무 벗어나지 않도록 KL 패널티를 부과했다.",
    results: "인간 평가에서 1.3B InstructGPT의 출력이 175B GPT-3 대비 압도적으로 선호되었다. TruthfulQA 벤치마크에서 진실성이 향상되었고, 유해한 출력 생성이 감소했다. 다만 코딩 태스크 등 일부 영역에서는 기존 GPT-3 대비 성능 저하(alignment tax)가 관찰되기도 했다.",
    impact: "InstructGPT는 RLHF를 통한 AI 정렬의 실용성을 대규모로 입증한 획기적 연구이다. ChatGPT의 직접적 기반이 되었으며, 이후 거의 모든 상용 LLM이 RLHF 또는 유사한 정렬 기법을 채택하게 만들었다. '도움이 되고, 해가 없고, 정직한' AI라는 정렬 목표를 실용적으로 구현하는 방법론의 표준을 제시했다.",
    relatedFoundations: ["gpt", "gpt3"],
    relatedPapers: [
      { id: "gpt3", fieldId: "foundations", title: "Language Models are Few-Shot Learners", relation: "prior" },
      { id: "rlhf", fieldId: "safety", title: "Training a Helpful and Harmless Assistant from Human Feedback", relation: "related" },
      { id: "chinchilla", fieldId: "llm", title: "Training Compute-Optimal Large Language Models", relation: "related" },
    ],
  },

  "chinchilla": {
    tldr: "고정된 연산 예산에서 모델 크기와 학습 데이터 양의 최적 비율을 분석하여, 기존 LLM들이 과도하게 크고 데이터가 부족하게 학습되었음을 밝혔다.",
    background: "GPT-3 이후 LLM 연구는 모델 파라미터 수를 늘리는 방향으로 경쟁하고 있었다. Kaplan 등의 스케일링 법칙 연구는 모델 크기를 키우는 것이 데이터 양을 늘리는 것보다 더 중요하다고 시사하여, 대규모 모델을 상대적으로 적은 데이터로 학습하는 경향이 지배적이었다.",
    keyIdea: "Hoffmann 등은 세 가지 독립적인 방법으로 연산 예산(compute budget)에 따른 최적의 모델 크기와 학습 토큰 수를 추정했다. 그 결과 모델 파라미터 수와 학습 토큰 수가 동일한 비율로 스케일링되어야 한다는 결론에 도달했다. 구체적으로 모델 파라미터 수가 두 배가 되면 학습 토큰 수도 두 배가 되어야 최적이다. 이 분석에 따르면 Gopher(280B)는 같은 연산 예산으로 70B 모델을 4배 더 많은 데이터로 학습시키는 것이 최적이었다. 이를 검증하기 위해 70B 파라미터의 Chinchilla를 1.4조 토큰으로 학습했다.",
    method: "세 가지 접근법을 사용했다. (1) 다양한 모델 크기에 대해 고정된 연산 예산별 손실 곡선을 피팅하여 최적점을 찾는 방법, (2) 고정된 FLOP에서 다양한 모델 크기로 실험하여 IsoFLOP 곡선을 분석하는 방법, (3) 모든 실험 결과에 파라메트릭 손실 함수를 직접 피팅하는 방법. 400개 이상의 학습 실행을 수행하여 분석했다.",
    results: "Chinchilla(70B, 1.4T 토큰)는 4배 큰 Gopher(280B, 300B 토큰)를 대부분의 평가 벤치마크에서 능가했다. MMLU에서 67.5%의 평균 정확도를 달성하여 Gopher의 60%를 크게 상회했으며, 추론 시 연산 비용도 4배 저렴했다.",
    impact: "Chinchilla 연구는 LLM 커뮤니티의 스케일링 전략을 근본적으로 바꾸었다. '모델을 크게 만드는 것보다 충분한 데이터로 학습시키는 것이 중요하다'는 메시지는 이후 LLaMA, Mistral 등 상대적으로 작지만 충분히 학습된 효율적 모델의 개발을 촉진했다. 컴퓨트 최적 학습이라는 개념을 실질적 지침으로 확립했다.",
    relatedFoundations: ["scaling-laws", "gpt3"],
    relatedPapers: [
      { id: "gpt3", fieldId: "foundations", title: "Language Models are Few-Shot Learners", relation: "prior" },
      { id: "llama", fieldId: "llm", title: "LLaMA: Open and Efficient Foundation Language Models", relation: "successor" },
    ],
  },

  "cot": {
    tldr: "프롬프트에 단계적 추론 과정(chain-of-thought)의 예시를 포함하면, 대규모 언어 모델의 산술, 상식, 기호적 추론 능력이 크게 향상됨을 발견했다.",
    background: "GPT-3 등 대규모 언어 모델은 few-shot 프롬프팅으로 다양한 태스크를 수행할 수 있었으나, 다단계 추론이 필요한 수학 문제나 논리적 추론 문제에서는 여전히 어려움을 겪었다. 기존 few-shot 프롬프팅은 입력-출력 쌍만 제시했지, 중간 추론 과정을 보여주지 않았다.",
    keyIdea: "Wei 등은 매우 단순하면서도 강력한 아이디어를 제안했다. Few-shot 프롬프트의 예시에서 최종 답만 제시하는 대신, 답에 도달하는 중간 추론 단계를 자연어로 함께 제시하면 모델이 유사한 추론 과정을 생성하며 정확도가 크게 향상된다는 것이다. 이를 chain-of-thought(CoT) 프롬프팅이라 명명했다. 중요한 발견은 이 효과가 충분히 큰 모델(약 100B 파라미터 이상)에서만 나타나는 창발적(emergent) 능력이라는 점이다. 또한 CoT는 모델의 추론 과정을 해석 가능하게 만들어 오류 디버깅을 용이하게 한다.",
    method: "GSM8K(초등 수학), SVAMP, AQuA 등의 산술 추론, CommonsenseQA, StrategyQA 등의 상식 추론, 그리고 기호적 추론 벤치마크에서 실험했다. 표준 few-shot 프롬프팅과 동일한 설정에서 예시의 답변 부분에만 추론 체인을 추가하여 비교했다. PaLM 540B, GPT-3 175B, LaMDA 137B 등 다양한 모델에서 검증했다.",
    results: "PaLM 540B + CoT 프롬프팅은 GSM8K에서 57%의 정확도를 달성하여, 표준 프롬프팅(18%)을 크게 능가하고 당시 미세 조정된 최고 성능 모델과 비교 가능한 수준이었다. 특히 문제가 더 많은 추론 단계를 요구할수록 CoT의 성능 향상 폭이 커지는 것을 확인했다.",
    impact: "Chain-of-thought 프롬프팅은 프롬프트 엔지니어링 분야의 가장 영향력 있는 기법 중 하나로 자리잡았다. 이후 Zero-shot CoT('Let's think step by step'), Self-consistency, Tree-of-Thought 등 다양한 확장 연구를 촉발했다. LLM의 추론 능력을 끌어내는 핵심 기법으로서, GPT-4, Claude 등 현대 AI 시스템에서 광범위하게 활용되고 있다.",
    relatedFoundations: ["gpt3"],
    relatedPapers: [
      { id: "gpt3", fieldId: "foundations", title: "Language Models are Few-Shot Learners", relation: "prior" },
      { id: "gpt4", fieldId: "llm", title: "GPT-4 Technical Report", relation: "successor" },
    ],
  },

  "llama": {
    tldr: "공개적으로 이용 가능한 데이터만으로 학습한 7B~65B 규모의 오픈 소스 파운데이션 모델 LLaMA를 공개하여, 소규모 모델도 충분한 데이터로 학습하면 대형 모델에 필적하는 성능을 달성할 수 있음을 보여주었다.",
    background: "GPT-3, PaLM, Chinchilla 등 강력한 LLM들이 등장했으나, 대부분 비공개이거나 접근이 제한되어 학술 연구 커뮤니티에서 재현하거나 연구하기 어려웠다. Chinchilla의 연구는 모델 크기와 데이터 양의 균형이 중요함을 보여주었지만, 추론 시 비용도 고려하면 더 작은 모델을 더 오래 학습시키는 것이 실용적으로 유리할 수 있었다.",
    keyIdea: "LLaMA는 Chinchilla의 스케일링 법칙을 넘어, 추론 시 효율성까지 고려한 새로운 관점을 제시한다. 학습 연산 예산이 아닌 추론 시 목표 성능 수준을 기준으로, 해당 성능을 가장 빠르게 달성하는 모델 크기와 데이터 양 조합을 탐구한다. 결과적으로 7B 모델을 1조 토큰, 65B 모델을 1.4조 토큰으로 학습시켜(Chinchilla 최적보다 훨씬 많은 토큰), 추론 시 저렴하면서도 강력한 모델을 만들었다. 모든 학습 데이터는 공개 데이터셋(CommonCrawl, C4, Wikipedia, ArXiv, GitHub 등)으로 구성하여 재현 가능성을 높였다.",
    method: "표준 Transformer 아키텍처에 RMSNorm 정규화, SwiGLU 활성화 함수, 로터리 위치 임베딩(RoPE) 등의 최신 기법을 적용했다. 효율적 구현을 위해 인과적 다중 헤드 어텐션의 메모리 최적화, 체크포인팅을 통한 역전파 메모리 절감, 모델 및 시퀀스 병렬화를 활용했다.",
    results: "LLaMA-13B는 대부분의 벤치마크에서 GPT-3(175B)를 능가했으며, 단일 GPU에서도 실행 가능했다. LLaMA-65B는 Chinchilla-70B 및 PaLM-540B와 비교 가능한 성능을 보여주었다. MMLU, HellaSwag, ARC, WinoGrande 등 다양한 벤치마크에서 경쟁력 있는 결과를 달성했다.",
    impact: "LLaMA의 공개는 오픈 소스 LLM 생태계의 폭발적 성장을 촉발한 결정적 사건이었다. Alpaca, Vicuna, LLaMA-2 등 수많은 파생 모델과 미세 조정 연구가 이어졌으며, 연구 커뮤니티와 스타트업이 자체 LLM을 개발할 수 있는 기반을 마련했다. 이후 Mistral, Mixtral 등으로 이어지는 오픈 소스 LLM 혁명의 시작점이 되었다.",
    relatedFoundations: ["transformer", "gpt", "gpt3", "scaling-laws"],
    relatedPapers: [
      { id: "chinchilla", fieldId: "llm", title: "Training Compute-Optimal Large Language Models", relation: "prior" },
      { id: "gpt4", fieldId: "llm", title: "GPT-4 Technical Report", relation: "related" },
    ],
  },

  "gpt4": {
    tldr: "멀티모달 입력(텍스트+이미지)을 처리할 수 있는 대규모 언어 모델 GPT-4를 공개하여, 전문가 수준의 시험 성적과 다양한 벤치마크에서 획기적 성능을 달성했다.",
    background: "GPT-3와 ChatGPT(InstructGPT)의 성공 이후, LLM의 능력을 한층 더 높이기 위한 연구가 진행되었다. 특히 복잡한 추론, 코딩, 수학, 전문 지식 등에서의 한계를 극복하고, 텍스트뿐 아니라 이미지도 이해할 수 있는 멀티모달 능력의 필요성이 대두되었다.",
    keyIdea: "GPT-4는 텍스트와 이미지를 모두 입력으로 받아 텍스트를 생성하는 대규모 멀티모달 모델이다. OpenAI는 경쟁적 고려와 안전 문제로 아키텍처, 모델 크기, 학습 데이터 등의 세부 사항을 공개하지 않았다. 대신 GPT-4의 능력을 다양한 시험과 벤치마크를 통해 실증적으로 보여주었다. 특히 주목할 점은 학습 과정에서의 예측 가능성(predictable scaling)인데, 소규모 모델의 성능으로부터 GPT-4의 최종 성능을 높은 정확도로 예측할 수 있는 인프라를 구축했다는 것이다. 또한 RLHF를 통한 안전성 향상 작업도 체계적으로 수행했다.",
    method: "GPT-4는 Transformer 기반 사전 학습 모델로, 다음 토큰 예측으로 학습된 후 RLHF로 정렬되었다. 구체적인 아키텍처와 학습 세부 사항은 비공개이다. 안전성을 위해 도메인 전문가의 적대적 테스트(red teaming)와 규칙 기반 보상 모델(RBRM) 등의 기법을 활용했다.",
    results: "GPT-4는 미국 변호사 시험에서 상위 약 10%(GPT-3.5는 하위 10%), SAT 수학에서 700/800, GRE 정량 추론에서 163/170을 달성했다. 학술 벤치마크에서는 MMLU 86.4%(few-shot), HellaSwag 95.3% 등 기존 모델을 크게 상회하는 결과를 보였다. 이미지 입력을 활용한 시각적 추론에서도 우수한 성능을 나타냈다.",
    impact: "GPT-4는 LLM의 실용적 능력이 전문가 수준에 도달할 수 있음을 보여주며 AI 산업 전반에 큰 영향을 미쳤다. 교육, 법률, 의료 등 다양한 분야에서의 AI 활용 가능성을 입증했고, 멀티모달 AI의 시대를 본격적으로 열었다. 동시에 기술 세부 사항 비공개에 대한 학술 커뮤니티의 투명성 논의를 촉발했다.",
    relatedFoundations: ["transformer", "gpt", "gpt3", "scaling-laws"],
    relatedPapers: [
      { id: "gpt3", fieldId: "foundations", title: "Language Models are Few-Shot Learners", relation: "prior" },
      { id: "llama", fieldId: "llm", title: "LLaMA: Open and Efficient Foundation Language Models", relation: "related" },
      { id: "cot", fieldId: "llm", title: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", relation: "prior" },
    ],
  },

  "rag": {
    tldr: "검색(retrieval)과 생성(generation)을 결합하여, 외부 지식 저장소에서 관련 문서를 검색한 뒤 이를 참조하여 답변을 생성하는 RAG 프레임워크를 제안했다.",
    background: "사전 학습된 언어 모델은 파라미터에 암묵적으로 저장된 지식을 활용하지만, 이는 학습 시점에 고정되어 업데이트가 어렵고, 환각(hallucination) 문제가 있으며, 출처를 추적하기 어렵다는 한계가 있었다. 지식 집약적 태스크에서는 이러한 한계가 특히 두드러졌다.",
    keyIdea: "RAG(Retrieval-Augmented Generation)는 사전 학습된 seq2seq 모델(BART)과 밀집 벡터 검색기(DPR)를 결합한 생성 모델이다. 질문이 주어지면 DPR로 위키피디아 등의 외부 문서에서 관련 구절을 검색하고, 검색된 문서를 문맥으로 활용하여 답변을 생성한다. 두 가지 변형이 제안되었는데, RAG-Sequence는 하나의 검색 문서로 전체 답변을 생성하고, RAG-Token은 각 토큰 생성 시 서로 다른 문서를 참조할 수 있다. 핵심적으로 검색기와 생성기가 함께 end-to-end로 학습되어 검색과 생성이 상호 최적화된다.",
    method: "질문 인코더(BERT 기반)가 질문을 벡터로 변환하고, MIPS(Maximum Inner Product Search)로 사전 인코딩된 문서 벡터 인덱스에서 상위 k개 문서를 검색한다. 검색된 문서와 질문을 결합하여 BART 디코더에 입력하고 답변을 생성한다. 학습 시 질문 인코더와 생성기는 end-to-end로 미세 조정되며, 문서 인코더는 고정된다.",
    results: "오픈 도메인 질의응답(Natural Questions, TriviaQA, WebQuestions)에서 당시 최고 성능의 추출형(extractive) QA 모델과 비교 가능하거나 능가하는 결과를 달성했다. 또한 사실 검증(FEVER)과 지식 기반 생성 태스크(Jeopardy 질문 생성)에서도 우수한 성능을 보였다.",
    impact: "RAG는 LLM의 환각 문제를 완화하고 최신 정보를 활용할 수 있게 하는 핵심 기법으로 자리잡았다. 이후 Bing Chat, Perplexity 등 검색 증강 AI 서비스의 기반이 되었으며, 기업용 AI 시스템에서 내부 문서를 활용한 질의응답 구축의 표준 아키텍처로 채택되었다. LLM 활용의 가장 중요한 실용 기법 중 하나로 평가받고 있다.",
    relatedFoundations: ["transformer", "bert"],
    relatedPapers: [
      { id: "bert", fieldId: "foundations", title: "BERT: Pre-training of Deep Bidirectional Transformers", relation: "prior" },
    ],
  },
};
