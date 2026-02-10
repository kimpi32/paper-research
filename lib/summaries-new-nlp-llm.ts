import type { PaperSummary } from "./paper-summaries";

export const newNlpLlmSummaries: Record<string, PaperSummary> = {
  "glove": {
    tldr: "단어-단어 동시 출현 행렬의 통계 정보를 활용하여 단어 벡터를 학습하는 GloVe를 제안했다. 전역 행렬 분해와 지역 문맥 윈도우 방법의 장점을 결합하여, Word2Vec과 동등하거나 우수한 단어 표현을 효율적으로 학습한다.",
    background: "단어 임베딩 연구는 크게 두 가지 흐름이 있었다. 첫째는 LSA(잠재 의미 분석)처럼 전역 동시 출현 통계를 활용하는 행렬 분해 방법이고, 둘째는 Word2Vec처럼 지역 문맥 윈도우를 순회하며 학습하는 방법이다. LSA는 전역 통계를 잘 포착하지만 단어 유추 태스크에서 약하고, Word2Vec은 유추에 강하지만 전역 통계를 직접 활용하지 않았다. 두 접근법의 장점을 결합하는 방법이 필요했다.",
    keyIdea: "GloVe(Global Vectors)는 동시 출현 확률의 비율(ratio)이 단어 간 의미적 관계를 인코딩한다는 통찰에서 출발한다. 예를 들어 'ice'와 'steam'에 대해 'solid'의 동시 출현 확률 비율은 크고, 'water'의 비율은 1에 가깝다. 이러한 비율 관계를 벡터 공간에서 보존하도록 가중 최소제곱 회귀 목적 함수를 설계했다. 구체적으로 두 단어 벡터의 내적이 해당 단어 쌍의 로그 동시 출현 횟수와 가까워지도록 학습한다. 빈도가 높은 동시 출현에 과도한 가중치가 부여되는 것을 방지하기 위해 가중 함수를 도입했다.",
    method: "전체 코퍼스에서 단어-단어 동시 출현 행렬 X를 구성한 뒤, 비용 함수 J = \u03a3 f(X_ij)(w_i^T w\u0303_j + b_i + b\u0303_j - log X_ij)^2를 최소화한다. f(x)는 빈도에 따른 가중 함수로 x_max(기본값 100)를 기준으로 클리핑된다. AdaGrad 옵티마이저를 사용하여 학습하며, 최종 단어 벡터는 w + w\u0303의 합으로 구성한다.",
    results: "단어 유추 태스크에서 GloVe 300차원 벡터는 Word2Vec Skip-gram을 능가하는 75%의 정확도를 달성했다. 단어 유사도 벤치마크(WordSim-353, MC, RG, SCWS 등)에서도 경쟁력 있는 결과를 보였다. 특히 벡터 차원과 학습 데이터 크기가 증가할수록 성능이 꾸준히 향상되는 안정적 스케일링 특성을 보여주었다.",
    impact: "GloVe는 Word2Vec과 함께 사전 학습된 단어 임베딩의 양대 산맥으로 자리잡았다. 공개된 사전 학습 벡터(6B, 42B, 840B 토큰)는 NLP 연구와 실무에서 폭넓게 활용되었으며, 전역 통계 기반 임베딩이 신경망 기반 방법과 동등한 성능을 달성할 수 있음을 보여주었다. 이후 FastText, ELMo 등 후속 임베딩 연구에도 중요한 비교 기준이 되었다.",
    relatedFoundations: ["backpropagation"],
    relatedPapers: [
      { id: "word2vec", fieldId: "nlp", title: "Efficient Estimation of Word Representations in Vector Space", relation: "prior" },
      { id: "elmo", fieldId: "nlp", title: "Deep contextualized word representations", relation: "successor" },
    ],
  },

  "bart": {
    tldr: "다양한 노이즈 함수를 적용하여 텍스트를 손상시킨 뒤 원본을 복원하도록 학습하는 디노이징 오토인코더 방식의 시퀀스-투-시퀀스 사전학습 모델 BART를 제안하여, 텍스트 생성과 이해 태스크 모두에서 우수한 성능을 달성했다.",
    background: "BERT는 마스크드 언어 모델로 자연어 이해(NLU)에서 뛰어난 성능을 보였으나, 자기 회귀적 생성에는 적합하지 않았다. GPT는 자기 회귀적 생성에 강하지만 양방향 문맥을 활용하지 못했다. T5는 텍스트-투-텍스트 형식으로 통일하는 접근을 제시했지만, 사전학습 목표의 선택에 대한 체계적 탐구가 필요했다. 이해와 생성을 모두 잘 수행하는 통합 모델에 대한 수요가 있었다.",
    keyIdea: "BART(Bidirectional and Auto-Regressive Transformers)는 표준 인코더-디코더 Transformer 아키텍처를 사용하는 디노이징 오토인코더이다. 핵심 아이디어는 입력 텍스트에 임의의 노이즈 함수를 적용하여 손상시킨 뒤, 모델이 원본 텍스트를 복원하도록 학습하는 것이다. 다양한 노이즈 변환을 체계적으로 비교 실험한 결과, 토큰 마스킹과 문장 순서 섞기(sentence permutation)를 결합한 방식이 가장 효과적임을 발견했다. 특히 연속된 여러 토큰을 하나의 마스크 토큰으로 대체하는 text infilling이 핵심 노이즈 함수로, 모델이 누락된 토큰의 수까지 예측해야 하므로 더 깊은 언어 이해를 요구한다.",
    method: "BERT와 동일한 규모(Base: 6층, Large: 12층)의 인코더-디코더 Transformer를 사용한다. 인코더는 손상된 텍스트를 양방향으로 처리하고, 디코더는 자기 회귀적으로 원본 텍스트를 생성한다. 사전학습 노이즈 함수로 text infilling(포아송 분포 람다=3으로 span 길이 결정)과 sentence permutation을 결합한다. 미세조정 시, 분류 태스크에는 디코더 최종 출력에 분류 헤드를 추가하고, 생성 태스크에는 입력을 인코더에 넣고 디코더로 직접 생성한다.",
    results: "BART-Large는 요약(CNN/DM에서 44.16 ROUGE-L, XSum에서 38.79 ROUGE-L)에서 당시 최고 성능을 달성하여 생성 태스크에서의 강점을 입증했다. 질의응답(SQuAD 2.0에서 88.8 F1), 자연어 추론(MNLI에서 89.9%)에서도 RoBERTa와 경쟁 가능한 성능을 보였다. 기계 번역(WMT16 Ro-En)에서도 역번역과 결합하여 우수한 결과를 달성했다.",
    impact: "BART는 인코더-디코더 구조의 사전학습 모델이 이해와 생성 모두에서 강력한 성능을 발휘할 수 있음을 보여주었다. 특히 텍스트 요약 분야에서 새로운 표준을 제시했으며, 이후 mBART(다국어 확장), PEGASUS(요약 특화) 등 후속 연구의 기반이 되었다. HuggingFace에서 요약, 번역 등 조건부 생성 태스크의 핵심 모델로 널리 활용되고 있으며, facebook/bart-large-mnli는 제로샷 분류의 표준 모델이 되었다.",
    relatedFoundations: ["transformer", "bert"],
    relatedPapers: [
      { id: "bert", fieldId: "foundations", title: "BERT: Pre-training of Deep Bidirectional Transformers", relation: "prior" },
      { id: "t5", fieldId: "nlp", title: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", relation: "related" },
      { id: "roberta", fieldId: "nlp", title: "RoBERTa: A Robustly Optimized BERT Pretraining Approach", relation: "related" },
    ],
  },

  "roberta": {
    tldr: "BERT의 학습 절차를 철저히 재검토하여, 더 큰 배치 크기, 더 많은 데이터, 더 긴 학습, 동적 마스킹, NSP 제거 등의 최적화를 통해 기존 BERT를 크게 능가하는 RoBERTa를 개발했다.",
    background: "BERT가 NLP에서 혁명적 성과를 거둔 뒤, XLNet, ERNIE 등 다양한 후속 모델들이 새로운 사전학습 목표나 아키텍처 변경을 통해 성능 향상을 시도했다. 그러나 BERT의 원래 학습 설정이 최적이었는지, 단순히 학습 절차를 잘 조정하는 것만으로도 성능이 크게 개선될 수 있는지는 체계적으로 검증되지 않았다.",
    keyIdea: "RoBERTa(Robustly optimized BERT approach)는 새로운 아키텍처나 학습 목표를 도입하지 않고, 기존 BERT의 학습 레시피만을 개선하여 놀라운 성능 향상을 달성한다. 핵심 발견들은 다음과 같다. (1) 정적 마스킹 대신 매 에포크마다 다른 마스킹 패턴을 적용하는 동적 마스킹이 동등 이상의 성능을 낸다. (2) 다음 문장 예측(NSP) 목표를 제거하면 오히려 다운스트림 성능이 향상된다. (3) 더 큰 배치(8K)와 더 많은 데이터(160GB)로 학습하면 성능이 지속적으로 향상된다. (4) BERT는 현저히 과소 학습(undertrained)되어 있었다.",
    method: "BERT-Large와 동일한 아키텍처(24층, 1024 히든, 16 헤드, 355M 파라미터)를 사용한다. 학습 데이터는 BookCorpus+Wikipedia(16GB)에서 CC-News, OpenWebText, Stories를 추가하여 총 160GB로 확장했다. 배치 크기 8K, 학습 스텝 500K으로 설정하고, NSP를 제거하고, 동적 마스킹을 적용했다. 바이트 수준 BPE(50K 어휘)를 사용한다.",
    results: "RoBERTa는 GLUE 벤치마크에서 88.5점으로 XLNet-Large(88.4)를 소폭 능가하고 BERT-Large(82.3)를 크게 상회했다. SQuAD v1.1에서 94.6 F1, v2.0에서 89.4 F1을 달성했으며, RACE 독해에서도 83.2%로 당시 최고 성능을 기록했다. 이 모든 성과가 BERT 아키텍처를 그대로 사용하면서 학습 절차만 개선한 결과라는 점이 핵심이다.",
    impact: "RoBERTa는 '좋은 학습 레시피'의 중요성을 강력하게 입증한 연구이다. 모델 아키텍처 혁신만큼이나 학습 설정의 철저한 튜닝이 중요하다는 교훈은 이후 연구에 큰 영향을 미쳤다. RoBERTa 사전학습 체크포인트는 NLP 연구에서 가장 널리 사용되는 인코더 모델 중 하나로 자리잡았으며, 특히 분류, NER, 정보 추출 등의 인코더 기반 태스크에서 강력한 기준선으로 활용되고 있다.",
    relatedFoundations: ["transformer", "bert"],
    relatedPapers: [
      { id: "bert", fieldId: "foundations", title: "BERT: Pre-training of Deep Bidirectional Transformers", relation: "prior" },
      { id: "xlnet", fieldId: "nlp", title: "XLNet: Generalized Autoregressive Pretraining for Language Understanding", relation: "related" },
      { id: "deberta", fieldId: "nlp", title: "DeBERTa: Decoding-enhanced BERT with Disentangled Attention", relation: "successor" },
    ],
  },

  "deberta": {
    tldr: "콘텐츠와 위치 정보를 분리하여 어텐션하는 분리 어텐션(disentangled attention) 메커니즘과 향상된 마스크 디코더를 도입하여, BERT/RoBERTa를 크게 능가하고 SuperGLUE에서 최초로 인간 수준을 돌파한 DeBERTa를 제안했다.",
    background: "BERT 이후 RoBERTa, XLNet 등이 사전학습 방법론을 개선해왔으나, Transformer의 셀프 어텐션 메커니즘 자체를 근본적으로 개선하려는 시도는 상대적으로 적었다. 기존 Transformer는 콘텐츠 임베딩과 위치 임베딩을 합산한 뒤 어텐션을 수행하는데, 이는 두 종류의 정보가 혼재되어 각각의 기여를 효과적으로 모델링하기 어렵다는 한계가 있었다.",
    keyIdea: "DeBERTa(Decoding-enhanced BERT with disentangled Attention)는 두 가지 핵심 기법을 도입한다. 첫째, 분리 어텐션 메커니즘은 각 토큰을 콘텐츠 벡터와 위치 벡터의 두 가지로 표현하고, 이들 사이의 어텐션을 content-to-content, content-to-position, position-to-content의 세 가지 성분으로 분리하여 계산한다. 이를 통해 '단어의 의미'와 '단어의 위치'가 서로에게 미치는 영향을 명시적으로 모델링할 수 있다. 둘째, 향상된 마스크 디코더(EMD)는 사전학습 시 마스킹된 토큰을 예측할 때, 디코딩 레이어에서 절대 위치 정보를 결합한다. 상대 위치만으로는 부족한 구문적 정보(예: 문장 내 위치에 따른 역할)를 보완하기 위함이다.",
    method: "BERT/RoBERTa와 동일한 마스크드 언어 모델(MLM) 사전학습 목표를 사용한다. 분리 어텐션에서는 상대 위치 인코딩만 사용하고, 최종 디코딩 레이어에서만 절대 위치를 추가한다. 가상 적대적 학습(virtual adversarial training)을 미세조정 단계에 적용하여 모델의 강건성을 향상시킨다. 모델 크기는 Base(140M)와 Large(350M)에 더해, 1.5B 파라미터의 DeBERTa-XL도 제공한다.",
    results: "DeBERTa-Large는 RoBERTa-Large 및 XLNet-Large 대비 대부분의 NLU 벤치마크에서 우수한 성능을 달성했다. 특히 1.5B 파라미터의 DeBERTa는 SuperGLUE 벤치마크에서 90.0점을 달성하여, 인간 베이스라인(89.8)을 최초로 능가한 단일 모델이 되었다. SQuAD v2.0에서도 RoBERTa 대비 유의미한 성능 향상을 보였다.",
    impact: "DeBERTa는 어텐션 메커니즘의 근본적 개선이 사전학습 모델의 성능을 크게 향상시킬 수 있음을 보여주었다. 분리 어텐션이라는 개념은 이후 위치 인코딩 연구에 영감을 주었으며, DeBERTa-v3는 현재까지도 인코더 기반 NLU 태스크의 최강 모델 중 하나로 널리 활용되고 있다. HuggingFace에서 가장 많이 다운로드되는 인코더 모델 중 하나이다.",
    relatedFoundations: ["transformer", "bert"],
    relatedPapers: [
      { id: "bert", fieldId: "foundations", title: "BERT: Pre-training of Deep Bidirectional Transformers", relation: "prior" },
      { id: "roberta", fieldId: "nlp", title: "RoBERTa: A Robustly Optimized BERT Pretraining Approach", relation: "prior" },
      { id: "xlnet", fieldId: "nlp", title: "XLNet: Generalized Autoregressive Pretraining for Language Understanding", relation: "related" },
    ],
  },

  "flan-t5": {
    tldr: "1,800개 이상의 태스크에 대해 지시 미세조정(instruction finetuning)을 수행하면 모델의 성능과 사용성이 비약적으로 향상되며, 이 효과가 모델 크기, 태스크 수, 체인-오브-소트 데이터 포함에 따라 스케일링됨을 체계적으로 입증했다.",
    background: "GPT-3의 few-shot 학습 이후, InstructGPT와 같이 인간 피드백으로 모델을 정렬하는 연구와 FLAN, T0 등 다양한 태스크로 지시 미세조정하는 연구가 동시에 발전했다. 그러나 지시 미세조정의 스케일링 특성(태스크 수, 모델 크기, 데이터 구성의 영향)은 체계적으로 연구되지 않았다.",
    keyIdea: "Flan-T5(Flan 2022)는 지시 미세조정을 1,800개 이상의 태스크로 대폭 확장하고, 체인-오브-소트(CoT) 추론 데이터를 학습에 포함시키며, 입력 역전(input inversion)과 다양한 템플릿을 활용한다는 세 가지 핵심 개선을 도입한다. 학습 태스크를 Muffin(473개), T0-SF(193개), NIV2(1,554개), CoT 데이터셋의 네 가지 혼합물로 구성하고, 다양한 프롬프트 템플릿(zero-shot, few-shot, CoT)을 사용한다. 핵심 발견은 (1) 태스크 수를 늘리면 성능이 지속적으로 향상되고, (2) 모델 크기를 키우면 태스크 수 증가의 이점이 더 커지며, (3) CoT 데이터를 포함하면 추론 능력이 크게 향상된다는 것이다.",
    method: "T5(80M~11B)와 PaLM(8B~540B) 모델에 대해 혼합된 지시 미세조정을 수행한다. 학습 데이터는 각 태스크별 예제 수의 상한(exemplars cap)을 설정하여 균형을 맞추고, zero-shot, few-shot, CoT 템플릿을 섞어 학습한다. 평가는 학습에 포함되지 않은 held-out 태스크들(MMLU, BBH, TyDiQA, MGSM 등)로 수행한다.",
    results: "Flan-PaLM 540B는 MMLU에서 75.2%를 달성하여 기존 PaLM(69.3%)을 크게 능가했다. Flan-T5-XL(3B)은 여러 벤치마크에서 GPT-3(175B)를 능가하는 효율성을 보여주었다. BBH(BIG-Bench Hard)에서 CoT 프롬프팅 시 Flan-PaLM이 PaLM 대비 평균 8.1% 향상되었다. 특히 Flan 미세조정 모델이 후속 few-shot 및 CoT 프롬프팅에도 더 잘 반응하여, 기본 능력을 유지하면서 지시 따르기 능력이 향상됨을 확인했다.",
    impact: "Flan-T5는 지시 미세조정의 스케일링 법칙을 실증적으로 확립하여, 이후 Alpaca, Vicuna 등 오픈소스 지시 미세조정 모델 개발의 이론적 토대를 제공했다. 특히 Flan-T5 체크포인트는 공개되어 학술 및 산업계에서 가장 널리 사용되는 instruction-tuned 인코더-디코더 모델이 되었으며, 적은 자원으로도 강력한 태스크 수행이 가능함을 보여주었다.",
    relatedFoundations: ["transformer", "bert"],
    relatedPapers: [
      { id: "t5", fieldId: "nlp", title: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", relation: "prior" },
      { id: "instructgpt", fieldId: "nlp", title: "Training language models to follow instructions with human feedback", relation: "related" },
      { id: "cot", fieldId: "llm", title: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", relation: "related" },
    ],
  },

  "codex": {
    tldr: "GPT 언어 모델을 GitHub 코드로 미세조정한 Codex를 개발하여, 자연어 설명으로부터 프로그램을 자동 생성하는 능력을 크게 향상시키고, 이를 평가하기 위한 HumanEval 벤치마크를 제안했다.",
    background: "GPT-3 등 대규모 언어 모델은 자연어 처리에서 놀라운 성과를 보였으나, 프로그래밍과 같은 형식적 언어 생성에서는 한계가 있었다. 코드는 자연어와 달리 구문 오류에 엄격하고 실행 가능성이 검증 가능하다는 특성이 있어, 새로운 평가 방법론이 필요했다. 기존의 코드 생성 벤치마크는 주로 코드 완성이나 코드 검색에 초점을 맞추고 있었다.",
    keyIdea: "Codex는 GPT 모델을 GitHub에서 수집한 159GB의 Python 코드로 미세조정하여 코드 생성에 특화시킨 모델이다. 핵심 기여는 두 가지이다. 첫째, docstring(함수 설명)으로부터 올바른 함수 본문을 생성하는 능력을 평가하는 HumanEval 벤치마크(164개 프로그래밍 문제)를 제안했다. 이는 단순 문자열 매칭이 아닌 실제 유닛 테스트 통과 여부로 정확성을 평가한다. 둘째, 모델에서 여러 후보를 샘플링한 뒤 올바른 것을 선택하는 pass@k 메트릭을 도입했다. k개의 샘플 중 하나라도 테스트를 통과하면 성공으로 간주하는 이 메트릭은 코드 생성의 실용적 활용 시나리오를 반영한다.",
    method: "GPT-3 12B 모델을 기반으로 GitHub 공개 저장소에서 수집한 Python 파일(159GB)로 미세조정했다. 학습률, 토크나이저(추가 코드 토큰), 컨텍스트 길이(4096 토큰) 등을 코드 도메인에 맞게 조정했다. pass@k 평가를 위한 비편향 추정량을 제안하고, 후보 선택을 위해 docstring 로그 확률 기반 리랭킹과 클러스터링 방법도 탐구했다.",
    results: "Codex-12B는 HumanEval에서 pass@1 28.8%, pass@100 72.3%를 달성하여, GPT-3 175B의 pass@1 0%를 크게 능가했다. 반복 샘플링과 리랭킹을 결합하면 pass@1이 44.5%까지 향상되었다. GPT-J 6B도 코드 학습 후 HumanEval에서 유의미한 성능을 보여, 코드 특화 학습의 효과가 모델 크기에 걸쳐 일반적임을 확인했다.",
    impact: "Codex는 GitHub Copilot의 핵심 기술로 상용화되어 AI 기반 프로그래밍 보조 도구 시장을 개척했다. HumanEval 벤치마크는 코드 생성 모델 평가의 사실상 표준이 되었으며, pass@k 메트릭은 이후 모든 코드 LLM 연구에서 채택되었다. 이 연구는 LLM의 응용 범위를 코드 생성으로 확장하는 결정적 계기가 되었고, StarCoder, CodeLlama 등 후속 코드 모델 연구를 촉발했다.",
    relatedFoundations: ["transformer", "gpt", "gpt3"],
    relatedPapers: [
      { id: "gpt3", fieldId: "foundations", title: "Language Models are Few-Shot Learners", relation: "prior" },
      { id: "gpt4", fieldId: "llm", title: "GPT-4 Technical Report", relation: "successor" },
    ],
  },

  "palm": {
    tldr: "Google의 Pathways 시스템을 활용하여 540B 파라미터의 PaLM을 학습시키고, 수백 개의 언어 이해 및 생성 태스크에서 획기적 성능을 달성하며 대규모 모델에서만 나타나는 창발적 능력(emergent abilities)을 체계적으로 분석했다.",
    background: "GPT-3 이후 대규모 언어 모델의 스케일링이 활발히 진행되었으나, 수천 개의 TPU를 효율적으로 활용하는 학습 인프라의 한계와 모델 규모 증가에 따른 새로운 능력의 출현(emergence) 현상에 대한 체계적 이해가 부족했다. Chinchilla가 컴퓨트 최적 학습의 중요성을 보여주었으나, 가장 큰 모델에서만 가능한 능력의 경계는 아직 충분히 탐구되지 않았다.",
    keyIdea: "PaLM(Pathways Language Model)은 두 가지 핵심 기여를 한다. 첫째, Google의 Pathways 시스템을 통해 6,144개의 TPU v4 칩에서 효율적으로 540B 모델을 학습하여, 대규모 분산 학습의 새로운 표준을 제시한다. 둘째, 모델 규모에 따른 성능 변화를 8B, 62B, 540B 세 가지 크기에서 체계적으로 분석하여, 불연속적 성능 향상(discontinuous improvement)이 나타나는 태스크를 발견한다. 특히 수학적 추론, 상식 추론, 코드 생성 등에서 540B 모델이 62B 대비 비선형적으로 큰 성능 향상을 보이는 창발적 능력을 관찰했다.",
    method: "밀집(dense) Transformer 디코더 아키텍처에 SwiGLU 활성화, RoPE 위치 인코딩, multi-query 어텐션, 입력-출력 임베딩 공유 등을 적용했다. 780B 토큰(웹 문서, 책, Wikipedia, 코드, 대화 등 다국어 데이터)으로 학습했다. Pathways의 데이터 및 모델 병렬화를 결합하여 6,144개 TPU v4에서 57.8%의 하드웨어 효율(MFU)을 달성했다.",
    results: "PaLM 540B는 29개 NLU 벤치마크 중 28개에서 미세조정 없이 당시 최고 성능을 달성했다. BIG-Bench의 150개 태스크 중 다수에서 인간 평균을 상회했으며, GSM8K 수학 추론에서 chain-of-thought 프롬프팅으로 58.1%를 달성하여 미세조정 최고 성능과 비교 가능한 수준이었다. 코드 생성에서도 Codex와 경쟁 가능한 성능을 보여주었다.",
    impact: "PaLM은 대규모 밀집 모델의 스케일링이 여전히 효과적임을 보여주는 동시에, 창발적 능력이라는 개념을 실증적으로 정립하여 LLM 연구의 방향성에 큰 영향을 미쳤다. Pathways 시스템을 통한 대규모 분산 학습 방법론은 이후 Gemini 등 Google 후속 모델의 기반이 되었다. PaLM-2는 Google의 주력 LLM으로 발전하여 Bard(현 Gemini) 서비스의 핵심 엔진으로 활용되었다.",
    relatedFoundations: ["transformer", "gpt3", "scaling-laws"],
    relatedPapers: [
      { id: "chinchilla", fieldId: "llm", title: "Training Compute-Optimal Large Language Models", relation: "related" },
      { id: "cot", fieldId: "llm", title: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", relation: "related" },
      { id: "gpt4", fieldId: "llm", title: "GPT-4 Technical Report", relation: "related" },
    ],
  },

  "llama2": {
    tldr: "Meta가 7B~70B 규모의 사전학습 및 미세조정 LLM 컬렉션 Llama 2를 공개하고, 특히 RLHF로 최적화된 Llama 2-Chat이 대부분의 벤치마크에서 기존 오픈소스 채팅 모델을 능가하며 상용 모델과 경쟁 가능한 수준임을 보여주었다.",
    background: "LLaMA의 공개가 오픈소스 LLM 생태계를 촉발했으나, 원래 LLaMA는 연구 전용 라이선스로 상업적 활용이 제한되었고, 채팅에 최적화된 공식 버전도 없었다. Alpaca, Vicuna 등 커뮤니티 파인튜닝 모델이 등장했지만, 안전성과 유용성 면에서 체계적 최적화가 부족했다. 상업적으로 활용 가능하면서 안전한 오픈소스 채팅 모델에 대한 수요가 컸다.",
    keyIdea: "Llama 2는 세 가지 핵심 개선을 도입한다. 첫째, 사전학습 데이터를 1.4조에서 2조 토큰으로 40% 확장하고 문맥 길이를 2048에서 4096으로 두 배 늘렸다. 둘째, RLHF를 통해 Llama 2-Chat을 개발하며, 기존의 단일 보상 모델 대신 유용성(helpfulness)과 안전성(safety) 두 가지 별도 보상 모델을 학습시키는 이중 보상 모델 접근법을 채택했다. 셋째, rejection sampling과 PPO를 반복적으로 적용하는 RLHF 파이프라인을 통해 점진적으로 모델을 개선하며, Ghost Attention(GAtt) 기법으로 다회전 대화에서의 시스템 프롬프트 일관성을 유지한다.",
    method: "표준 Transformer 아키텍처에 RMSNorm, SwiGLU, GQA(Grouped-Query Attention, 34B/70B에서), RoPE를 적용했다. SFT 단계에서 27,540개의 고품질 시연 데이터를 사용하고, RLHF 단계에서 100만개 이상의 인간 비교 데이터로 보상 모델을 학습했다. Rejection sampling(RS)으로 높은 보상 점수의 응답을 선별한 뒤 추가 SFT를 수행하고, 이후 PPO로 정책을 최적화하는 과정을 여러 반복(iteration) 수행했다.",
    results: "Llama 2-Chat 70B는 인간 평가에서 ChatGPT에 근접한 유용성 점수를 달성하고, 안전성에서는 ChatGPT를 상회했다. MMLU에서 Llama 2 70B는 68.9%, Llama 2-Chat 70B는 63.9%를 기록했다. 오픈소스 모델 중에서는 MPT, Falcon 등을 대부분의 벤치마크에서 능가했다. 특히 안전성 평가에서 유해 응답 비율이 RLHF 반복 횟수에 따라 지속적으로 감소하는 것을 확인했다.",
    impact: "Llama 2는 상업적 사용이 가능한 라이선스로 공개되어 오픈소스 LLM의 산업적 활용을 본격화한 이정표적 모델이다. RLHF 파이프라인의 상세한 기술을 공개하여 안전한 채팅 모델 개발의 로드맵을 제공했다. 이후 Code Llama, Llama 3 등으로 이어지는 Meta의 오픈소스 LLM 전략의 핵심 모델로서, 기업과 연구기관의 LLM 자체 구축 역량을 비약적으로 높였다.",
    relatedFoundations: ["transformer", "gpt", "scaling-laws"],
    relatedPapers: [
      { id: "llama", fieldId: "llm", title: "LLaMA: Open and Efficient Foundation Language Models", relation: "prior" },
      { id: "instructgpt", fieldId: "nlp", title: "Training language models to follow instructions with human feedback", relation: "prior" },
      { id: "gpt4", fieldId: "llm", title: "GPT-4 Technical Report", relation: "related" },
    ],
  },

  "mixtral": {
    tldr: "희소 전문가 혼합(Sparse Mixture of Experts) 아키텍처를 활용한 Mixtral 8x7B를 공개하여, 각 토큰마다 8개 전문가 중 2개만 활성화함으로써 GPT-3.5 수준의 성능을 13B 파라미터 활성화 비용으로 달성했다.",
    background: "대규모 언어 모델의 성능은 파라미터 수에 따라 향상되지만, 추론 비용도 비례하여 증가한다는 근본적 트레이드오프가 있었다. Mixture of Experts(MoE) 아키텍처는 총 파라미터 수를 늘리되 각 입력에 대해 일부만 활성화하는 방식으로 이 트레이드오프를 완화할 수 있는 잠재력이 있었다. GShard, Switch Transformer 등이 MoE를 탐구했으나, 오픈소스 고성능 MoE 모델은 부재했다.",
    keyIdea: "Mixtral 8x7B는 각 Transformer 레이어의 피드포워드 블록을 8개의 전문가(expert) 네트워크로 대체하고, 라우터(router) 네트워크가 각 토큰에 대해 상위 2개 전문가를 선택하여 출력을 가중합산하는 Sparse Mixture of Experts 구조를 채택한다. 총 파라미터는 46.7B이지만, 각 토큰 처리 시 실제 활성화되는 파라미터는 약 12.9B에 불과하다. 이를 통해 밀집 모델 대비 동일한 추론 속도에서 훨씬 더 많은 지식을 모델에 저장할 수 있다. 라우터의 전문가 선택은 학습 과정에서 자동으로 최적화되며, 특정 전문가가 특정 유형의 토큰이나 주제에 특화되는 경향이 관찰된다.",
    method: "Mistral 7B를 기반으로 각 레이어의 FFN을 8개 전문가로 대체했다. 라우터는 소프트맥스 게이팅 함수로 구현되어 상위 2개 전문가를 선택한다. 전문가 병렬화를 통해 8개 전문가를 서로 다른 GPU에 분산 배치할 수 있다. 32K 토큰의 문맥 길이를 지원하며, Sliding Window Attention을 활용한다. DPO(Direct Preference Optimization)로 정렬한 Mixtral 8x7B-Instruct 버전도 함께 공개했다.",
    results: "Mixtral 8x7B는 MMLU에서 70.6%, HellaSwag 84.4%, ARC-Challenge 66.4%를 달성하여 Llama 2 70B(69.8%, 87.3%, 67.3%)와 동등하거나 근소한 차이를 보이면서 추론 속도는 6배 빠르다. 코드 생성(HumanEval 40.2%), 수학(GSM8K 74.4%) 등에서도 경쟁력 있는 성능을 보였다. GPT-3.5 Turbo와 비교 시 대부분의 벤치마크에서 동등하거나 우수한 결과를 달성했다.",
    impact: "Mixtral은 오픈소스 MoE 언어 모델의 실용성을 대규모로 입증하여, 효율적 LLM 아키텍처 연구에 새로운 방향을 제시했다. Mistral AI가 오픈소스 LLM 분야의 주요 플레이어로 부상하는 계기가 되었으며, 이후 DeepSeek-MoE, Qwen-MoE, DBRX 등 다양한 MoE 모델 개발을 촉진했다. 추론 효율성과 성능의 균형이라는 실용적 관점에서 LLM 아키텍처 선택의 패러다임을 변화시켰다.",
    relatedFoundations: ["transformer", "scaling-laws"],
    relatedPapers: [
      { id: "llama", fieldId: "llm", title: "LLaMA: Open and Efficient Foundation Language Models", relation: "prior" },
      { id: "llama2", fieldId: "llm", title: "Llama 2: Open Foundation and Fine-Tuned Chat Models", relation: "prior" },
      { id: "gemma", fieldId: "llm", title: "Gemma: Open Models Based on Gemini Research and Technology", relation: "related" },
    ],
  },

  "gemma": {
    tldr: "Google DeepMind가 Gemini 연구 기술을 기반으로 2B와 7B 규모의 경량 오픈 모델 Gemma를 공개하여, 동일 크기 대비 최고 수준의 성능을 달성하고 책임 있는 AI 개발 도구킷을 함께 제공했다.",
    background: "Llama 2, Mistral 등 오픈소스 LLM이 큰 성공을 거두면서 Google도 자사의 최첨단 연구 기술을 기반으로 한 오픈 모델 공개의 필요성을 인식했다. 특히 2B와 7B 같은 작은 규모의 모델은 개인 기기나 제한된 리소스 환경에서 활용 가능하여, 연구 접근성과 실용적 배포 측면에서 중요했다. 동시에 오픈 모델의 안전한 배포를 위한 체계적 프레임워크도 필요했다.",
    keyIdea: "Gemma는 Gemini 모델과 동일한 연구 및 기술을 활용하되, 2B와 7B의 작은 규모에 최적화된 오픈 모델이다. 핵심 특징은 세 가지이다. 첫째, 6조 토큰이라는 방대한 학습 데이터(주로 영어 웹 문서, 코드, 수학)를 사용하여 Chinchilla 최적 대비 훨씬 많은 데이터로 과잉 학습(over-training)시켜 작은 모델의 성능을 극대화한다. 둘째, Gemini 아키텍처의 개선 사항(Multi-Query Attention(2B), Grouped-Query Attention(7B), RoPE, GeGLU 활성화, RMSNorm)을 적용한다. 셋째, 사전학습 모델과 함께 지시 미세조정(instruction-tuned) 모델, 안전성 분류기, 디버깅 도구를 포함한 Responsible Generative AI Toolkit을 함께 공개한다.",
    method: "디코더 전용 Transformer 아키텍처를 기반으로, 2B 모델은 18층/2048 히든/8 헤드/256K 어휘, 7B 모델은 28층/3072 히든/16 헤드/256K 어휘로 구성했다. 6T 토큰의 데이터를 SentencePiece 토크나이저(256K 어휘)로 처리하고, 8192 토큰 문맥 길이로 학습했다. 지시 미세조정 모델은 SFT와 RLHF를 적용하여 개발했으며, 안전성 필터링과 레드팀 평가를 거쳤다.",
    results: "Gemma 7B는 MMLU에서 64.3%, HellaSwag 81.2%, GSM8K 46.4%를 달성하여, 동일 크기의 Llama 2 7B(45.3%, 77.2%, 14.6%)와 Mistral 7B(62.5%, 81.0%, 37.8%)를 크게 능가했다. Gemma 2B도 MMLU 42.3%로 2B 규모 모델 중 최고 성능을 기록했다. 코드 생성(HumanEval), 수학 추론, 상식 추론 등 전 영역에서 동일 크기 대비 우수한 결과를 보여주었다.",
    impact: "Gemma는 Google DeepMind의 첨단 기술을 오픈 생태계로 가져온 의미 있는 첫 걸음이다. 작은 모델도 충분한 데이터와 좋은 아키텍처로 강력한 성능을 달성할 수 있음을 재확인하며, 특히 온디바이스 AI와 리소스 제약 환경에서의 LLM 활용 가능성을 넓혔다. Gemma 2, CodeGemma, PaliGemma 등으로 이어지는 Gemma 패밀리의 시작점이며, 책임 있는 AI 도구킷의 동시 공개는 안전한 오픈소스 AI 배포의 모범 사례를 제시했다.",
    relatedFoundations: ["transformer", "scaling-laws"],
    relatedPapers: [
      { id: "llama2", fieldId: "llm", title: "Llama 2: Open Foundation and Fine-Tuned Chat Models", relation: "related" },
      { id: "mixtral", fieldId: "llm", title: "Mixtral of Experts", relation: "related" },
      { id: "chinchilla", fieldId: "llm", title: "Training Compute-Optimal Large Language Models", relation: "prior" },
    ],
  },
};
