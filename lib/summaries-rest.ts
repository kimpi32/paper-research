import type { PaperSummary } from "./paper-summaries";

export const restSummaries: Record<string, PaperSummary> = {
  // ===== Safety Field =====
  "constitutional-ai": {
    tldr: "헌법적 원칙(Constitution)을 기반으로 AI가 스스로 출력을 평가·수정하게 하여, 인간 라벨러 없이도 무해하고 유용한 AI를 훈련하는 RLAIF 방법론을 제시한 논문.",
    background:
      "RLHF는 인간 피드백에 크게 의존하지만 유해성 판단을 위한 인간 라벨링은 비용이 높고 일관성이 떨어질 수 있다. 또한 인간 라벨러가 모델의 교묘한 유해 출력을 놓칠 수 있어 확장성에 한계가 있었다. 이에 원칙 기반의 자동화된 피드백 체계가 요구되었다.",
    keyIdea:
      "Constitutional AI는 두 단계로 구성된다. 첫째, 'Critique → Revision' 단계에서 모델이 자신의 출력을 헌법적 원칙에 따라 비판하고 수정한다. 둘째, RLAIF(Reinforcement Learning from AI Feedback) 단계에서 AI가 생성한 선호 데이터로 보상 모델을 학습하여 PPO를 수행한다. 핵심은 '무해성', '정직성' 등의 원칙을 명시적 텍스트로 제공하여 모델이 스스로 판단 기준을 내재화하도록 한 것이다. 이를 통해 인간 라벨러 없이도 무해성을 크게 향상시키면서 유용성은 유지할 수 있었다.",
    method:
      "SFT 모델에서 유해한 응답을 유도한 뒤, 헌법 원칙에 따라 self-critique와 revision을 반복하여 개선된 데이터를 생성한다. 이 데이터로 SFT를 수행한 후, AI가 쌍별 비교로 선호도를 매겨 보상 모델을 학습하고 PPO로 최종 정책을 최적화한다.",
    results:
      "인간 피드백 없이 학습한 모델이 RLHF 모델과 동등하거나 더 나은 무해성을 달성했다. 동시에 유용성(helpfulness)은 유지되어 무해성-유용성 트레이드오프를 효과적으로 완화했다.",
    impact:
      "RLAIF 패러다임을 개척하여 AI 정렬(alignment)의 확장성을 크게 높였다. 이후 Claude 시리즈를 비롯한 상용 AI 시스템의 안전성 파이프라인에 직접적 영향을 미쳤으며, 원칙 기반 자기 개선이라는 새로운 연구 방향을 열었다.",
    relatedFoundations: ["gpt3"],
    relatedPapers: [
      { id: "rlhf", fieldId: "safety", title: "RLHF", relation: "related" },
      { id: "interpretability-circuits", fieldId: "safety", title: "Interpretability Circuits", relation: "related" },
    ],
  },

  "rlhf": {
    tldr: "인간 피드백 기반 강화학습(RLHF) 파이프라인을 체계화하여, 유용하면서도 무해한 AI 어시스턴트를 훈련하는 방법을 실증적으로 분석한 논문.",
    background:
      "대규모 언어모델은 뛰어난 능력에도 불구하고 유해하거나 부정확한 출력을 생성할 수 있다. 단순한 지시 미세조정만으로는 인간의 선호를 충분히 반영하기 어려웠으며, 유용성(helpfulness)과 무해성(harmlessness)을 동시에 최적화하는 체계적 방법론이 필요했다.",
    keyIdea:
      "SFT(Supervised Fine-Tuning) → 보상 모델(Reward Model) 학습 → PPO 강화학습의 3단계 파이프라인을 제시했다. 인간 평가자가 모델 응답 쌍에 대해 선호도를 매기고, 이를 기반으로 보상 모델을 학습한다. 유용성과 무해성 각각에 대해 별도의 선호 데이터를 수집하여, 두 목표 간의 파레토 프론티어(Pareto frontier)를 분석했다. 모델 크기가 커질수록 RLHF의 효과가 더 크게 나타남을 발견했다.",
    method:
      "52B 파라미터 모델에 대해 인간 선호 데이터를 수집하고, 유용성과 무해성 각각에 대한 보상 모델을 훈련했다. 이후 PPO를 적용하되, 보상 모델의 가중치를 조절하여 두 목표 간 트레이드오프를 탐색했다.",
    results:
      "RLHF 훈련 후 모델은 유용성과 무해성 모두에서 SFT 모델을 크게 앞섰다. 특히 대형 모델에서 alignment tax(정렬 비용)가 낮아지는 경향을 확인했으며, 유용성-무해성의 파레토 최적 조합을 달성할 수 있음을 보였다.",
    impact:
      "RLHF를 AI 안전 분야의 표준 훈련 파이프라인으로 확립했다. InstructGPT, ChatGPT, Claude 등 현대 대화형 AI 시스템의 기반 방법론이 되었으며, 이후 Constitutional AI 등 더 발전된 정렬 기법의 토대가 되었다.",
    relatedFoundations: ["gpt3"],
    relatedPapers: [
      { id: "constitutional-ai", fieldId: "safety", title: "Constitutional AI", relation: "successor" },
      { id: "ppo", fieldId: "rl", title: "PPO", relation: "prior" },
      { id: "instructgpt", fieldId: "nlp", title: "InstructGPT", relation: "related" },
    ],
  },

  "interpretability-circuits": {
    tldr: "희소 오토인코더(Sparse Autoencoder)를 사용하여 언어모델의 MLP 활성화를 해석 가능한 단일의미(monosemantic) 특징으로 분해하는 데 성공한 논문.",
    background:
      "신경망의 개별 뉴런은 대부분 다의적(polysemantic)이어서 여러 개념이 하나의 뉴런에 혼재되어 있다. 이는 중첩(superposition) 현상으로 설명되는데, 모델이 뉴런 수보다 더 많은 특징을 표현하기 위해 특징들을 겹쳐 저장하기 때문이다. 이를 해소하여 해석 가능한 특징을 추출하는 것이 해석 가능성 연구의 핵심 과제였다.",
    keyIdea:
      "1층짜리 트랜스포머의 MLP 활성화에 희소 오토인코더를 적용하여, 뉴런보다 훨씬 많은 수의 해석 가능한 특징을 추출했다. 각 특징은 단일 개념(예: DNA 서열, 히브리어 텍스트, 수학 표현 등)에 대응하는 단일의미적 성질을 보였다. 특징의 활성화 패턴, 빈도, 상호 관계를 체계적으로 분석하여 해석 가능성과 충실성(faithfulness) 사이의 관계를 규명했다. 특히 사전의 크기를 늘릴수록 더 세밀하고 해석 가능한 특징이 나타남을 확인했다.",
    method:
      "512차원 MLP 활성화에 대해 다양한 크기(512~131072)의 희소 오토인코더를 학습했다. 학습 목표는 재구성 오류 최소화와 활성화 희소성 유지의 균형이다. 추출된 각 특징에 대해 자동화된 해석 가능성 점수를 매기고, 인간 평가와 비교하여 검증했다.",
    results:
      "추출된 특징의 대부분이 인간이 이해할 수 있는 명확한 의미를 가지고 있었다. 특징은 깔끔한 빈도-크기 분포를 보이며, 유사 의미의 특징들이 코사인 유사도 공간에서 클러스터를 형성했다. 더 큰 사전이 더 세밀한 해석을 가능하게 함을 입증했다.",
    impact:
      "중첩 가설(superposition hypothesis)을 실증적으로 뒷받침한 획기적 연구로, 대규모 언어모델 해석 가능성 분야의 새로운 방법론적 기준을 세웠다. 이후 더 큰 모델에 대한 확장 연구로 이어지며, AI 안전성의 근본적 이해에 기여하고 있다.",
    relatedFoundations: ["transformer"],
    relatedPapers: [
      { id: "constitutional-ai", fieldId: "safety", title: "Constitutional AI", relation: "related" },
    ],
  },

  // ===== Optimization Field =====
  "adamw": {
    tldr: "Adam 옵티마이저에서 가중치 감쇠(weight decay)와 L2 정규화의 등가성이 깨지는 문제를 규명하고, 이를 분리(decoupled)하여 AdamW를 제안한 논문.",
    background:
      "SGD에서는 가중치 감쇠와 L2 정규화가 수학적으로 동치이지만, Adam과 같은 적응적 학습률 옵티마이저에서는 그렇지 않다. 기존 구현들은 이 둘을 혼용하여 최적이 아닌 정규화 효과를 초래했으며, 이로 인해 Adam이 SGD+모멘텀 대비 일반화 성능이 떨어진다는 인식이 있었다.",
    keyIdea:
      "Adam에서 L2 정규화 항은 그래디언트에 추가된 뒤 적응적 학습률로 스케일링되므로, 파라미터마다 감쇠 강도가 달라져 의도한 정규화 효과를 왜곡한다. AdamW는 가중치 감쇠를 그래디언트 업데이트와 완전히 분리하여, 학습률 스케줄에 관계없이 일정한 비율로 가중치를 줄인다. 추가로 학습률 워밍업 스케줄과의 분리(decoupled)도 제안하여 하이퍼파라미터 탐색 공간을 단순화했다.",
    method:
      "Adam 업데이트 규칙에서 L2 페널티를 그래디언트 계산에 포함시키는 대신, 파라미터 업데이트 이후 별도로 가중치 감쇠를 적용한다. 이를 CIFAR-10과 ImageNet-32x32에서 SGD+모멘텀, 기존 Adam과 비교 실험했다.",
    results:
      "AdamW는 기존 Adam 대비 일반화 성능이 크게 향상되어 SGD+모멘텀과 동등하거나 더 나은 결과를 보였다. 학습률과 가중치 감쇠를 독립적으로 튜닝할 수 있어 하이퍼파라미터 탐색이 효율적이 되었다.",
    impact:
      "사실상 모든 현대 트랜스포머 훈련의 기본 옵티마이저가 되었다. BERT, GPT 시리즈, ViT 등 거의 모든 대규모 모델 훈련에 AdamW가 채택되었으며, 적응적 옵티마이저의 정규화에 대한 이해를 근본적으로 바꾸었다.",
    relatedFoundations: ["adam"],
    relatedPapers: [
      { id: "lion", fieldId: "optimization", title: "Lion", relation: "successor" },
      { id: "lottery-ticket", fieldId: "optimization", title: "Lottery Ticket", relation: "related" },
    ],
  },

  "lottery-ticket": {
    tldr: "밀집 신경망 내부에 '당첨 복권(winning ticket)'과 같은 희소 부분 네트워크가 존재하며, 이를 초기 가중치부터 독립적으로 학습해도 원래 네트워크의 성능에 도달할 수 있다는 가설을 제시한 논문.",
    background:
      "신경망 가지치기(pruning)는 학습 후 파라미터를 제거하여 모델을 압축하는 기법으로 널리 사용되어 왔다. 그러나 가지치기된 구조를 처음부터 학습하면 원래 성능에 미치지 못하는 것이 일반적이었다. 이는 밀집 네트워크의 과잉 파라미터화가 학습에 본질적으로 필요한 것인지에 대한 근본적 질문을 제기했다.",
    keyIdea:
      "복권 가설(Lottery Ticket Hypothesis)은 무작위로 초기화된 밀집 네트워크에 특정 초기화 값과 구조의 조합으로 이루어진 '당첨 복권' 부분 네트워크가 존재한다고 주장한다. 이 부분 네트워크를 원래의 초기 가중치로 리셋한 후 학습하면, 전체 네트워크와 동등한 성능을 유사하거나 더 적은 반복 횟수로 달성한다. 반복적 가지치기(iterative magnitude pruning)를 통해 이러한 부분 네트워크를 찾을 수 있으며, 원래 파라미터의 10-20%만으로 충분한 경우가 많다.",
    method:
      "밀집 네트워크를 학습한 뒤 가중치 크기가 작은 연결을 제거하고, 남은 연결의 가중치를 초기값으로 되돌린다. 이 과정을 반복하여 점점 더 작은 부분 네트워크를 추출한다. MNIST와 CIFAR-10에서 완전연결망과 CNN으로 실험했다.",
    results:
      "발견된 당첨 복권 부분 네트워크는 원래 네트워크 대비 10-20% 크기에서도 동등하거나 더 나은 테스트 정확도를 달성했다. 무작위 재초기화 시에는 성능이 크게 떨어져, 초기 가중치 값 자체가 중요함을 확인했다.",
    impact:
      "신경망의 과잉 파라미터화에 대한 근본적 이해를 제공하여, 효율적 학습과 모델 압축 연구에 큰 영향을 미쳤다. 네트워크 구조 탐색, 가지치기 이론, 희소 학습 등 후속 연구의 촉매제가 되었으며, NeurIPS 2019 Best Paper Award를 수상했다.",
    relatedFoundations: ["dropout", "backpropagation"],
    relatedPapers: [
      { id: "lora", fieldId: "efficient", title: "LoRA", relation: "related" },
      { id: "distillation", fieldId: "efficient", title: "Knowledge Distillation", relation: "related" },
    ],
  },

  "lion": {
    tldr: "프로그램 탐색을 통해 발견된 Lion 옵티마이저는 부호(sign) 기반 업데이트만 사용하여 Adam보다 단순하고 메모리 효율적이면서 동등 이상의 성능을 달성한다.",
    background:
      "Adam과 그 변형들이 딥러닝 최적화의 표준이 되었지만, 이들이 이론적으로 최적인지는 불명확했다. 수작업 설계 대신 자동 탐색으로 더 나은 옵티마이저를 발견할 수 있다는 아이디어가 있었으나, 탐색 공간이 방대하여 실용적 적용이 어려웠다.",
    keyIdea:
      "진화적 프로그램 탐색(evolutionary program search)을 활용하여 수백만 개의 옵티마이저 후보 프로그램을 평가한 결과, Lion(EvoLved Sign Momentum)이 발견되었다. Lion은 놀랍도록 단순한 구조를 가지며, 업데이트 방향으로 그래디언트와 모멘텀의 보간(interpolation)에 sign 함수를 적용한다. 모든 파라미터에 동일한 크기의 업데이트가 적용되므로 적응적 2차 모멘트 추정이 불필요하다. 이로 인해 Adam 대비 메모리 사용량이 약 절반으로 줄어든다.",
    method:
      "AutoML-Zero 스타일의 프로그램 탐색 공간에서 진화 알고리즘으로 옵티마이저를 탐색했다. 소규모 작업에서 발견된 최적 프로그램을 분석·정규화하여 Lion을 추출하고, ImageNet, JFT-300M, 다양한 NLP 벤치마크에서 대규모 검증을 수행했다.",
    results:
      "Lion은 ImageNet에서 AdamW 대비 동등하거나 더 나은 정확도를 달성하면서 메모리를 절약했다. 특히 대규모 배치와 대형 모델에서 이점이 두드러졌으며, 비전-언어 모델 학습에서도 우수한 성능을 보였다.",
    impact:
      "옵티마이저 자동 발견이라는 새로운 패러다임을 실증하여 최적화 알고리즘 설계의 지평을 넓혔다. Sign 기반 업데이트의 효과를 재조명하고, 메모리 제약이 큰 대규모 모델 학습에서 실용적 대안을 제시했다.",
    relatedFoundations: ["adam"],
    relatedPapers: [
      { id: "adamw", fieldId: "optimization", title: "AdamW", relation: "prior" },
    ],
  },

  // ===== Representation Learning Field =====
  "simclr": {
    tldr: "데이터 증강, 대규모 배치, 비선형 프로젝션 헤드를 결합한 간단한 대조 학습 프레임워크로, 라벨 없이도 강력한 시각 표현을 학습할 수 있음을 보인 논문.",
    background:
      "자기지도 시각 표현 학습은 라벨 의존도를 줄이는 핵심 과제였지만, 기존 방법들은 pretext task 설계에 의존하거나 지도학습 대비 큰 성능 격차가 있었다. 대조 학습(contrastive learning)이 유망한 방향으로 떠올랐으나, 어떤 구성요소가 핵심인지 체계적 분석이 부족했다.",
    keyIdea:
      "SimCLR은 같은 이미지의 두 가지 증강 뷰를 긍정 쌍(positive pair)으로, 배치 내 다른 이미지들을 부정 쌍(negative pair)으로 사용하는 대조 학습 프레임워크이다. 핵심 발견으로, (1) 랜덤 크롭과 색상 변환의 조합이 가장 효과적인 증강이며, (2) 인코더 뒤의 비선형 프로젝션 헤드가 표현 품질을 크게 향상시키고, (3) 대규모 배치와 긴 학습이 더 많은 부정 예시를 제공하여 성능을 높인다는 것을 밝혔다.",
    method:
      "ResNet 인코더로 두 증강 뷰의 표현을 추출하고, 2층 MLP 프로젝션 헤드를 통과시킨 뒤 NT-Xent(정규화된 온도 크로스엔트로피) 손실로 학습한다. 배치 크기 4096~8192, 100~1000 에폭의 대규모 학습을 수행했다.",
    results:
      "ImageNet 선형 평가에서 76.5% top-1 정확도를 달성하여 이전 자기지도 방법을 7% 이상 앞섰다. 라벨의 1%만 사용한 준지도 학습에서도 지도학습의 85% 이상 성능을 달성했다.",
    impact:
      "대조 학습의 핵심 요소를 체계적으로 규명하여 자기지도 시각 표현 학습의 르네상스를 이끌었다. BYOL, DINO 등 후속 연구의 기반이 되었으며, CLIP 등 멀티모달 학습에도 대조 학습 원리가 확산되는 데 기여했다.",
    relatedFoundations: ["resnet", "batch-normalization"],
    relatedPapers: [
      { id: "byol", fieldId: "representation", title: "BYOL", relation: "successor" },
      { id: "dino", fieldId: "representation", title: "DINO", relation: "successor" },
      { id: "clip", fieldId: "multimodal", title: "CLIP", relation: "successor" },
    ],
  },

  "byol": {
    tldr: "부정 쌍(negative pairs) 없이도 대조 학습에 필적하는 자기지도 시각 표현을 학습할 수 있음을 보인 논문으로, 온라인/타겟 네트워크 구조와 EMA를 활용한다.",
    background:
      "SimCLR 등 대조 학습 방법은 부정 예시를 필요로 하며, 이를 위해 대규모 배치나 메모리 뱅크가 필수적이었다. 부정 예시 없이 학습하면 모든 입력을 같은 표현으로 매핑하는 붕괴(collapse)가 발생하는 것이 상식이었다. 이러한 제약을 극복하는 방법이 요구되었다.",
    keyIdea:
      "BYOL은 온라인 네트워크(online network)와 타겟 네트워크(target network)의 두 네트워크를 사용한다. 온라인 네트워크는 타겟 네트워크의 출력을 예측하도록 학습되고, 타겟 네트워크는 온라인 네트워크의 지수이동평균(EMA)으로 느리게 업데이트된다. 온라인 네트워크에만 추가 예측기(predictor)를 두어 비대칭성을 만들고, 이것이 표현 붕괴를 방지하는 핵심 역할을 한다. 부정 쌍을 완전히 제거함으로써 배치 크기와 증강 전략에 대한 민감도를 크게 줄였다.",
    method:
      "두 증강 뷰를 각각 온라인과 타겟 네트워크에 통과시키고, 온라인 네트워크의 예측기 출력이 타겟의 프로젝션 출력과 일치하도록 MSE 손실을 최소화한다. 타겟 네트워크는 EMA(τ=0.996)로 업데이트하며, 대칭적으로 뷰를 교환하여 손실을 합산한다.",
    results:
      "ImageNet 선형 평가에서 74.3% top-1 정확도로 SimCLR(69.3%)를 크게 앞섰다. 배치 크기 변화에 강건하여 256 배치에서도 성능 저하가 적었으며, 전이 학습에서도 우수한 성능을 보였다.",
    impact:
      "부정 예시 없는 자기지도 학습의 가능성을 입증하여 대조 학습의 패러다임을 확장했다. EMA 기반 타겟 네트워크 구조는 DINO 등 후속 연구의 핵심 설계 요소가 되었으며, 표현 붕괴 방지 메커니즘에 대한 활발한 이론적 연구를 촉발했다.",
    relatedFoundations: ["resnet", "batch-normalization"],
    relatedPapers: [
      { id: "simclr", fieldId: "representation", title: "SimCLR", relation: "prior" },
      { id: "dino", fieldId: "representation", title: "DINO", relation: "successor" },
    ],
  },

  "dino": {
    tldr: "비전 트랜스포머(ViT)에 자기증류(self-distillation)를 적용하면 라벨 없이도 명시적 세그멘테이션 정보가 자연스럽게 출현함을 발견한 논문.",
    background:
      "자기지도 학습은 CNN 기반에서 큰 성과를 보였지만, ViT에 대한 적용은 초기 단계였다. ViT가 CNN과 다른 특성(어텐션 맵, 글로벌 수용장 등)을 가지므로, 자기지도 학습에서 어떤 고유한 성질이 나타나는지 탐구할 필요가 있었다.",
    keyIdea:
      "DINO(Self-DIstillation with NO labels)는 학생-교사 프레임워크에서 두 네트워크 모두 같은 구조를 사용하되, 교사는 학생의 EMA로 업데이트된다. 학생은 로컬 크롭(작은 영역)을, 교사는 글로벌 크롭(넓은 영역)을 입력받아, 학생이 교사의 출력 분포를 맞추도록 학습한다. 핵심 발견은 이렇게 학습된 ViT의 자기어텐션 맵이 사전 학습만으로 객체의 세그멘테이션 경계를 정확하게 포착한다는 것이다. 또한 출력 센터링(centering)과 샤프닝(sharpening)으로 모드 붕괴를 방지한다.",
    method:
      "멀티크롭 전략으로 2개의 글로벌 뷰(224x224)와 여러 로컬 뷰(96x96)를 생성한다. 교사-학생 출력의 크로스엔트로피 손실을 최소화하되, 교사 출력에 센터링을 적용하여 붕괴를 방지한다. ViT-S/16부터 ViT-B/8까지 다양한 규모로 실험했다.",
    results:
      "ImageNet 선형 평가에서 ViT-B/8 기준 80.1% top-1 정확도를 달성했다. [CLS] 토큰의 k-NN 분류도 78.3%에 달해 별도 학습 없이도 강력한 표현을 형성함을 입증했다. 어텐션 맵의 자동 세그멘테이션 품질이 지도학습 모델을 능가했다.",
    impact:
      "ViT의 자기지도 학습에서 출현하는 성질을 최초로 체계적으로 분석하여, 기초 모델(foundation model) 시대의 표현 학습 방향을 제시했다. DINOv2로 발전하며 다양한 비전 태스크의 범용 백본이 되었고, MAE 등과 함께 비전 자기지도 학습의 양대 축을 형성했다.",
    relatedFoundations: ["vit"],
    relatedPapers: [
      { id: "simclr", fieldId: "representation", title: "SimCLR", relation: "prior" },
      { id: "byol", fieldId: "representation", title: "BYOL", relation: "prior" },
      { id: "mae", fieldId: "representation", title: "MAE", relation: "related" },
    ],
  },

  "mae": {
    tldr: "이미지 패치의 75%를 마스킹하고 비대칭 인코더-디코더로 복원하는 마스크 오토인코더(MAE)가 확장 가능하고 강력한 비전 자기지도 학습 방법임을 보인 논문.",
    background:
      "NLP에서 BERT의 마스크 언어 모델링이 큰 성공을 거두었지만, 비전에서는 마스킹 기반 사전학습이 대조 학습 대비 뒤처져 있었다. 이미지는 텍스트와 달리 공간적 중복성이 높고 연속적이어서, 단순한 마스킹으로는 의미 있는 표현을 학습하기 어렵다는 인식이 있었다.",
    keyIdea:
      "MAE의 핵심 통찰은 이미지의 높은 공간적 중복성을 역으로 활용하는 것이다. 무려 75%의 패치를 마스킹하면 남은 25%로는 저수준 보간이 불가능하여 모델이 고수준 의미를 이해해야만 복원할 수 있다. 비대칭 설계에서 인코더는 보이는 패치만 처리하고(75% 연산량 절감), 가벼운 디코더가 마스크 토큰과 함께 전체를 복원한다. 이 설계 덕분에 학습이 매우 효율적이며, 큰 ViT 모델로의 확장이 용이하다.",
    method:
      "ViT 인코더에 보이는 패치(25%)만 입력하고, 디코더에서 마스크 토큰을 추가하여 원본 픽셀을 MSE 손실로 복원한다. 디코더는 인코더보다 훨씬 작은 8층 트랜스포머를 사용한다. ImageNet-1K에서 1600 에폭 사전학습 후 미세조정했다.",
    results:
      "ViT-H/14로 ImageNet 미세조정에서 87.8% top-1 정확도를 달성하여, 지도학습과 대조 학습 기반 방법을 모두 능가했다. 사전학습이 3.5배 이상 빨라 대규모 ViT 학습의 실용성을 크게 높였다.",
    impact:
      "마스킹 기반 자기지도 학습이 비전에서도 가장 효과적인 사전학습 전략이 될 수 있음을 입증했다. BERT의 성공을 비전으로 확장한 결정적 연구로, 이후 VideoMAE, AudioMAE 등 다양한 모달리티로 확산되었으며 대규모 비전 모델 학습의 표준 방법론이 되었다.",
    relatedFoundations: ["vit", "bert"],
    relatedPapers: [
      { id: "dino", fieldId: "representation", title: "DINO", relation: "related" },
      { id: "simclr", fieldId: "representation", title: "SimCLR", relation: "prior" },
    ],
  },

  // ===== Science Field =====
  "alphafold2": {
    tldr: "Evoformer와 구조 모듈을 결합하여 원자 수준의 정확도로 단백질 3D 구조를 예측, 50년 난제인 단백질 접힘 문제를 사실상 해결한 논문.",
    background:
      "단백질의 아미노산 서열로부터 3D 구조를 예측하는 문제는 1972년 Anfinsen의 열역학 가설 이래 생물학의 최대 난제 중 하나였다. CASP(Critical Assessment of protein Structure Prediction) 대회에서 수십 년간 점진적 개선만 있었으며, 실험적 구조 결정(X-선 결정학, cryo-EM)은 비용과 시간이 막대했다.",
    keyIdea:
      "AlphaFold2는 두 가지 핵심 모듈로 구성된다. Evoformer는 MSA(다중 서열 정렬) 표현과 쌍별(pairwise) 표현 사이에 정보를 반복적으로 교환하는 새로운 어텐션 아키텍처로, 진화적 공변이(covariation)와 공간적 관계를 동시에 추론한다. 구조 모듈(Structure Module)은 SE(3)-등변(equivariant) 변환을 사용하여 각 잔기의 3D 좌표를 직접 예측하며, 반복적 정제(recycling)를 통해 정확도를 높인다. 전체가 end-to-end로 미분 가능하여, FAPE 손실로 직접 3D 좌표를 학습한다.",
    method:
      "입력 서열에서 MSA와 템플릿을 검색하고, 48층 Evoformer로 표현을 구축한 뒤 구조 모듈에서 원자 좌표를 생성한다. 3회 반복(recycling)으로 정제하며, 학습 시 PDB의 실험적 구조를 정답으로 사용한다.",
    results:
      "CASP14에서 GDT 점수 중앙값 92.4를 달성하여 2위(약 67)를 압도적으로 앞섰다. 이는 실험적 방법에 필적하는 정확도로, 대부분의 예측 구조가 1Å 이내의 오차를 보였다.",
    impact:
      "구조 생물학을 근본적으로 변혁하여, 2억 개 이상 단백질 구조의 예측 데이터베이스가 공개되었다. 신약 개발, 효소 설계, 질병 이해 등 광범위한 생명과학 분야에 혁명적 영향을 미쳤으며, 2024년 노벨 화학상 수상으로 이어졌다.",
    relatedFoundations: ["transformer", "attention-mechanism"],
    relatedPapers: [
      { id: "alphageometry", fieldId: "science", title: "AlphaGeometry", relation: "related" },
    ],
  },

  "alphageometry": {
    tldr: "LLM이 보조 구성을 생성하고 기호적 추론 엔진이 증명을 수행하는 신경-기호(neuro-symbolic) 시스템으로, 인간 시연 없이 올림피아드 수준 기하 문제를 풀어낸 논문.",
    background:
      "수학 올림피아드 기하 문제는 창의적 보조선 구성과 엄밀한 논리적 추론을 동시에 요구하는 극도로 어려운 과제이다. 기존 AI 시스템은 대수적 방법이나 좌표 기하로 일부 문제만 풀 수 있었으며, 순수 기하학적 증명에는 한계가 있었다. 특히 학습 데이터가 거의 없는 것이 큰 제약이었다.",
    keyIdea:
      "AlphaGeometry는 두 시스템의 시너지를 활용한다. 기호적 추론 엔진(DD+AR)은 기존 조건으로부터 연역적으로 도출 가능한 모든 사실을 빠르게 계산한다. 추론이 막히면 LLM이 새로운 보조점이나 보조선을 제안하여 추론 공간을 확장한다. 핵심 혁신은 합성 데이터 생성에 있는데, 무작위 기하 구성에서 역추론으로 1억 개 이상의 증명 데이터를 자동 생성하여 LLM을 학습시켰다. 이를 통해 인간이 만든 증명 데이터 없이도 효과적인 보조 구성 능력을 확보했다.",
    method:
      "기호적 엔진이 현재 상태에서 도출 가능한 사실을 모두 찾고, 목표에 도달하지 못하면 LLM에 상태를 직렬화하여 보조 구성을 요청한다. LLM이 제안한 구성을 추가하고 기호적 추론을 재실행하는 루프를 반복한다. LLM은 합성 증명 데이터로 사전학습한 트랜스포머이다.",
    results:
      "IMO(국제수학올림피아드) 기하 문제 30개 중 25개를 풀어, 은메달 수준의 성적을 달성했다. 이는 이전 최고 기록(10개)을 크게 앞서며, 기호적 엔진만으로는 14개, LLM만으로는 불가능한 성과이다.",
    impact:
      "AI의 수학적 추론 능력에서 신경-기호적 접근의 강력한 가능성을 입증했다. 합성 데이터를 통한 수학 추론 학습이라는 새로운 패러다임을 제시했으며, AI가 인간 수준의 창의적 수학 문제 해결에 다가가고 있음을 보여주었다.",
    relatedFoundations: ["transformer"],
    relatedPapers: [
      { id: "alphafold2", fieldId: "science", title: "AlphaFold2", relation: "related" },
    ],
  },

  // ===== Efficient AI Field =====
  "distillation": {
    tldr: "대형 교사 모델의 소프트 타겟(soft targets)을 사용하여 소형 학생 모델에 '어두운 지식(dark knowledge)'을 전달하는 지식 증류 프레임워크를 제안한 논문.",
    background:
      "대형 앙상블 모델이나 깊은 네트워크는 뛰어난 성능을 보이지만 추론 비용이 크다. 학습 시에는 큰 모델의 용량이 유리하지만, 배포 시에는 효율적인 작은 모델이 필요하다. 모델의 일반화 능력을 작은 모델로 옮기는 체계적 방법론이 요구되었다.",
    keyIdea:
      "핵심 통찰은 교사 모델의 소프트맥스 출력에 담긴 클래스 간 유사도 정보(dark knowledge)가 원-핫 라벨보다 훨씬 풍부하다는 것이다. 예를 들어 '자동차' 이미지에 대해 '트럭'에 높은 확률을 부여하는 것은 두 클래스의 유사성을 반영한다. 온도(temperature) 파라미터 T를 높여 소프트맥스를 부드럽게 하면 이 정보가 더 잘 드러난다. 학생 모델은 하드 라벨과 교사의 소프트 타겟을 동시에 학습하여, 교사의 일반화 능력을 흡수한다.",
    method:
      "소프트맥스에 온도 T를 적용하여 교사의 출력 분포를 부드럽게 만들고, 학생 모델도 같은 온도에서 교사의 소프트 타겟을 KL-divergence로 모방하도록 학습한다. 최종 손실은 소프트 타겟 손실과 하드 라벨 크로스엔트로피의 가중 합이다.",
    results:
      "MNIST에서 단일 학생 모델이 앙상블과 동등한 성능을 달성했으며, 음성 인식에서는 10개 모델 앙상블의 지식을 단일 모델로 성공적으로 압축했다.",
    impact:
      "모델 압축과 배포 효율화의 표준 방법론이 되었다. 이후 DistilBERT, TinyBERT 등 NLP 모델 압축에 광범위하게 적용되었으며, 자기증류(self-distillation), 온라인 증류 등 다양한 변형으로 발전했다. 에지 디바이스 AI 배포의 핵심 기술이다.",
    relatedFoundations: ["dropout", "alexnet"],
    relatedPapers: [
      { id: "lora", fieldId: "efficient", title: "LoRA", relation: "related" },
      { id: "qlora", fieldId: "efficient", title: "QLoRA", relation: "related" },
    ],
  },

  "lora": {
    tldr: "사전학습된 가중치를 동결하고 저랭크 행렬 쌍(A, B)만 학습하여, 추론 지연 없이 파라미터 효율적 미세조정을 가능하게 한 논문.",
    background:
      "GPT-3 등 대규모 언어모델의 전체 파라미터 미세조정은 엄청난 GPU 메모리와 저장 공간을 요구한다. 태스크마다 전체 모델 복사본을 유지해야 하는 문제도 있었다. 어댑터 모듈 등 기존 방법은 추론 시 추가 지연을 유발하는 단점이 있었다.",
    keyIdea:
      "대규모 모델의 미세조정 시 가중치 변화(ΔW)가 실제로 저랭크(low-rank) 구조를 가진다는 가설에 기반한다. 원래 가중치 W를 동결하고, ΔW = BA 형태의 저랭크 분해를 학습한다(B ∈ R^{d×r}, A ∈ R^{r×k}, r ≪ min(d,k)). 학습할 파라미터가 원래의 0.01% 수준으로 줄어들며, 추론 시 BA를 W에 병합(merge)하면 추가 연산이 전혀 없다. 어텐션 모듈의 쿼리와 밸류 프로젝션에 적용하는 것이 가장 효과적이다.",
    method:
      "트랜스포머의 각 어텐션 층에서 W_q, W_v에 저랭크 행렬 A(가우시안 초기화), B(영 초기화)를 추가한다. 순전파 시 h = Wx + BAx로 계산하며, 역전파에서 A와 B만 업데이트한다. GPT-3 175B에 r=4~8로 적용하여 검증했다.",
    results:
      "GPT-3 175B에서 학습 파라미터 0.01%(약 4.7M)만으로 전체 미세조정과 동등하거나 더 나은 성능을 달성했다. 학습 시 GPU 메모리를 3배 절감하고, 체크포인트 크기를 10000배 이상 줄였다. 추론 지연은 전혀 증가하지 않았다.",
    impact:
      "대규모 언어모델의 미세조정을 민주화한 핵심 기술이다. 이후 QLoRA, LoRA+, DoRA 등 수많은 변형이 등장했으며, Stable Diffusion, LLaMA 등 다양한 모델의 커스터마이징에 사실상 표준으로 자리잡았다. 개인 연구자와 소규모 팀이 대형 모델을 활용할 수 있게 한 실용적 기여가 크다.",
    relatedFoundations: ["transformer", "gpt3"],
    relatedPapers: [
      { id: "qlora", fieldId: "efficient", title: "QLoRA", relation: "successor" },
      { id: "distillation", fieldId: "efficient", title: "Knowledge Distillation", relation: "related" },
    ],
  },

  "qlora": {
    tldr: "4비트 NormalFloat 양자화와 LoRA를 결합하여, 단일 48GB GPU에서 65B 파라미터 모델을 미세조정할 수 있게 한 논문.",
    background:
      "LoRA가 파라미터 효율적 미세조정을 가능하게 했지만, 65B 이상의 초대형 모델은 모델 가중치를 메모리에 올리는 것 자체가 어려웠다. 기존 양자화 기법은 추론 효율화에 초점을 맞추었으며, 양자화된 상태에서의 미세조정은 성능 저하가 우려되었다.",
    keyIdea:
      "QLoRA는 세 가지 핵심 기술을 도입한다. (1) 4-bit NormalFloat(NF4): 정규분포를 가정하고 각 양자화 구간에 동일한 확률 질량을 배분하는 정보 이론적으로 최적의 데이터 타입이다. (2) 이중 양자화(Double Quantization): 양자화 상수 자체를 다시 양자화하여 파라미터당 평균 0.37비트를 추가로 절약한다. (3) 페이지드 옵티마이저: GPU 메모리 스파이크 시 옵티마이저 상태를 CPU 메모리로 자동 오프로딩한다. 이들을 결합하면 65B 모델을 단일 48GB GPU에서 미세조정할 수 있다.",
    method:
      "사전학습된 모델을 NF4로 양자화하여 동결하고, 그 위에 BFloat16의 LoRA 어댑터를 추가한다. 역전파 시 그래디언트는 양자화된 가중치를 BF16으로 역양자화하여 계산한다. Guanaco 데이터셋으로 LLaMA 모델을 미세조정하여 검증했다.",
    results:
      "QLoRA로 미세조정한 Guanaco 65B가 Vicuna 벤치마크에서 ChatGPT 대비 99.3% 수준에 도달했다. 4비트 양자화에도 불구하고 16비트 전체 미세조정 대비 성능 저하가 거의 없었으며, 메모리 사용량은 780GB에서 48GB로 대폭 감소했다.",
    impact:
      "대규모 언어모델의 미세조정을 소비자급 GPU에서 가능하게 하여 AI 연구의 민주화에 크게 기여했다. NF4 데이터 타입은 이후 양자화 연구의 기준이 되었으며, 오픈소스 LLM 생태계의 폭발적 성장을 촉진한 실용적 핵심 기술이다.",
    relatedFoundations: ["transformer"],
    relatedPapers: [
      { id: "lora", fieldId: "efficient", title: "LoRA", relation: "prior" },
      { id: "llama", fieldId: "llm", title: "LLaMA", relation: "related" },
    ],
  },

  // ===== World Models Field =====
  "world-models-ha": {
    tldr: "VAE(시각) + MDN-RNN(동역학) + 작은 컨트롤러로 구성된 월드 모델을 학습하고, '꿈(dream)' 속에서 에이전트를 훈련할 수 있음을 보인 논문.",
    background:
      "모델 프리 강화학습은 환경과의 막대한 상호작용을 필요로 하며, 이는 실제 세계 적용에서 큰 제약이다. 인간은 세상의 내부 모델을 구축하여 상상 속에서 계획하고 행동을 시뮬레이션하는데, 이를 AI에 구현하려는 시도가 월드 모델 연구의 기원이다.",
    keyIdea:
      "세 가지 모듈로 구성된다. (1) Vision Model (VAE): 고차원 관측을 저차원 잠재 벡터 z로 압축한다. (2) Memory Model (MDN-RNN): 잠재 상태의 시간적 동역학을 학습하여 미래 상태의 확률 분포를 예측한다. 혼합 밀도 네트워크(MDN)로 다중 모드 분포를 포착한다. (3) Controller: 작은 선형 모델이 z와 RNN 은닉 상태를 입력받아 행동을 출력한다. 핵심 아이디어는 학습된 월드 모델 내부에서 완전히 시뮬레이션된 환경('꿈')을 만들어 컨트롤러를 훈련하는 것이다.",
    method:
      "먼저 랜덤 정책으로 환경 데이터를 수집하여 VAE와 MDN-RNN을 학습한다. 이후 학습된 월드 모델에서 생성된 가상 롤아웃으로 CMA-ES 진화 전략을 사용해 컨트롤러를 최적화한다. CarRacing-v0과 VizDoom 환경에서 실험했다.",
    results:
      "CarRacing에서 꿈 속 학습만으로 경쟁력 있는 성능을 달성했으며, 실제 환경 상호작용을 대폭 줄였다. VizDoom에서도 월드 모델 기반 학습의 가능성을 확인했으나, 모델 부정확성으로 인한 한계도 관찰되었다.",
    impact:
      "월드 모델 기반 강화학습의 현대적 프레임워크를 확립했다. 이후 Dreamer 시리즈, PlaNet 등 잠재 공간 월드 모델 연구의 직접적 기반이 되었으며, Sora와 같은 비디오 생성 모델이 세계 시뮬레이터로 발전하는 사상적 기원이 되었다.",
    relatedFoundations: ["vae", "lstm"],
    relatedPapers: [
      { id: "muzero", fieldId: "rl", title: "MuZero", relation: "related" },
      { id: "sora", fieldId: "world-models", title: "Sora", relation: "successor" },
    ],
  },

  "sora": {
    tldr: "시공간 패치에 대한 디퓨전 트랜스포머로 1분 이상의 일관된 비디오를 생성하며, 비디오 생성 모델이 세계 시뮬레이터로 기능할 수 있음을 제시한 기술 보고서.",
    background:
      "비디오 생성은 이미지 생성보다 훨씬 어려운 과제로, 시간적 일관성, 물리적 사실성, 긴 시퀀스 생성이 핵심 도전이었다. 기존 모델들은 짧은 클립이나 낮은 해상도에 제한되었으며, 현실 세계의 동역학을 일관되게 시뮬레이션하는 데 한계가 있었다.",
    keyIdea:
      "Sora는 비디오를 시공간 패치(spacetime patches)로 토큰화하여 트랜스포머의 확장성을 비디오 도메인에 가져온다. 다양한 해상도, 종횡비, 길이의 비디오를 통일된 표현으로 처리할 수 있다. 디퓨전 트랜스포머(DiT) 아키텍처를 사용하여 노이즈에서 고품질 비디오를 생성하며, 텍스트 조건부 생성이 가능하다. 핵심 관찰은 학습 데이터와 모델을 충분히 확장하면 비디오 모델이 3D 일관성, 물체 영속성, 물리적 상호작용 등 세계의 성질을 자연스럽게 학습한다는 것이다.",
    method:
      "비디오를 시각 인코더로 저차원 시공간 잠재 공간에 압축하고, 시공간 패치로 분할한다. 디퓨전 트랜스포머가 노이즈 패치에서 깨끗한 잠재 패치를 예측하며, 텍스트 프롬프트로 생성을 조건화한다. 대규모 비디오-텍스트 데이터에서 학습했다.",
    results:
      "최대 1분 길이의 고품질 1080p 비디오를 생성하며, 카메라 이동, 물체 상호작용, 장면 전환 등에서 높은 시각적 일관성을 보였다. 물리적 시뮬레이션과 3D 공간 이해 능력이 출현했으나, 일부 물리 법칙 위반도 관찰되었다.",
    impact:
      "비디오 생성의 품질과 길이에서 전례 없는 도약을 이루어, 영상 제작과 시각 콘텐츠 산업에 혁명적 변화를 예고했다. '비디오 생성 모델 = 세계 시뮬레이터'라는 비전을 제시하여, 월드 모델 연구에 새로운 방향을 열었다.",
    relatedFoundations: ["ddpm", "transformer", "vit"],
    relatedPapers: [
      { id: "world-models-ha", fieldId: "world-models", title: "World Models", relation: "prior" },
      { id: "genie", fieldId: "world-models", title: "Genie", relation: "related" },
    ],
  },

  "genie": {
    tldr: "단일 이미지에서 플레이 가능한 2D 게임 환경을 생성하는 모델로, 잠재 행동 모델(latent action model)을 통해 라벨 없이 행동 공간을 학습한다.",
    background:
      "인터랙티브 환경 생성은 비디오 생성보다 더 어려운 과제로, 사용자 입력에 따라 일관되게 반응하는 동적 세계를 만들어야 한다. 기존 세계 모델은 특정 게임이나 환경에 특화되어 있었으며, 범용적인 인터랙티브 환경 생성은 미개척 분야였다.",
    keyIdea:
      "Genie는 세 가지 구성요소로 이루어진다. (1) 잠재 행동 모델(Latent Action Model): 비디오 프레임 쌍 사이의 변환을 분석하여, 행동 라벨 없이도 이산적 잠재 행동 공간을 자동으로 학습한다. (2) 비디오 토크나이저: VQ-VAE로 프레임을 이산 토큰으로 변환한다. (3) 동역학 모델: MaskGIT 스타일의 트랜스포머가 현재 프레임과 잠재 행동을 조건으로 다음 프레임을 생성한다. 핵심은 인터넷의 플랫포머 게임 영상에서 행동 라벨 없이 학습하여, 추론 시 사용자가 잠재 행동을 선택하여 환경과 상호작용할 수 있다는 것이다.",
    method:
      "200,000시간 이상의 인터넷 게임 영상에서 학습한다. ST-transformer(Spatiotemporal Transformer)로 시공간 토큰을 처리하며, VQ 코드북으로 행동을 이산화한다. 텍스트 프롬프트나 단일 이미지에서 인터랙티브 환경을 생성할 수 있다.",
    results:
      "단일 이미지에서 일관된 물리적 동역학을 가진 플레이 가능한 2D 환경을 생성했다. 학습된 잠재 행동이 실제 게임 조작(이동, 점프 등)과 의미적으로 대응함을 확인했다. 11B 파라미터 모델에서 가장 좋은 품질을 달성했다.",
    impact:
      "인터랙티브 세계 생성이라는 새로운 연구 분야를 개척했다. AI가 단순히 콘텐츠를 생성하는 것을 넘어 상호작용 가능한 환경을 만들 수 있음을 보여주었으며, 게임 개발, 시뮬레이션, 로봇 학습을 위한 환경 생성 등에 광범위한 응용 가능성을 제시했다.",
    relatedFoundations: ["transformer", "vit"],
    relatedPapers: [
      { id: "sora", fieldId: "world-models", title: "Sora", relation: "related" },
      { id: "world-models-ha", fieldId: "world-models", title: "World Models", relation: "prior" },
    ],
  },

  // ===== Audio Field =====
  "wavenet": {
    tldr: "팽창 인과 컨볼루션(dilated causal convolutions)으로 원시 오디오 파형을 직접 자기회귀 생성하여, 기존 TTS를 압도하는 자연스러운 음성을 합성한 논문.",
    background:
      "기존 음성 합성(TTS)은 연결 합성(concatenative)이나 파라메트릭 합성에 의존했으며, 생성된 음성이 기계적이고 부자연스러웠다. 원시 오디오는 초당 16,000~48,000 샘플의 매우 긴 시퀀스로, 이를 직접 모델링하는 것은 시퀀스 길이와 계산량 면에서 극도로 도전적이었다.",
    keyIdea:
      "WaveNet은 원시 오디오 파형의 각 샘플을 이전 샘플들로부터 자기회귀적으로 예측하는 확률 모델이다. 핵심 아키텍처는 팽창 인과 컨볼루션으로, 미래 정보를 사용하지 않으면서(인과) 지수적으로 증가하는 수용장(receptive field)을 확보한다(팽창). μ-law 압축으로 16비트 오디오를 256 카테고리로 양자화하고, 게이트 활성화와 잔차 연결을 사용한다. 텍스트나 화자 정보를 조건으로 부여하여 TTS와 다화자 합성이 가능하다.",
    method:
      "30층의 팽창 인과 컨볼루션 스택으로 구성하며, 팽창 계수를 1, 2, 4, ..., 512로 기하급수적으로 증가시켜 수만 샘플의 수용장을 확보한다. 소프트맥스 출력으로 다음 샘플의 분포를 예측하고, 학습 시 teacher-forcing을 사용한다.",
    results:
      "MOS(Mean Opinion Score) 평가에서 영어 4.21, 중국어 4.08로 기존 최고 시스템을 큰 차이로 능가했다. 음악 생성에서도 현실적인 오디오를 생성했으며, 화자 조건화를 통해 다양한 목소리를 합성할 수 있었다.",
    impact:
      "딥러닝 기반 음성 합성의 시대를 열어, 이후 모든 신경망 TTS 시스템의 기반이 되었다. Google Assistant에 실제 적용되어 상용화되었으며, Tacotron, Parallel WaveNet 등 후속 연구를 촉발했다. 오디오 생성 AI 분야 전체의 출발점이 된 기념비적 논문이다.",
    relatedFoundations: ["lstm"],
    relatedPapers: [
      { id: "whisper", fieldId: "audio", title: "Whisper", relation: "successor" },
      { id: "vall-e", fieldId: "audio", title: "VALL-E", relation: "successor" },
    ],
  },

  "whisper": {
    tldr: "68만 시간의 약한 지도학습 데이터로 인코더-디코더 트랜스포머를 훈련하여, 다국어 음성 인식·번역·언어 감지 등 다중 태스크를 하나의 모델로 수행하는 범용 음성 시스템.",
    background:
      "음성 인식 연구는 오랫동안 깨끗하게 전사된 소규모 데이터셋에 의존했다. 이로 인해 특정 도메인이나 언어에 과적합되어, 실제 환경의 다양한 억양, 배경 소음, 전문 용어에 취약했다. 대규모 약한 지도학습이 이 문제를 해결할 수 있는지가 핵심 질문이었다.",
    keyIdea:
      "Whisper의 핵심 전략은 인터넷에서 수집한 68만 시간의 오디오-텍스트 쌍을 약한 지도학습(weak supervision)으로 활용하는 것이다. 데이터 품질이 완벽하지 않더라도, 규모의 힘으로 강건한 일반화를 달성한다. 단일 모델이 다국어 음성 인식, 음성 번역, 언어 감지, 타임스탬프 예측 등 여러 태스크를 텍스트 토큰의 특수 포맷으로 통합 처리한다. 태스크와 언어를 나타내는 특수 토큰을 사용하여, 디코더가 맥락에 따라 적절한 출력을 생성하도록 한다.",
    method:
      "멜 스펙트로그램을 입력으로 받는 트랜스포머 인코더와, 텍스트를 자기회귀적으로 생성하는 디코더로 구성된다. 다양한 크기(39M~1.55B 파라미터)로 학습하며, 멀티태스크 학습을 통해 하나의 모델에서 모든 기능을 수행한다.",
    results:
      "영어 음성 인식에서 사전 학습만으로(미세조정 없이) LibriSpeech에서 기존 지도학습 모델과 동등한 성능을 달성했다. 다국어에서는 Fleurs 벤치마크의 많은 언어에서 최고 성능을 기록했으며, 배경 소음, 억양, 전문 용어에 대한 강건성이 크게 향상되었다.",
    impact:
      "음성 인식을 '풀린 문제(solved problem)'에 가깝게 만든 전환점이다. 오픈소스로 공개되어 전 세계적으로 활용되고 있으며, 실시간 자막, 회의 기록, 접근성 도구 등 수많은 응용에 직접 적용되고 있다. 약한 지도학습의 확장이 전문 데이터셋을 대체할 수 있음을 실증했다.",
    relatedFoundations: ["transformer", "seq2seq"],
    relatedPapers: [
      { id: "wavenet", fieldId: "audio", title: "WaveNet", relation: "prior" },
      { id: "vall-e", fieldId: "audio", title: "VALL-E", relation: "related" },
    ],
  },

  "vall-e": {
    tldr: "TTS를 신경 오디오 코덱 토큰에 대한 언어 모델링 문제로 재정의하여, 단 3초의 음성 샘플만으로 화자의 목소리를 복제하는 제로샷 TTS를 달성한 논문.",
    background:
      "기존 TTS 시스템은 새로운 화자에 적응하기 위해 해당 화자의 상당한 녹음 데이터와 미세조정이 필요했다. 제로샷 TTS(한 번도 학습하지 않은 화자의 음성 합성)는 극도로 어려운 과제였으며, 특히 음색, 감정, 운율 등 화자 특성의 충실한 재현이 도전적이었다.",
    keyIdea:
      "VALL-E는 TTS를 조건부 언어 모델링으로 재정의한다. EnCodec 같은 신경 오디오 코덱이 음성을 이산 토큰 시퀀스로 변환하면, 이를 '오디오 언어'로 취급하여 GPT 스타일의 자기회귀 모델링을 적용한다. 코덱의 다중 양자화 레벨을 활용하여, 첫 번째 레벨은 자기회귀(AR) 모델로, 나머지는 비자기회귀(NAR) 모델로 생성하는 계층적 구조를 사용한다. 3초의 프롬프트 음성을 접두사로 제공하면, 모델이 해당 화자의 특성을 유지하며 임의의 텍스트를 음성으로 변환한다.",
    method:
      "60,000시간의 LibriLight 데이터에서 EnCodec 토큰을 추출하여 학습한다. 텍스트 음소 시퀀스와 3초 프롬프트의 코덱 토큰을 조건으로, AR 트랜스포머가 첫 번째 코덱 레벨을 생성하고, NAR 트랜스포머가 나머지 7개 레벨을 동시에 생성한다.",
    results:
      "VALL-E는 기존 최고 TTS 시스템 대비 화자 유사도에서 큰 개선을 보였으며, 음성 자연스러움도 동등 수준이었다. 단 3초의 프롬프트로 화자의 감정, 음향 환경까지 재현할 수 있었다.",
    impact:
      "TTS 패러다임을 코덱 기반 언어 모델링으로 전환한 획기적 연구이다. VALL-E X(다국어), VALL-E 2 등 후속 연구를 촉발했으며, 음성 합성의 품질과 유연성을 크게 높였다. 동시에 딥페이크 음성 등 잠재적 오용 위험에 대한 사회적 논의도 촉진했다.",
    relatedFoundations: ["transformer", "gpt3"],
    relatedPapers: [
      { id: "wavenet", fieldId: "audio", title: "WaveNet", relation: "prior" },
      { id: "whisper", fieldId: "audio", title: "Whisper", relation: "related" },
    ],
  },
};
