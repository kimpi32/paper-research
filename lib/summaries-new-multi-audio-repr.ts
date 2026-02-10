import type { PaperSummary } from "./paper-summaries";

export const newMultiAudioReprSummaries: Record<string, PaperSummary> = {
  // ===== Multimodal Field =====
  "vilbert": {
    tldr: "이미지와 텍스트를 각각 독립적으로 처리하는 두 개의 BERT 스트림을 공동 어텐션(co-attention) 트랜스포머 레이어로 연결하여, 비전-언어 태스크를 위한 사전학습 모델을 제안한 논문.",
    background:
      "BERT가 NLP에서 큰 성공을 거두었지만, 시각과 언어를 동시에 이해하는 멀티모달 사전학습은 초기 단계였다. 기존 접근은 이미지와 텍스트를 단일 스트림에 합치거나, 단순한 후기 결합(late fusion)에 의존하여 모달리티 간 깊은 상호작용이 부족했다. 각 모달리티의 독립적 표현을 유지하면서도 효과적으로 교차하는 아키텍처가 필요했다.",
    keyIdea:
      "ViLBERT는 이미지와 텍스트를 위한 두 개의 병렬 BERT 스트림을 설계하고, 공동 어텐션 트랜스포머(co-attentional transformer) 레이어를 도입하여 두 스트림 사이에서 정보를 교환한다. 각 스트림의 쿼리가 상대 스트림의 키-밸류를 참조하여 교차 모달 어텐션을 수행한다. 이미지 영역은 사전학습된 Faster R-CNN으로 추출하고, 텍스트는 WordPiece 토큰으로 처리한다. 마스크 언어/이미지 모델링과 이미지-텍스트 일치 예측의 두 가지 사전학습 목표를 사용한다.",
    method:
      "Conceptual Captions 3.3M 데이터셋에서 사전학습을 수행한다. 이미지는 Faster R-CNN으로 36개 영역 특징을 추출하고, 텍스트는 BERT 토크나이저로 처리한다. 공동 어텐션 레이어에서 각 모달리티의 쿼리가 상대방의 키-밸류를 어텐드하며, 6개 층에서 교차 정보 교환이 이루어진다.",
    results:
      "VQA, VCR, Grounding, Image Retrieval 등 4가지 비전-언어 태스크에서 기존 최고 성능을 달성했다. 특히 단일 스트림 방식 대비 공동 어텐션이 더 효과적임을 입증했으며, 사전학습된 표현의 전이 학습 효과가 두드러졌다.",
    impact:
      "비전-언어 사전학습의 초기 핵심 연구로, 두 스트림 공동 어텐션 구조가 이후 멀티모달 모델 설계의 중요한 기준점이 되었다. LXMERT, UNITER 등 후속 멀티모달 사전학습 모델에 직접적 영향을 미쳤으며, CLIP과 Flamingo로 이어지는 비전-언어 모델 발전의 토대를 놓았다.",
    relatedFoundations: ["transformer", "bert"],
    relatedPapers: [
      { id: "clip", fieldId: "multimodal", title: "CLIP", relation: "successor" },
    ],
  },

  "dall-e": {
    tldr: "이산 VAE(dVAE)로 이미지를 토큰화한 뒤 자기회귀 트랜스포머로 텍스트-이미지 토큰을 생성하여, 제로샷 텍스트-이미지 생성을 최초로 실현한 논문.",
    background:
      "텍스트 설명으로부터 이미지를 생성하는 것은 오랜 AI 연구 과제였지만, 기존 GAN 기반 방법들은 특정 도메인에 제한되거나 텍스트의 세밀한 의미를 반영하지 못했다. GPT 시리즈의 성공으로 자기회귀 모델의 강력한 생성 능력이 입증되었고, 이를 이미지 생성에 확장하려는 시도가 자연스럽게 이어졌다.",
    keyIdea:
      "DALL-E는 두 단계로 구성된다. 첫째, 이산 VAE(dVAE)가 256x256 이미지를 32x32 격자의 8192개 코드북 토큰으로 압축한다. Gumbel-Softmax 완화를 사용하여 이산 잠재 변수에 대해 end-to-end 학습이 가능하다. 둘째, 120억 파라미터의 자기회귀 트랜스포머가 256개의 BPE 텍스트 토큰과 1024개의 이미지 토큰을 하나의 시퀀스로 연결하여, 텍스트 조건부 이미지 생성을 수행한다. 추론 시 CLIP을 사용한 리랭킹으로 최적 이미지를 선택한다.",
    method:
      "2.5억 개의 이미지-텍스트 쌍에서 학습한다. dVAE를 먼저 학습하여 이미지 토크나이저를 만들고, 이후 텍스트+이미지 토큰 시퀀스에 대해 자기회귀 트랜스포머를 학습한다. 희소 어텐션 패턴을 사용하여 긴 시퀀스(1280 토큰)의 계산 비용을 줄였다.",
    results:
      "다양한 텍스트 프롬프트에 대해 창의적이고 의미적으로 일관된 이미지를 생성했다. '아보카도 모양의 안락의자'와 같은 새로운 개념 조합도 합성할 수 있어, 강력한 구성적 일반화(compositional generalization) 능력을 보였다.",
    impact:
      "텍스트-이미지 생성 분야를 개척한 기념비적 연구로, AI 창작의 새로운 시대를 열었다. DALL-E 2, Stable Diffusion, Midjourney 등 후속 이미지 생성 모델의 직접적 영감이 되었으며, 자기회귀 모델링을 비전 도메인으로 확장하는 핵심 사례가 되었다.",
    relatedFoundations: ["transformer", "vae", "gpt"],
    relatedPapers: [
      { id: "dalle2", fieldId: "generative", title: "DALL-E 2", relation: "successor" },
      { id: "clip", fieldId: "multimodal", title: "CLIP", relation: "related" },
    ],
  },

  "align": {
    tldr: "정제되지 않은 18억 개의 노이즈 이미지-텍스트 쌍을 듀얼 인코더로 학습하여, 데이터 규모의 힘으로 CLIP에 필적하는 비전-언어 정렬을 달성한 논문.",
    background:
      "CLIP이 4억 개의 정제된 이미지-텍스트 쌍으로 뛰어난 비전-언어 표현을 학습했지만, 고품질 데이터 큐레이션은 비용이 크다. 웹에서 수집한 노이즈 데이터를 정제 없이 대규모로 사용해도 유사한 성능을 달성할 수 있는지가 핵심 질문이었다. 데이터 품질과 규모 사이의 트레이드오프를 체계적으로 탐구할 필요가 있었다.",
    keyIdea:
      "ALIGN은 최소한의 필터링만 적용한 18억 개의 노이즈 이미지-텍스트 쌍(alt-text 기반)을 사용한다. 이미지 인코더로 EfficientNet, 텍스트 인코더로 BERT를 사용하는 듀얼 인코더 구조에서, 대조 학습(contrastive learning)으로 이미지-텍스트 임베딩을 정렬한다. 핵심 발견은 데이터의 노이즈가 있어도, 충분한 규모가 있으면 노이즈를 자연스럽게 극복하여 강력한 표현을 학습한다는 것이다. 간단한 빈도 기반 필터링만으로 충분하며, 복잡한 큐레이션은 불필요하다.",
    method:
      "EfficientNet-L2를 이미지 인코더, BERT-Large를 텍스트 인코더로 사용하여 인배치 대조 손실(in-batch contrastive loss)로 학습한다. 배치 크기 16,384로 노이즈 이미지-alt text 쌍에서 직접 학습하며, 이미지-텍스트 유사도를 정규화된 내적으로 계산한다.",
    results:
      "ImageNet 제로샷 분류에서 CLIP에 필적하는 76.4% top-1 정확도를 달성했다. Flickr30K 이미지-텍스트 검색에서 새로운 최고 성능을 기록했으며, 다국어 텍스트-이미지 검색에서도 강력한 성능을 보였다.",
    impact:
      "대규모 노이즈 데이터의 효용성을 실증하여, 멀티모달 학습에서 데이터 큐레이션의 필요성을 재고하게 만들었다. CLIP과 함께 비전-언어 듀얼 인코더의 확장 가능성을 확인했으며, 이후 SigLIP, EVA-CLIP 등 효율적 대조 학습 연구에 영향을 미쳤다.",
    relatedFoundations: ["transformer", "bert", "resnet"],
    relatedPapers: [
      { id: "clip", fieldId: "multimodal", title: "CLIP", relation: "related" },
      { id: "blip2", fieldId: "multimodal", title: "BLIP-2", relation: "successor" },
    ],
  },

  "blip2": {
    tldr: "동결된 이미지 인코더와 동결된 대규모 언어모델 사이를 경량 Q-Former로 연결하여, 효율적이면서 강력한 멀티모달 이해와 생성을 달성한 논문.",
    background:
      "비전-언어 모델의 규모가 커지면서, 이미지 인코더와 언어모델을 처음부터 공동 학습하는 것은 엄청난 계산 비용을 요구했다. Flamingo 등이 동결된 모델 위에 어댑터를 추가하는 방식을 시도했지만, 두 모달리티 간의 효과적 정렬이 도전적이었다. 최소한의 학습 가능 파라미터로 사전학습된 모델의 능력을 최대한 활용하는 방법이 필요했다.",
    keyIdea:
      "BLIP-2의 핵심은 Q-Former(Querying Transformer)라는 경량 브릿지 모듈이다. Q-Former는 학습 가능한 32개의 쿼리 토큰을 사용하여 동결된 이미지 인코더에서 가장 유용한 시각 정보를 추출한다. 2단계 사전학습 전략을 사용하는데, 1단계에서 Q-Former를 이미지 인코더에 정렬하고(ITC, ITM, ITG 손실), 2단계에서 Q-Former의 출력을 동결된 LLM의 입력 공간에 투영한다. 이를 통해 이미지 인코더와 LLM을 모두 동결한 채 188M 파라미터만 학습하여 효율성을 극대화한다.",
    method:
      "1단계에서 ViT-G/14를 동결하고 Q-Former를 이미지-텍스트 대조(ITC), 이미지-텍스트 매칭(ITM), 이미지 조건부 텍스트 생성(ITG)으로 학습한다. 2단계에서 Q-Former 출력을 FC 레이어로 LLM 임베딩 공간에 투영하고, 동결된 OPT 또는 FlanT5를 사용하여 시각 조건부 언어 생성을 학습한다.",
    results:
      "VQAv2에서 제로샷 65.0%를 달성하여 Flamingo-80B(56.3%)를 54배 적은 학습 파라미터로 앞섰다. 이미지 캡셔닝, 시각 추론 등 다양한 태스크에서 기존 최고 성능을 갱신했으며, 학습 비용은 기존 대비 크게 절감되었다.",
    impact:
      "동결된 사전학습 모델을 경량 브릿지로 연결하는 효율적 멀티모달 학습 패러다임을 확립했다. LLaVA, InstructBLIP, MiniGPT-4 등 후속 비전-언어 모델의 설계 철학에 직접적 영향을 미쳤으며, 대규모 모델의 재사용과 합성을 통한 멀티모달 AI 구축의 실용적 경로를 제시했다.",
    relatedFoundations: ["transformer", "vit", "bert"],
    relatedPapers: [
      { id: "clip", fieldId: "multimodal", title: "CLIP", relation: "prior" },
      { id: "flamingo", fieldId: "multimodal", title: "Flamingo", relation: "prior" },
      { id: "llava", fieldId: "multimodal", title: "LLaVA", relation: "related" },
    ],
  },

  "cogvlm": {
    tldr: "트랜스포머의 각 레이어에 학습 가능한 시각 전문가(visual expert) 모듈을 삽입하여, 언어모델 성능을 유지하면서 깊은 수준의 비전-언어 융합을 달성한 논문.",
    background:
      "기존 비전-언어 모델은 이미지 특징을 언어모델의 입력 공간에 투영하는 얕은 정렬(shallow alignment)에 의존했다. 이는 시각 정보가 트랜스포머의 깊은 층에서 충분히 활용되지 못하는 한계가 있었다. MLP 어댑터나 크로스 어텐션은 부가적 모듈에 불과하여, 시각과 언어의 진정한 심층 통합에 한계가 있었다.",
    keyIdea:
      "CogVLM은 트랜스포머의 모든 어텐션 레이어와 FFN 레이어에 시각 전문가(visual expert)를 추가한다. 각 레이어에서 텍스트 토큰은 원래의 가중치로, 시각 토큰은 별도의 시각 전문가 가중치로 처리된다. 이미지 토큰의 QKV 프로젝션과 FFN에 각각 별도의 학습 가능한 행렬을 배치하여, 시각 정보가 네트워크의 모든 깊이에서 전문적으로 처리된다. 핵심은 원래 언어모델의 가중치를 동결하여 언어 능력을 보존하면서, 시각 전문가만 학습하여 시각적 이해를 추가하는 것이다.",
    method:
      "EVA2-CLIP-E를 이미지 인코더로 사용하고, Vicuna-7B 등의 언어모델에 시각 전문가 모듈을 삽입한다. 1단계에서 1.5B 이미지-텍스트 쌍으로 시각 전문가를 사전학습하고, 2단계에서 시각 질의응답(VQA) 등 다운스트림 데이터로 미세조정한다. 시각 전문가의 파라미터 수는 언어모델과 동일 규모이다.",
    results:
      "VQAv2, OKVQA, TextVQA, ScienceQA 등 17개 벤치마크 중 10개에서 기존 최고 성능을 달성했다. 특히 시각적 근거 추론(visual grounding)에서 뛰어난 성능을 보였으며, 언어모델의 원래 NLP 능력도 잘 보존되었다.",
    impact:
      "심층 비전-언어 융합의 새로운 패러다임을 제시하여, 얕은 정렬 방식의 한계를 극복하는 방향을 열었다. 시각 전문가 구조는 이후 CogVLM2, CogAgent 등으로 발전했으며, 멀티모달 모델에서 모달리티별 전문화 처리의 중요성을 부각시켰다.",
    relatedFoundations: ["transformer", "vit"],
    relatedPapers: [
      { id: "llava", fieldId: "multimodal", title: "LLaVA", relation: "related" },
      { id: "blip2", fieldId: "multimodal", title: "BLIP-2", relation: "prior" },
    ],
  },

  "gemini": {
    tldr: "텍스트, 이미지, 오디오, 비디오, 코드를 처음부터 네이티브하게 이해하고 생성하는 멀티모달 모델로, 다양한 벤치마크에서 SOTA를 달성한 구글의 차세대 AI 시스템.",
    background:
      "GPT-4가 텍스트와 이미지를 처리할 수 있었지만, 오디오와 비디오는 별도 파이프라인에 의존했다. 대부분의 멀티모달 모델은 각 모달리티를 별도로 인코딩한 뒤 결합하는 방식이었으며, 진정한 네이티브 멀티모달 모델--모든 모달리티를 처음부터 통합 학습한--은 아직 실현되지 않았다.",
    keyIdea:
      "Gemini는 텍스트, 이미지, 오디오, 비디오를 네이티브하게 처리하도록 처음부터 설계된 트랜스포머 기반 모델이다. 각 모달리티를 별도 인코더로 처리한 뒤 결합하는 기존 방식과 달리, 다양한 모달리티의 인터리브된(interleaved) 입출력을 자연스럽게 처리한다. Ultra, Pro, Nano의 세 가지 크기로 제공되며, 긴 컨텍스트(32K 토큰)를 효율적으로 처리한다. 비디오를 프레임 시퀀스가 아닌 연속적 시공간 스트림으로 이해하고, 오디오의 음높이·감정·배경소리 등을 직접 인식한다.",
    method:
      "대규모 멀티모달 데이터셋(웹 문서, 이미지, 오디오, 비디오)에서 혼합 학습을 수행한다. TPUv4/v5e 클러스터에서 효율적인 학습 인프라를 구축하고, 텍스트·이미지·오디오·비디오 모달리티를 동시에 학습한다. 구체적 아키텍처 세부사항은 비공개이나, 디코더 전용 트랜스포머를 기반으로 한다.",
    results:
      "Gemini Ultra는 MMLU에서 90.0%로 인간 전문가 수준을 처음으로 넘었고, 32개 멀티모달 벤치마크 중 30개에서 기존 SOTA를 갱신했다. 수학 추론(MATH), 코드 생성(HumanEval), 멀티모달 이해(MMMU) 등에서 GPT-4V를 상회하는 성능을 보였다.",
    impact:
      "네이티브 멀티모달 AI의 가능성을 대규모로 실증하여, AI 시스템이 인간처럼 여러 감각을 통합적으로 처리할 수 있는 방향을 제시했다. GPT-4와의 경쟁 구도를 형성하며 AI 모델의 멀티모달화를 가속시켰고, 이후 Gemini 1.5 Pro의 100만 토큰 컨텍스트 윈도우 등 혁신적 후속 발전으로 이어졌다.",
    relatedFoundations: ["transformer", "attention-mechanism", "scaling-laws"],
    relatedPapers: [
      { id: "clip", fieldId: "multimodal", title: "CLIP", relation: "prior" },
      { id: "gpt4", fieldId: "llm", title: "GPT-4", relation: "related" },
      { id: "internvl", fieldId: "multimodal", title: "InternVL", relation: "related" },
    ],
  },

  "internvl": {
    tldr: "비전 파운데이션 모델을 60억 파라미터 규모로 확장하고 LLM과 점진적으로 정렬하여, 이미지·비디오·문서 이해를 아우르는 범용 비전-언어 모델을 구축한 논문.",
    background:
      "CLIP 등의 비전-언어 모델은 비전 인코더의 규모가 상대적으로 작아(ViT-L/14, ~300M), 언어모델의 급격한 성장에 비해 비전 측의 표현력이 부족했다. 비전 인코더를 수십억 파라미터로 확장하면서도 효과적으로 LLM과 정렬하는 것이 도전적 과제였다.",
    keyIdea:
      "InternVL은 InternViT-6B라는 60억 파라미터의 비전 트랜스포머를 설계하여, 비전 측의 표현력을 대폭 강화한다. 점진적 정렬 전략을 사용하여, 대조 학습으로 비전-언어 정렬을 먼저 수행한 뒤, 생성적 학습으로 LLM과의 연결을 심화한다. 동적 고해상도(dynamic high-resolution) 입력을 지원하여 다양한 종횡비와 해상도의 이미지를 효율적으로 처리한다. 비전 인코더의 특징을 QLLaMA를 통해 LLM 입력 공간으로 투영하며, 비전과 언어 모두에서 강력한 성능을 달성한다.",
    method:
      "InternViT-6B를 웹 규모 이미지-텍스트 데이터에서 대조 학습으로 사전학습한다. 이후 QLLaMA(Q-Former 스타일의 LLaMA 변형)를 통해 InternLM-7B/20B 등의 LLM과 연결한다. 다단계 학습으로 저해상도 정렬, 고해상도 미세조정, 인스트럭션 튜닝을 순차적으로 수행한다.",
    results:
      "이미지 분류, 시각적 질의응답, 문서 이해, 비디오 이해 등 광범위한 벤치마크에서 경쟁력 있는 성능을 달성했다. 특히 OCR, 차트 이해 등 세밀한 시각적 인식이 필요한 태스크에서 우수했으며, GPT-4V에 필적하는 결과를 다수 벤치마크에서 보였다.",
    impact:
      "비전 파운데이션 모델의 규모 확장이 멀티모달 성능에 핵심적임을 입증하여, 비전 인코더 연구의 새로운 방향을 제시했다. InternVL 1.5, 2.0으로 빠르게 발전하며 오픈소스 멀티모달 모델의 최전선에 자리잡았고, 상용 모델에 비견되는 오픈소스 대안으로서 커뮤니티에 큰 기여를 했다.",
    relatedFoundations: ["transformer", "vit"],
    relatedPapers: [
      { id: "clip", fieldId: "multimodal", title: "CLIP", relation: "prior" },
      { id: "llava", fieldId: "multimodal", title: "LLaVA", relation: "related" },
      { id: "gemini", fieldId: "multimodal", title: "Gemini", relation: "related" },
    ],
  },

  // ===== Audio Field =====
  "tacotron2": {
    tldr: "어텐션 기반 시퀀스-투-시퀀스 모델로 텍스트에서 멜 스펙트로그램을 생성하고, 수정된 WaveNet 보코더로 파형을 합성하여 인간에 근접한 음성 품질을 달성한 TTS 시스템.",
    background:
      "WaveNet이 고품질 음성 합성의 가능성을 보여주었지만, 텍스트에서 언어적 특징을 추출하여 WaveNet에 입력하는 복잡한 파이프라인이 필요했다. Tacotron 1세대가 시퀀스-투-시퀀스 접근을 시도했으나, 음성 품질과 강건성에 개선의 여지가 있었다. 언어학적 전처리를 최소화하면서 인간 수준의 음질에 도달하는 end-to-end 시스템이 필요했다.",
    keyIdea:
      "Tacotron 2는 두 단계의 생성 파이프라인을 사용한다. 첫째, 어텐션 기반 시퀀스-투-시퀀스 모델이 문자(character) 시퀀스에서 멜 스펙트로그램을 생성한다. 인코더는 문자 임베딩을 3층 CNN과 양방향 LSTM으로 처리하고, 디코더는 위치 민감(location-sensitive) 어텐션으로 단조 정렬을 유도한다. 둘째, 수정된 WaveNet이 멜 스펙트로그램을 조건으로 원시 파형을 합성한다. 핵심은 복잡한 언어학적 전처리 없이 문자에서 직접 자연스러운 음성을 생성한다는 것이다.",
    method:
      "인코더의 3층 CNN(각 512 필터, 커널 5)과 양방향 LSTM(512 유닛)으로 텍스트를 인코딩한다. 디코더는 2층 LSTM과 위치 민감 어텐션으로 80채널 멜 스펙트로그램을 프레임 단위로 생성한다. 이후 30층 수정 WaveNet이 24kHz 파형을 합성한다. 내부 미국 영어 데이터셋(24.6시간)으로 학습했다.",
    results:
      "MOS 평가에서 4.53점을 기록하여 전문 녹음(4.58)에 근접했으며, 기존 파라메트릭 TTS와 연결 합성 시스템을 크게 앞섰다. 합성음과 실제 음성의 차이가 통계적으로 유의미하지 않은 수준에 도달했다.",
    impact:
      "신경망 TTS의 사실상 표준 아키텍처가 되어, 이후 대부분의 TTS 연구가 Tacotron 2를 기반선(baseline)으로 채택했다. Google Assistant, Google Cloud TTS 등 상용 서비스에 직접 적용되었으며, FastSpeech, VITS 등 비자기회귀 TTS 연구의 출발점이 되었다.",
    relatedFoundations: ["seq2seq", "attention-mechanism", "lstm"],
    relatedPapers: [
      { id: "wavenet", fieldId: "audio", title: "WaveNet", relation: "prior" },
      { id: "vall-e", fieldId: "audio", title: "VALL-E", relation: "successor" },
    ],
  },

  "hubert": {
    tldr: "클러스터링된 음성 특징의 마스크 예측을 통해 자기지도 음성 표현을 학습하여, 라벨 없이도 강력한 음성 인식과 다양한 음성 태스크에 활용 가능한 범용 표현을 획득한 논문.",
    background:
      "NLP에서 BERT의 마스크 언어 모델링이 큰 성공을 거두었지만, 음성에 직접 적용하기 어려웠다. 텍스트는 이산적 토큰이지만 음성은 연속적 신호이며, 사전 정의된 어휘가 없어 마스크 예측의 타겟을 정의하기 어려웠다. wav2vec 2.0이 대조 학습으로 접근했으나, 마스크 예측 방식의 가능성은 아직 충분히 탐구되지 않았다.",
    keyIdea:
      "HuBERT(Hidden-Unit BERT)의 핵심 아이디어는 오프라인 클러스터링으로 음성의 이산적 유사 라벨을 생성하여 마스크 예측의 타겟으로 사용하는 것이다. MFCC나 이전 반복의 모델 특징에 k-means 클러스터링을 적용하여 각 프레임에 이산 라벨을 할당한다. 트랜스포머 인코더가 마스킹된 구간의 클러스터 라벨을 예측하도록 학습한다. 반복적 리파인먼트(iterative refinement)가 핵심으로, 학습된 모델의 중간 표현으로 다시 클러스터링하여 더 나은 타겟을 생성하고 재학습한다.",
    method:
      "1차 반복에서 MFCC 39차원 특징에 k-means(100 클러스터)를 적용하여 초기 라벨을 생성한다. CNN 특징 추출기와 트랜스포머 인코더로 마스크 예측 학습 후, 모델의 6번째 층 특징으로 재클러스터링(500 클러스터)하여 2차 반복을 수행한다. LibriSpeech 960시간으로 학습했다.",
    results:
      "LibriSpeech test-clean에서 10분 라벨로 미세조정 시 WER 4.3%를 달성하여 wav2vec 2.0(4.8%)을 앞섰다. SUPERB 벤치마크의 음성 인식, 화자 인식, 감정 인식 등 다양한 태스크에서 범용 표현으로서 우수한 성능을 보였다.",
    impact:
      "오프라인 클러스터링과 마스크 예측의 결합이라는 새로운 음성 자기지도 학습 패러다임을 확립했다. 이후 음성 언어 모델, 코덱 기반 음성 생성 등에서 HuBERT 표현이 널리 활용되었으며, data2vec 등 크로스모달 자기지도 학습 연구에도 영향을 미쳤다.",
    relatedFoundations: ["transformer", "bert"],
    relatedPapers: [
      { id: "wavenet", fieldId: "audio", title: "WaveNet", relation: "prior" },
      { id: "audiolm", fieldId: "audio", title: "AudioLM", relation: "successor" },
      { id: "data2vec", fieldId: "representation", title: "data2vec", relation: "related" },
    ],
  },

  "soundstream": {
    tldr: "잔여 벡터 양자화(RVQ)를 갖춘 엔드투엔드 신경 오디오 코덱으로, 가변 비트레이트에서 기존 코덱을 능가하는 품질을 달성하며 코덱 기반 오디오 생성의 토대를 놓은 논문.",
    background:
      "오디오 압축은 전통적으로 Opus, EVS 등 신호처리 기반 코덱에 의존했다. 이들은 수십 년간 최적화되었지만, 저비트레이트에서 품질 저하가 불가피했다. 신경망 기반 오디오 코덱이 등장하기 시작했으나, 실시간 처리와 가변 비트레이트 지원에서 실용적 한계가 있었다.",
    keyIdea:
      "SoundStream은 인코더-양자화기-디코더의 end-to-end 구조를 가진다. 인코더가 원시 오디오를 저차원 임베딩으로 압축하고, 잔여 벡터 양자화(Residual Vector Quantization, RVQ)가 이를 이산 코드로 변환한다. RVQ는 여러 단계의 코드북을 순차적으로 적용하여, 각 단계가 이전 단계의 잔차를 양자화한다. 코드북 수를 조절하여 단일 모델에서 3~18kbps의 가변 비트레이트를 달성한다. 적대적 학습(discriminator)과 재구성 손실을 결합하여 고품질 오디오를 생성하며, 실시간보다 빠른 처리가 가능하다.",
    method:
      "인코더는 1D 컨볼루션 스택, 디코더는 전치 컨볼루션 스택으로 구성된다. RVQ는 최대 12개의 코드북(각 1024 엔트리)을 사용한다. 웨이브 기반 판별기와 STFT 기반 판별기를 결합한 적대적 손실, 멜 스펙트로그램 재구성 손실, 특징 매칭 손실로 학습한다.",
    results:
      "3kbps에서 Opus 9kbps와 동등한 품질을, 6kbps에서 Opus 12kbps를 능가하는 품질을 달성했다. MUSHRA 평가에서 모든 비트레이트에서 기존 코덱 대비 우수한 주관적 품질을 보였으며, 단일 TPU에서 실시간보다 빠른 인코딩/디코딩이 가능했다.",
    impact:
      "신경 오디오 코덱 분야를 개척하여 RVQ 기반 토큰화가 오디오 생성 AI의 핵심 인프라가 되었다. AudioLM, MusicLM, VALL-E 등 코덱 토큰을 언어 모델로 생성하는 패러다임의 직접적 기반이 되었으며, EnCodec, DAC 등 후속 코덱 연구를 촉발했다.",
    relatedFoundations: ["gan"],
    relatedPapers: [
      { id: "wavenet", fieldId: "audio", title: "WaveNet", relation: "prior" },
      { id: "encodec", fieldId: "audio", title: "Encodec", relation: "successor" },
      { id: "audiolm", fieldId: "audio", title: "AudioLM", relation: "successor" },
    ],
  },

  "audiolm": {
    tldr: "SoundStream 코덱 토큰과 w2v-BERT 의미 토큰을 계층적으로 언어 모델링하여, 텍스트 프롬프트 없이도 일관된 음성과 음악을 자기회귀적으로 생성하는 오디오 생성 프레임워크.",
    background:
      "오디오 생성은 음성 합성과 음악 생성으로 나뉘어 각각 별도의 방법론이 사용되었다. 언어 모델링의 '다음 토큰 예측' 패러다임이 텍스트에서 강력한 생성 능력을 보여주었지만, 연속적이고 다층적인 오디오 신호에 이를 적용하기 위해서는 적절한 이산 토큰 표현이 필요했다.",
    keyIdea:
      "AudioLM은 오디오를 두 종류의 토큰으로 표현한다. (1) 의미 토큰(semantic tokens): w2v-BERT의 중간 표현을 k-means로 클러스터링하여 음성의 의미적·언어적 내용을 포착한다. (2) 음향 토큰(acoustic tokens): SoundStream의 RVQ 코드로 음색, 화자 특성 등 세밀한 음향 정보를 담는다. 생성은 3단계 계층적으로 진행되는데, 먼저 의미 토큰을 생성하여 전체적 구조를 결정하고, 이를 조건으로 거친 음향 토큰, 마지막으로 세밀한 음향 토큰을 순차적으로 생성한다.",
    method:
      "각 단계에서 디코더 전용 트랜스포머를 사용한다. 1단계는 의미 토큰의 자기회귀 생성, 2단계는 의미 토큰을 조건으로 SoundStream의 처음 4개 RVQ 레벨을 생성, 3단계는 나머지 RVQ 레벨을 생성한다. LibriLight(음성)과 MusicCaps(음악) 데이터로 학습했다.",
    results:
      "음성 연속 생성에서 화자 특성, 운율, 녹음 조건을 유지하면서 의미적으로 일관된 발화를 생성했다. 피아노 음악에서도 장기적 구조를 가진 자연스러운 연속을 생성했으며, 인간 평가에서 실제 오디오와 구분하기 어려운 수준을 달성했다.",
    impact:
      "오디오 생성을 언어 모델링 문제로 재정의하는 패러다임을 확립하여, 이후 MusicLM, VALL-E, Bark 등 코덱 기반 오디오 생성 연구의 직접적 기반이 되었다. 의미-음향 토큰의 계층적 생성이라는 프레임워크는 오디오 AI 분야의 핵심 설계 원리로 자리잡았다.",
    relatedFoundations: ["transformer"],
    relatedPapers: [
      { id: "soundstream", fieldId: "audio", title: "SoundStream", relation: "prior" },
      { id: "musiclm", fieldId: "audio", title: "MusicLM", relation: "successor" },
      { id: "vall-e", fieldId: "audio", title: "VALL-E", relation: "related" },
    ],
  },

  "encodec": {
    tldr: "잔여 벡터 양자화(RVQ)와 트랜스포머 언어 모델을 결합한 Meta의 고충실도 신경 오디오 코덱으로, 1.5kbps의 극저비트레이트에서도 우수한 음질을 달성하는 실시간 오디오 압축 시스템.",
    background:
      "SoundStream이 신경 오디오 코덱의 가능성을 보여주었지만, 극저비트레이트에서의 품질과 다양한 오디오 유형(음성, 음악, 일반 오디오)에 대한 범용성에서 개선의 여지가 있었다. 또한 코덱의 잠재 공간을 언어 모델로 모델링하여 추가적 품질 향상을 꾀하는 방향이 탐구되지 않았다.",
    keyIdea:
      "EnCodec은 세 가지 핵심 요소로 구성된다. (1) SEANet 기반의 인코더-디코더 구조로 24kHz/48kHz 오디오를 처리한다. (2) RVQ(최대 32개 코드북)로 1.5~24kbps의 광범위한 비트레이트를 지원하며, 학습 시 비트레이트를 무작위로 샘플링하여 단일 모델에서 가변 비트레이트를 달성한다. (3) 소형 트랜스포머 언어 모델이 RVQ 코드의 분포를 학습하여 추가적 무손실 압축(엔트로피 코딩)을 수행한다. 멀티스케일 STFT 판별기와 밸런서(balancer) 메커니즘으로 안정적 적대적 학습을 보장한다.",
    method:
      "인코더는 1D 컨볼루션과 LSTM, 디코더는 전치 컨볼루션과 LSTM으로 구성된다. 멀티스케일 STFT 판별기, 재구성 손실(시간 도메인 + 주파수 도메인), 특징 매칭 손실을 결합한다. 손실 가중치를 자동 조정하는 밸런서를 도입하여 학습 안정성을 높였다. DNS Challenge, Common Voice, Jamendo 등 다양한 데이터로 학습했다.",
    results:
      "3kbps에서 Opus 6kbps와 동등한 MUSHRA 점수를 달성했고, 6kbps에서는 기존 모든 코덱을 능가했다. 엔트로피 코딩 적용 시 평균 25~40%의 추가 비트레이트 절감을 달성했으며, 스트리밍 모드에서 실시간보다 빠른 처리가 가능했다.",
    impact:
      "오픈소스로 공개되어 오디오 AI 연구의 핵심 인프라가 되었다. VALL-E, MusicGen, AudioCraft 등 Meta의 후속 오디오 생성 모델의 토큰화 기반이며, 코덱 기반 오디오 생성 생태계의 확산에 결정적 역할을 했다. 통신, 스트리밍 등 실용적 오디오 압축 분야에도 영향을 미쳤다.",
    relatedFoundations: ["lstm", "gan"],
    relatedPapers: [
      { id: "soundstream", fieldId: "audio", title: "SoundStream", relation: "prior" },
      { id: "vall-e", fieldId: "audio", title: "VALL-E", relation: "related" },
      { id: "musiclm", fieldId: "audio", title: "MusicLM", relation: "related" },
    ],
  },

  "musiclm": {
    tldr: "MuLan 음악-텍스트 임베딩을 조건으로 AudioLM의 계층적 토큰 생성을 확장하여, 텍스트 설명으로부터 고충실도 음악을 생성하는 최초의 실용적 텍스트-투-뮤직 모델.",
    background:
      "AudioLM이 오디오 연속 생성에서 인상적인 결과를 보여주었지만, 텍스트 프롬프트에 따른 조건부 생성은 지원하지 않았다. 텍스트에서 음악을 생성하는 것은 주관적 미적 요소, 장기적 구조, 다양한 악기의 조화 등 고유한 난제를 가진다. 기존 텍스트-투-뮤직 시스템은 품질이 제한적이었다.",
    keyIdea:
      "MusicLM은 MuLan(Music-Language)이라는 음악-텍스트 공동 임베딩 모델을 조건화 신호로 활용한다. AudioLM의 계층적 생성 구조를 확장하여, MuLan 음악 토큰→의미 토큰→음향 토큰의 3단계로 음악을 생성한다. 텍스트 설명이 MuLan 텍스트 인코더를 통해 임베딩되면, 이를 조건으로 계층적 트랜스포머가 24kHz 고충실도 음악을 자기회귀적으로 생성한다. MusicCaps라는 고품질 음악-텍스트 쌍 평가 데이터셋도 함께 공개했다.",
    method:
      "MuLan으로 오디오와 텍스트를 공유 임베딩 공간에 매핑한다. 1단계 트랜스포머가 MuLan 토큰을 조건으로 의미 토큰(w2v-BERT)을 생성하고, 2단계가 거친 SoundStream 토큰을, 3단계가 세밀한 SoundStream 토큰을 생성한다. Free Music Archive 등 대규모 음악 데이터로 학습했다.",
    results:
      "MusicCaps 평가에서 기존 Mubert, Riffusion 등을 오디오 품질과 텍스트 충실도 모두에서 크게 앞섰다. 30초 이상의 일관된 음악 생성이 가능했으며, '재즈 풍의 슬픈 피아노 곡'과 같은 세밀한 텍스트 지시를 잘 반영했다. 멜로디 조건부 생성도 지원했다.",
    impact:
      "텍스트-투-뮤직 생성을 실용적 수준으로 끌어올린 획기적 연구로, AI 음악 창작의 새로운 시대를 열었다. MusicGen, Stable Audio, Udio 등 후속 음악 생성 모델에 직접적 영감을 주었으며, 음악 산업에서 AI 활용에 대한 활발한 논의를 촉발했다.",
    relatedFoundations: ["transformer", "bert", "resnet"],
    relatedPapers: [
      { id: "audiolm", fieldId: "audio", title: "AudioLM", relation: "prior" },
      { id: "soundstream", fieldId: "audio", title: "SoundStream", relation: "prior" },
      { id: "whisper", fieldId: "audio", title: "Whisper", relation: "related" },
    ],
  },

  "bark": {
    tldr: "GPT 스타일의 자기회귀 트랜스포머로 텍스트에서 음성, 음악, 효과음까지 다양한 오디오를 다국어로 생성하는 범용 텍스트-투-오디오 모델.",
    background:
      "기존 TTS 시스템은 깨끗한 음성 합성에 특화되어 있었으며, 웃음, 한숨, 음악 등 비언어적 오디오는 별도의 시스템이 필요했다. VALL-E가 코덱 기반 음성 합성의 가능성을 보여주었지만, 음성 외의 오디오 유형을 통합 생성하는 모델은 부재했다. 단일 모델에서 다양한 오디오 유형을 다국어로 생성하는 것이 도전적 과제였다.",
    keyIdea:
      "Bark는 GPT 아키텍처를 오디오 토큰 생성에 적용하여, 텍스트에서 다양한 유형의 오디오를 생성한다. 3단계 계층 구조를 사용하는데, (1) 시맨틱 모델이 텍스트에서 의미 토큰을 생성하고, (2) 거친 음향 모델이 의미 토큰에서 EnCodec의 처음 2개 코드북을 생성하며, (3) 세밀 음향 모델이 나머지 6개 코드북을 생성한다. 핵심 특징은 음성뿐 아니라 [laughs], [music] 등의 특수 태그로 비언어적 소리를 생성할 수 있으며, 13개 이상의 언어를 지원한다는 것이다.",
    method:
      "각 단계의 GPT 스타일 트랜스포머를 대규모 다국어 오디오-텍스트 데이터에서 학습한다. EnCodec을 오디오 토크나이저로 사용하며, 텍스트 입력에 화자 프롬프트를 추가하여 목소리 특성을 조절한다. 추론 시 각 단계의 온도(temperature)를 조절하여 생성 다양성을 제어한다.",
    results:
      "다국어 음성 합성에서 자연스러운 발화를 생성하며, 웃음, 한숨, 음악적 요소 등 비언어적 오디오도 포함할 수 있었다. 화자 프롬프트를 통한 제로샷 목소리 복제가 가능하며, 13개 이상의 언어에서 안정적 생성을 보였다.",
    impact:
      "오픈소스로 공개되어 커뮤니티에서 가장 널리 사용되는 텍스트-투-오디오 도구 중 하나가 되었다. 음성, 음악, 효과음을 단일 모델로 통합 생성하는 접근이 오디오 AI의 범용화 방향을 제시했으며, TTS 분야에서 오픈소스 대안의 중요한 이정표가 되었다.",
    relatedFoundations: ["transformer", "gpt"],
    relatedPapers: [
      { id: "vall-e", fieldId: "audio", title: "VALL-E", relation: "related" },
      { id: "encodec", fieldId: "audio", title: "Encodec", relation: "prior" },
      { id: "hubert", fieldId: "audio", title: "HuBERT", relation: "prior" },
    ],
  },

  // ===== Representation Learning Field =====
  "moco": {
    tldr: "모멘텀으로 업데이트되는 키 인코더와 큐(queue) 기반 딕셔너리를 도입하여, 대규모 배치 없이도 효과적인 대조 학습을 가능하게 한 자기지도 시각 표현 학습 프레임워크.",
    background:
      "대조 학습은 유사한 쌍은 가깝게, 다른 쌍은 멀게 하는 원리로 표현을 학습한다. 효과적인 대조 학습에는 많은 부정 예시(negative samples)가 필요한데, 이를 위해 SimCLR은 거대한 배치를, 메모리 뱅크 방식은 오래된 표현을 사용하여 각각 한계가 있었다. 크고 일관된 부정 예시 집합을 효율적으로 유지하는 메커니즘이 필요했다.",
    keyIdea:
      "MoCo(Momentum Contrast)는 대조 학습을 딕셔너리 탐색(dictionary look-up) 문제로 정의한다. 두 가지 핵심 메커니즘을 도입하는데, (1) 큐(queue) 기반 딕셔너리: 현재와 이전 미니배치의 키를 큐에 저장하여, 배치 크기에 관계없이 큰 딕셔너리(65536)를 유지한다. (2) 모멘텀 인코더: 키 인코더를 쿼리 인코더의 지수이동평균(EMA)으로 느리게 업데이트하여, 큐 내 키 표현의 일관성을 보장한다. 이로써 큰 배치나 메모리 뱅크 없이도 대규모의 일관된 부정 예시 집합을 확보한다.",
    method:
      "쿼리 인코더는 역전파로 업데이트하고, 키 인코더는 m=0.999의 모멘텀으로 EMA 업데이트한다. 각 미니배치에서 생성된 키를 큐에 추가하고, 가장 오래된 키를 제거한다. InfoNCE 손실로 쿼리-양성키 유사도를 최대화하고 쿼리-음성키 유사도를 최소화한다. ResNet-50으로 ImageNet에서 학습했다.",
    results:
      "ImageNet 선형 평가에서 60.6% top-1 정확도를 달성하여 당시 자기지도 방법 중 최고를 기록했다. PASCAL VOC 객체 검출에서 지도학습 사전학습을 앞서는 전이 학습 성능을 보였으며, 256 배치에서도 안정적 학습이 가능했다.",
    impact:
      "모멘텀 인코더와 큐 메커니즘은 이후 BYOL, DINO 등 자기지도 학습 연구의 핵심 설계 요소가 되었다. 대조 학습의 실용적 한계를 극복하여 자기지도 시각 표현 학습의 급격한 발전을 촉발했으며, MoCo v2, v3로 지속적으로 개선되었다.",
    relatedFoundations: ["resnet", "backpropagation"],
    relatedPapers: [
      { id: "simclr", fieldId: "representation", title: "SimCLR", relation: "related" },
      { id: "byol", fieldId: "representation", title: "BYOL", relation: "successor" },
      { id: "dino", fieldId: "representation", title: "DINO", relation: "successor" },
    ],
  },

  "swav": {
    tldr: "온라인 클러스터링과 멀티크롭 증강을 결합하여, 부정 쌍(negative pairs) 없이도 대조 학습에 필적하는 자기지도 시각 표현을 학습하는 프레임워크.",
    background:
      "SimCLR과 MoCo가 대조 학습의 효과를 보여주었지만, 대규모 부정 예시 집합에 의존하는 한계가 있었다. 클러스터링 기반 접근(DeepCluster 등)은 오프라인 클러스터링의 계산 비용이 크고 확장성이 떨어졌다. 부정 예시 없이도 표현 붕괴를 방지하면서 효과적으로 학습하는 방법이 필요했다.",
    keyIdea:
      "SwAV(Swapped Assignments between Views)는 대조 학습과 클러스터링을 통합한다. 같은 이미지의 두 뷰를 프로토타입(learnable prototypes)에 할당하고, 한 뷰의 할당이 다른 뷰에서도 일관되도록 학습한다(교차 예측). Sinkhorn-Knopp 알고리즘으로 온라인 클러스터 할당의 균형을 맞추어 모든 표현이 하나의 클러스터로 붕괴하는 것을 방지한다. 멀티크롭(multi-crop) 전략도 도입하는데, 2개의 글로벌 뷰(224x224)와 여러 개의 로컬 뷰(96x96)를 사용하여 계산 비용 증가 없이 학습 효율을 크게 높인다.",
    method:
      "3000개의 학습 가능한 프로토타입 벡터를 유지한다. 각 뷰의 표현을 프로토타입에 소프트 할당하되, Sinkhorn-Knopp으로 배치 내 할당이 균등하도록 정규화한다. 한 뷰의 프로토타입 할당을 다른 뷰의 특징으로 예측하는 크로스엔트로피 손실을 사용한다. 멀티크롭은 2x224 + 6x96으로 구성한다.",
    results:
      "ImageNet 선형 평가에서 75.3% top-1 정확도를 달성하여 SimCLR(69.3%)와 MoCo v2(71.1%)를 크게 앞섰다. 멀티크롭만으로 2% 이상 개선되었으며, 준지도 학습과 전이 학습에서도 우수한 성능을 보였다.",
    impact:
      "온라인 클러스터링과 멀티크롭 증강이라는 두 가지 핵심 기법을 도입하여 자기지도 학습의 효율성과 성능을 동시에 향상시켰다. 멀티크롭은 이후 DINO 등에서 표준 기법으로 채택되었으며, 클러스터링 기반 자기지도 학습의 확장 가능성을 실증했다.",
    relatedFoundations: ["resnet", "backpropagation"],
    relatedPapers: [
      { id: "simclr", fieldId: "representation", title: "SimCLR", relation: "prior" },
      { id: "moco", fieldId: "representation", title: "MoCo", relation: "related" },
      { id: "dino", fieldId: "representation", title: "DINO", relation: "successor" },
    ],
  },

  "beit": {
    tldr: "BERT의 마스크 토큰 예측 방식을 비전 트랜스포머(ViT)에 적용하여, 이미지 패치를 마스킹하고 시각적 토큰을 예측하는 사전학습으로 강력한 시각 표현을 학습한 논문.",
    background:
      "BERT가 NLP에서 마스크 언어 모델링으로 획기적 성공을 거두었지만, 비전에서의 마스크 예측 사전학습은 효과적이지 않았다. 이미지 패치는 텍스트 토큰과 달리 연속적이어서 예측 타겟을 정의하기 어려웠다. ViT의 등장으로 이미지를 패치 시퀀스로 처리할 수 있게 되었지만, 효과적인 자기지도 사전학습 방법은 아직 확립되지 않았다.",
    keyIdea:
      "BEiT(Bidirectional Encoder representation from Image Transformers)는 두 단계로 구성된다. 먼저 dVAE(discrete VAE) 토크나이저를 학습하여 이미지 패치를 이산적 시각 토큰으로 변환한다. 이후 ViT를 사전학습할 때, 입력 이미지의 약 40% 패치를 마스킹하고, 마스킹된 위치의 시각 토큰을 예측하도록 학습한다. 핵심은 원시 픽셀이 아닌 이산 시각 토큰을 예측함으로써, 모델이 저수준 세부사항보다 의미적으로 풍부한 표현을 학습하게 한다는 것이다. 블록 단위 마스킹(blockwise masking)을 사용하여 연속적인 패치 영역을 가리는 것도 효과적이다.",
    method:
      "DALL-E의 dVAE를 이미지 토크나이저로 사용하여 14x14 격자의 시각 토큰을 생성한다. ViT-B/16 인코더에서 약 40%의 패치를 블록 단위로 마스킹하고, 마스킹된 위치에 [MASK] 토큰을 삽입한다. 소프트맥스 분류기로 해당 위치의 시각 토큰(8192 클래스)을 예측한다. ImageNet-1K 300 에폭 사전학습 후 미세조정했다.",
    results:
      "ImageNet 미세조정에서 ViT-B 기준 83.2% top-1 정확도를 달성하여, DeiT(81.8%)와 DINO(82.8%)를 앞섰다. ADE20K 시맨틱 세그멘테이션에서도 사전학습의 효과가 두드러졌으며, 특히 저데이터 레짐에서 지도학습 사전학습 대비 큰 우위를 보였다.",
    impact:
      "BERT 스타일의 마스크 예측 사전학습이 비전에서도 효과적임을 최초로 대규모로 입증하여, MAE, data2vec 등 후속 마스크 이미지 모델링 연구의 직접적 선구자가 되었다. 이산 시각 토큰을 예측 타겟으로 사용하는 아이디어는 BEiT v2, PeCo 등으로 발전했다.",
    relatedFoundations: ["transformer", "bert", "vit", "vae"],
    relatedPapers: [
      { id: "mae", fieldId: "representation", title: "MAE", relation: "successor" },
      { id: "dino", fieldId: "representation", title: "DINO", relation: "related" },
      { id: "dall-e", fieldId: "multimodal", title: "DALL-E", relation: "related" },
    ],
  },

  "data2vec": {
    tldr: "음성, 비전, 언어의 세 가지 모달리티에서 동일한 교사-학생 자기지도 학습 프레임워크를 적용하여, 통합된 자기지도 표현 학습 방법론을 제시한 논문.",
    background:
      "자기지도 학습은 각 모달리티에서 개별적으로 발전해왔다. NLP에서는 마스크 토큰 예측, 비전에서는 대조 학습이나 마스크 패치 예측, 음성에서는 HuBERT의 클러스터 기반 예측이 주류였다. 모달리티마다 다른 방법론을 사용해야 하는 비효율성을 해소하고, 통합된 프레임워크를 구축할 수 있는지가 핵심 질문이었다.",
    keyIdea:
      "data2vec의 핵심 아이디어는 모든 모달리티에서 '마스킹된 입력의 학생 모델이 마스킹되지 않은 입력의 교사 모델의 잠재 표현을 예측하는' 동일한 프레임워크를 사용하는 것이다. 교사는 학생의 EMA로 업데이트된다. 기존 방법들과의 핵심 차이는 예측 타겟이 이산 토큰이나 픽셀이 아닌, 교사의 상위 레이어 표현을 평균한 연속적 잠재 벡터라는 것이다. 이 연속적 타겟이 더 풍부한 학습 신호를 제공하여, 모달리티별 귀납적 편향을 최소화하면서도 강력한 표현을 학습한다.",
    method:
      "트랜스포머 기반 모델에서 입력의 일부를 마스킹(음성/비전: 연속 마스크, 텍스트: 토큰 마스크)한다. 교사 모델은 마스킹되지 않은 전체 입력을 받고, 상위 K개 층의 표현을 평균하여 타겟을 생성한다. 학생은 마스킹된 입력을 받아 마스킹 위치에서 교사 타겟을 평균 제곱근(smooth L1) 손실로 예측한다. 음성은 wav2vec 2.0, 비전은 ViT, 텍스트는 RoBERTa 아키텍처를 사용했다.",
    results:
      "음성 인식(LibriSpeech)에서 wav2vec 2.0과 HuBERT를 앞서는 성능을 달성했고, 이미지 분류(ImageNet)에서 BEiT를 능가했으며, 자연어 이해(GLUE)에서도 RoBERTa에 필적했다. 세 모달리티 모두에서 동일한 프레임워크가 경쟁력 있는 성능을 보였다.",
    impact:
      "모달리티에 구애받지 않는 통합 자기지도 학습의 가능성을 최초로 대규모로 실증했다. data2vec 2.0에서 효율성을 크게 개선했으며, 범용 AI 시스템 구축을 위한 통합 표현 학습이라는 비전을 제시하여 멀티모달 기초 모델 연구에 영향을 미쳤다.",
    relatedFoundations: ["transformer", "bert", "vit"],
    relatedPapers: [
      { id: "mae", fieldId: "representation", title: "MAE", relation: "related" },
      { id: "byol", fieldId: "representation", title: "BYOL", relation: "prior" },
      { id: "hubert", fieldId: "audio", title: "HuBERT", relation: "related" },
    ],
  },

  "vicreg": {
    tldr: "분산(Variance), 불변(Invariance), 공분산(Covariance)의 세 가지 정규화 항으로 표현 붕괴를 방지하여, 부정 쌍·모멘텀·클러스터링 없이 경쟁력 있는 자기지도 시각 표현을 학습하는 방법.",
    background:
      "SimCLR은 부정 쌍, MoCo는 모멘텀 인코더, BYOL은 EMA + 비대칭 예측기, SwAV는 온라인 클러스터링으로 각각 표현 붕괴를 방지했다. 이러한 다양한 메커니즘 중 어떤 원리가 본질적인지, 더 간단하고 직접적인 정규화 방식으로 붕괴를 방지할 수 있는지가 근본적 질문이었다.",
    keyIdea:
      "VICReg는 세 가지 명시적 정규화 항으로 표현 붕괴를 직접 방지한다. (1) Invariance: 같은 이미지의 두 뷰 표현 간 MSE를 최소화한다. (2) Variance: 배치 내 각 특징 차원의 분산이 임계값 이상을 유지하도록 강제하여, 모든 표현이 같아지는 붕괴를 방지한다. (3) Covariance: 서로 다른 특징 차원 간의 공분산을 0으로 만들어, 차원 간 중복(redundancy)을 제거한다. 이 세 항의 조합이 부정 쌍, 모멘텀 업데이트, 클러스터링 등 복잡한 메커니즘 없이도 충분한 정규화를 제공한다.",
    method:
      "ResNet-50 인코더와 3층 MLP 프로젝터를 사용한다. 두 증강 뷰의 프로젝션 출력에 대해 세 가지 손실을 계산한다. 분산 항은 hinge 손실로 각 차원의 표준편차가 1 이상이 되도록 하고, 공분산 항은 상관 행렬의 비대각 원소를 0으로 만든다. 손실 가중치 λ_inv=25, λ_var=25, λ_cov=1로 설정했다. ImageNet에서 1000 에폭 학습했다.",
    results:
      "ImageNet 선형 평가에서 73.2% top-1 정확도로, BYOL(74.3%)에 근접하면서 SimCLR(69.3%)를 크게 앞섰다. 전이 학습에서 다양한 벤치마크에서 경쟁력 있는 성능을 보였으며, 멀티모달 학습으로의 자연스러운 확장(VICRegL)도 가능했다.",
    impact:
      "자기지도 학습에서 표현 붕괴 방지의 본질을 분산과 공분산 정규화로 명확히 규명하여, 이론적 이해를 깊게 했다. 부정 쌍 없는 학습의 원리를 가장 직관적으로 설명하는 프레임워크로, Barlow Twins와 함께 정규화 기반 자기지도 학습의 핵심 연구로 자리잡았다.",
    relatedFoundations: ["resnet", "backpropagation"],
    relatedPapers: [
      { id: "byol", fieldId: "representation", title: "BYOL", relation: "prior" },
      { id: "simclr", fieldId: "representation", title: "SimCLR", relation: "prior" },
      { id: "dino", fieldId: "representation", title: "DINO", relation: "related" },
    ],
  },

  "dinov2": {
    tldr: "자동 큐레이션된 대규모 데이터, 개선된 자기지도 학습, 지식 증류를 결합하여 미세조정 없이도 다양한 비전 태스크에서 범용적으로 작동하는 시각 특징을 학습한 논문.",
    background:
      "DINO v1이 ViT의 자기지도 학습에서 인상적인 성질(자동 세그멘테이션 등)을 보였지만, ImageNet-1K에서만 학습하여 데이터 다양성에 한계가 있었다. 또한 NLP에서 기초 모델(GPT, BERT 등)이 범용적 표현을 제공하는 반면, 비전에서는 태스크마다 미세조정이 필수적이었다. 미세조정 없이 범용적으로 작동하는 시각 기초 모델이 부재했다.",
    keyIdea:
      "DINOv2는 세 가지 핵심 개선을 도입한다. (1) 자동 데이터 큐레이션: 웹에서 수집한 비큐레이트 데이터를 ImageNet의 의미적 다양성에 맞춰 자동으로 필터링하여 LVD-142M 데이터셋을 구축한다. 코사인 유사도 기반 중복 제거와 균형 잡힌 클러스터 샘플링을 수행한다. (2) 학습 안정화: DINO의 자기증류 + iBOT의 마스크 이미지 모델링을 결합하고, KoLeo 정규화, Sinkhorn-Knopp, 적응적 학습률 등으로 대규모 학습의 안정성을 확보한다. (3) 지식 증류: ViT-g/14 교사를 먼저 학습한 뒤, 더 작은 모델(ViT-S/B/L)로 증류하여 효율적 모델군을 구축한다.",
    method:
      "LVD-142M 데이터셋에서 ViT-g/14를 DINO+iBOT 결합 손실로 학습한다. DINO 헤드(이미지 수준 자기증류)와 iBOT 헤드(패치 수준 마스크 예측)를 동시에 최적화한다. 학습된 ViT-g에서 ViT-S/14, ViT-B/14, ViT-L/14로 증류한다. A100 GPU 클러스터에서 대규모 분산 학습을 수행했다.",
    results:
      "ImageNet k-NN 분류에서 ViT-g 기준 83.5%로 미세조정 없이 최고 성능을 달성했다. 깊이 추정, 시맨틱 세그멘테이션, 인스턴스 검색 등에서 선형 프로브만으로 태스크별 미세조정 모델에 필적하거나 능가했다. OpenCLIP 대비 이미지 분류와 검색 모두에서 우수했다.",
    impact:
      "비전 분야에서 '미세조정 없는 범용 특징'이라는 기초 모델의 비전을 사실상 처음으로 실현했다. 오픈소스로 공개되어 의료 이미지 분석, 위성 영상, 로봇 비전 등 다양한 실용 분야에서 기본 백본으로 채택되고 있으며, 비전 기초 모델의 새로운 표준을 확립했다.",
    relatedFoundations: ["transformer", "vit"],
    relatedPapers: [
      { id: "dino", fieldId: "representation", title: "DINO", relation: "prior" },
      { id: "mae", fieldId: "representation", title: "MAE", relation: "related" },
      { id: "beit", fieldId: "representation", title: "BEiT", relation: "prior" },
    ],
  },
};
