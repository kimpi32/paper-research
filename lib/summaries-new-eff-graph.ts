import type { PaperSummary } from "./paper-summaries";

export const newEffGraphSummaries: Record<string, PaperSummary> = {
  // ============================================================
  // Efficient Field (efficient) - 7 Papers
  // ============================================================

  "moe": {
    tldr: "Mixture of Experts(MoE) 라우팅을 단순화하여 각 토큰이 하나의 전문가만 선택하는 Switch Transformer를 제안했다. 이를 통해 통신 비용과 학습 불안정성을 줄이면서 조 단위 파라미터 규모로 효율적으로 확장할 수 있음을 보였다.",
    background: "Transformer 모델의 성능은 파라미터 수에 따라 향상되지만, 밀집(dense) 모델은 모든 입력에 대해 전체 파라미터를 활성화하므로 계산 비용이 파라미터 수에 비례하여 증가한다. Mixture of Experts(MoE)는 입력에 따라 일부 파라미터만 활성화하여 이 문제를 해결할 수 있지만, 기존 MoE(Shazeer et al., 2017)는 top-k(보통 k=2) 전문가를 선택하는 라우팅의 복잡성, 전문가 간 부하 불균형, 학습 불안정성 등의 문제가 있어 대규모 적용이 어려웠다. 특히 분산 학습 환경에서 다수 전문가 간 통신 비용이 병목이었다.",
    keyIdea: "Switch Transformer의 핵심 아이디어는 MoE 라우팅을 극단적으로 단순화하는 것이다. 기존의 top-2 이상 전문가 선택 대신 각 토큰이 정확히 하나의 전문가(top-1)만 선택하는 'Switch 라우팅'을 도입한다. 이는 직관에 반하지만, 라우터 계산량 감소, 전문가 용량(capacity factor) 절반화, 통신 비용 감소라는 세 가지 이점을 동시에 달성한다. 또한 전문가 부하 균형을 위한 보조 손실(auxiliary load-balancing loss)을 단순화하고, bfloat16 정밀도로 학습 안정성을 확보하며, 소수 전문가에서 다수 전문가로 점진적으로 확장하는 전략을 제안한다.",
    method: "Transformer의 각 피드포워드(FFN) 레이어를 N개의 독립적인 전문가 FFN으로 대체하고, 학습 가능한 라우터 네트워크가 각 토큰을 하나의 전문가에 할당한다. 라우터는 토큰 임베딩에 대한 선형 변환 후 소프트맥스를 적용하여 확률 분포를 생성하고, 가장 높은 확률의 전문가를 선택한다. 전문가 용량 팩터(capacity factor)를 설정하여 한 전문가에 과도한 토큰이 할당되는 것을 방지하며, 초과 토큰은 잔차 연결을 통해 다음 레이어로 전달된다. 부하 균형 손실 alpha * N * sum(f_i * P_i)를 추가하여 전문가 활용의 균형을 유도한다. 1.6조 파라미터 모델까지 확장하여 C4 데이터셋에서 학습했다.",
    results: "Switch Transformer는 동일한 계산 예산(FLOPS)에서 밀집 T5 모델 대비 최대 7배 빠른 사전학습 속도를 달성했다. T5-Base 규모에서 Switch-Base(128 전문가)는 동일 학습 시간 대비 밀집 모델보다 현저히 낮은 perplexity를 보였다. T5-XXL과 비교하여 Switch-XXL은 4배 적은 학습 스텝으로 동등한 성능에 도달했다. 다운스트림 태스크(SuperGLUE, ARC, Winogrande 등)에서도 밀집 모델을 일관되게 능가했으며, 1.6T 파라미터 모델의 안정적 학습을 시연했다.",
    impact: "Switch Transformer는 MoE 아키텍처를 대규모 언어 모델에 실용적으로 적용할 수 있는 경로를 열었다. top-1 라우팅의 성공은 이후 GShard, GLaM, Mixtral 등 MoE 기반 모델 설계에 직접적 영향을 미쳤으며, 특히 Mixtral 8x7B의 상업적 성공에 이론적 기반을 제공했다. 희소 활성화(sparse activation)를 통해 추론 효율성을 유지하면서 모델 용량을 확장하는 패러다임은 현재 LLM 아키텍처 연구의 핵심 방향 중 하나가 되었다.",
    relatedFoundations: ["transformer", "scaling-laws"],
    relatedPapers: [
      { id: "mixtral", fieldId: "llm", title: "Mixtral of Experts", relation: "successor" },
      { id: "flash-attention", fieldId: "efficient", title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", relation: "related" },
    ],
  },

  "flash-attention": {
    tldr: "GPU 메모리 계층 구조를 고려한 IO-aware 알고리즘으로 어텐션 연산을 재구성하여, 근사 없이 정확한(exact) 어텐션을 기존 대비 2-4배 빠르게 수행하면서 메모리 사용량을 시퀀스 길이에 대해 선형으로 줄였다.",
    background: "Transformer의 셀프 어텐션은 시퀀스 길이 N에 대해 O(N^2)의 시간 및 메모리 복잡도를 가져 긴 시퀀스 처리에 심각한 병목이었다. 기존 연구들은 이를 해결하기 위해 sparse attention(Longformer), low-rank approximation(Linformer), kernel-based methods(Performer) 등 근사 어텐션(approximate attention)을 제안했으나, 정확도 손실이 불가피했고 실제 wall-clock 속도 향상이 제한적이었다. 이는 기존 접근들이 FLOPS 절감에만 집중하고, GPU의 메모리 접근 패턴(IO 복잡도)을 무시했기 때문이다.",
    keyIdea: "FlashAttention의 핵심 통찰은 어텐션 연산의 병목이 산술 연산(FLOPS)이 아니라 GPU HBM(고대역폭 메모리)과 SRAM(온칩 캐시) 간의 데이터 이동(IO)이라는 것이다. 표준 어텐션은 N x N 크기의 어텐션 행렬 전체를 HBM에 저장하고 다시 읽어야 하므로 O(N^2)의 메모리 접근이 발생한다. FlashAttention은 타일링(tiling)과 재계산(recomputation) 기법을 결합하여, 어텐션 행렬을 절대 HBM에 저장하지 않고 SRAM 내에서 블록 단위로 계산을 완료한다. 소프트맥스의 온라인 정규화(online softmax normalization)를 통해 블록 단위 계산에서도 수학적으로 정확한 결과를 보장한다.",
    method: "입력 Q, K, V 행렬을 SRAM에 적재 가능한 블록 크기로 분할한다. 외부 루프에서 K, V의 블록을, 내부 루프에서 Q의 블록을 순회하며, 각 블록 조합에 대해 부분 어텐션 출력을 SRAM 내에서 계산한다. 소프트맥스의 분자와 분모를 별도로 추적하는 온라인 소프트맥스 기법을 사용하여 블록 간 결과를 점진적으로 병합한다. 역전파 시에는 어텐션 행렬을 저장하지 않고 Q, K, V와 출력 통계량(row-wise max, sum)만 저장한 뒤 필요 시 재계산한다. 이를 통해 메모리 사용량이 O(N)으로 감소한다. CUDA 커널로 구현되어 Fused 연산을 통해 커널 론치 오버헤드도 제거한다.",
    results: "FlashAttention은 표준 PyTorch 어텐션 대비 2-4배의 wall-clock 속도 향상과 5-20배의 메모리 절감을 달성했다. GPT-2 학습에서 HuggingFace 및 Megatron-LM 구현 대비 최대 3배 빠른 학습 속도를 보였다. 시퀀스 길이를 최대 16K까지 확장할 수 있게 하여, Long Range Arena 벤치마크에서 기존 근사 어텐션 방법들을 능가하는 정확도를 달성했다. Path-X(16K 길이) 태스크에서 Transformer가 최초로 랜덤 이상의 성능을 달성했다.",
    impact: "FlashAttention은 Transformer 추론 및 학습의 효율성을 획기적으로 개선한 시스템 수준의 혁신이다. PyTorch 2.0의 기본 어텐션 구현으로 통합되었으며, 현재 거의 모든 LLM 학습 및 추론 프레임워크(Hugging Face, DeepSpeed, Megatron 등)에서 사용되고 있다. IO-aware 알고리즘 설계라는 원칙은 이후 FlashAttention-2, FlashAttention-3, FlashDecoding 등 후속 최적화의 기반이 되었으며, 하드웨어 인식 알고리즘 연구의 중요성을 학계에 각인시켰다.",
    relatedFoundations: ["transformer", "attention"],
    relatedPapers: [
      { id: "flash-attention-2", fieldId: "efficient", title: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", relation: "successor" },
      { id: "vllm", fieldId: "efficient", title: "Efficient Memory Management for Large Language Model Serving with PagedAttention", relation: "related" },
    ],
  },

  "gptq": {
    tldr: "대규모 생성 사전학습 모델(GPT)을 재학습 없이 한 번에(one-shot) 3-4비트로 양자화하는 GPTQ를 제안하여, 175B 파라미터 모델을 단일 GPU에서 실행할 수 있게 했다. 양자화 과정은 수 시간 내에 완료되며 정확도 손실이 미미하다.",
    background: "GPT-3(175B), OPT-175B, BLOOM-176B 등 초대규모 언어 모델은 뛰어난 성능을 보이지만, 추론 시 수백 GB의 메모리가 필요하여 여러 대의 고가 GPU가 요구되었다. 가중치 양자화는 모델 크기를 줄이는 효과적 방법이지만, 기존 양자화 기법들(AdaRound, BRECQ 등)은 수십억 파라미터 규모에서 비실용적이거나 정확도 손실이 컸다. 특히 사후 학습 양자화(post-training quantization, PTQ)는 재학습이 필요 없어 실용적이지만, 대규모 생성 모델에 적용하기 위한 효율적이고 정확한 방법이 부재했다.",
    keyIdea: "GPTQ는 최적 뇌 양자화(Optimal Brain Quantization, OBQ) 프레임워크를 대규모 모델에 확장하되, 핵심적인 계산 효율성 개선을 도입한다. OBQ는 가중치를 하나씩 양자화하면서 나머지 가중치를 갱신하여 양자화 오차를 보상하지만, 이 순차적 처리가 대규모 모델에서는 비실용적이다. GPTQ의 핵심 혁신은 세 가지이다: (1) 가중치 양자화 순서를 임의(arbitrary)로 고정하여도 성능 저하가 미미하다는 발견(lazy batch updates), (2) 같은 열(column)의 모든 행을 동시에 양자화하여 행렬 연산으로 병렬화, (3) Cholesky 분해를 활용한 수치적으로 안정적인 역 헤시안 갱신. 이를 통해 OBQ 대비 수백 배 빠른 양자화를 달성한다.",
    method: "각 레이어의 가중치 행렬을 독립적으로 양자화한다. 소량의 보정(calibration) 데이터(보통 128개 샘플)를 사용하여 각 레이어의 입력 활성화에 대한 헤시안 행렬 H = 2X^TX를 계산한다. 가중치 열을 128개 단위 블록으로 나누어, 블록 내에서는 순차적으로 각 열을 양자화하고 블록 내 나머지 열의 가중치를 갱신(lazy batch updates)한 뒤, 블록이 끝나면 나머지 전체 열에 대해 일괄 갱신을 수행한다. 이 과정을 Cholesky 분해된 역 헤시안을 사용하여 수치적으로 안정화한다. 그룹 양자화(group quantization)를 적용하여 연속된 가중치 그룹별로 별도의 양자화 파라미터를 사용할 수 있다.",
    results: "GPTQ는 OPT-175B와 BLOOM-176B를 3비트로 양자화했을 때 perplexity 증가가 0.5 미만이었다. OPT-175B를 3비트로 양자화하면 약 63GB로 단일 A100 80GB GPU에 적재 가능해졌다. 양자화 시간은 OPT-175B 기준 약 4 GPU-시간으로, 기존 OBQ 대비 수백 배 빠르다. 4비트 양자화에서는 거의 무손실(lossless)에 가까운 정확도를 유지했으며, 제로샷 태스크(LAMBADA, ARC, PIQA 등)에서도 원본 모델과 유사한 성능을 보였다.",
    impact: "GPTQ는 대규모 LLM 양자화의 사실상 표준으로 자리잡아, 오픈소스 LLM 생태계의 접근성을 크게 높였다. AutoGPTQ 라이브러리를 통해 HuggingFace에 통합되어 수천 개의 양자화 모델이 공유되고 있으며, TheBloke 등 커뮤니티 기여자들이 주요 모델의 GPTQ 양자화 버전을 지속적으로 제공하고 있다. 이후 AWQ, SqueezeLLM, QuIP 등 후속 양자화 연구에 직접적 영감을 제공했으며, 소비자 하드웨어에서 LLM 실행이라는 새로운 사용 사례를 개척했다.",
    relatedFoundations: ["transformer", "gpt"],
    relatedPapers: [
      { id: "awq", fieldId: "efficient", title: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration", relation: "successor" },
      { id: "vllm", fieldId: "efficient", title: "Efficient Memory Management for Large Language Model Serving with PagedAttention", relation: "related" },
    ],
  },

  "awq": {
    tldr: "모든 가중치가 동등하게 중요하지 않으며, 활성화 분포를 기반으로 소수의 핵심 가중치 채널을 식별하여 보호하는 활성화 인식 양자화 AWQ를 제안했다. 이를 통해 GPTQ보다 빠른 양자화와 더 나은 일반화를 달성했다.",
    background: "LLM의 배포와 서빙을 위해 모델 압축이 필수적이었으며, GPTQ가 사후 학습 양자화의 효과를 입증한 이후 더 효율적이고 하드웨어 친화적인 양자화 방법에 대한 수요가 증가했다. GPTQ는 보정 데이터에 의존하여 가중치를 재조정하므로 과적합 위험이 있고, 양자화 과정 자체가 상대적으로 느렸다. 또한 양자화된 모델의 하드웨어 가속 실행(efficient kernel)에 대한 고려가 부족했다.",
    keyIdea: "AWQ의 핵심 관찰은 가중치의 중요도가 가중치 자체의 크기가 아니라 대응하는 활성화(activation)의 크기에 의해 결정된다는 것이다. 활성화 크기가 큰 채널의 가중치는 전체 출력에 대한 기여가 크므로 양자화 오차에 더 민감하다. AWQ는 이러한 핵심 채널의 가중치를 보호하기 위해, 양자화 전에 채널별 스케일링 팩터를 적용하여 중요한 가중치의 유효 범위를 확대한다. 이 스케일링은 가중치의 양자화 그리드를 조밀하게 만들어 양자화 오차를 줄이며, 역스케일링은 다음 레이어의 활성화에 흡수시킨다. 최적 스케일링 팩터는 보정 데이터에 대한 그리드 탐색으로 결정한다.",
    method: "각 레이어에 대해 소량의 보정 데이터(보통 128개)를 순전파하여 채널별 활성화 크기의 평균을 계산한다. 활성화 크기가 큰 상위 1%의 채널을 핵심 채널로 식별한다. 채널별 스케일링 팩터 s를 탐색하되, 스케일링 후 양자화된 출력과 원본 출력 간의 MSE를 최소화하는 s를 그리드 탐색으로 찾는다. 스케일링된 가중치에 대해 표준 그룹 양자화(group-size 128)를 적용한다. 양자화된 모델의 효율적 실행을 위해 W4A16(4비트 가중치, 16비트 활성화) 커널을 구현하여, 메모리 대역폭 병목 상황에서 실제 추론 속도를 향상시킨다.",
    results: "AWQ는 LLaMA-1/2(7B~70B), OPT(6.7B~66B) 등에서 4비트 양자화 시 GPTQ와 동등하거나 우수한 perplexity를 달성했다. 특히 보정 데이터와 다른 도메인의 평가에서 GPTQ 대비 더 나은 일반화를 보여, 양자화 과정에서의 과적합이 적음을 확인했다. 양자화 속도는 GPTQ 대비 수배 빠르다. 맞춤 W4A16 커널은 FP16 대비 3.2배 메모리 절감과 함께 1.45배의 추론 속도 향상을 달성했다. 지시 미세조정 모델(Vicuna)에서도 양자화 후 다중 도메인 벤치마크 성능이 잘 유지되었다.",
    impact: "AWQ는 GPTQ와 함께 LLM 양자화의 양대 표준으로 자리잡았으며, 특히 하드웨어 효율적 배포를 중시하는 산업계에서 널리 채택되고 있다. NVIDIA TensorRT-LLM에 통합되어 상용 추론 서빙에 활용되고 있으며, HuggingFace에서도 AutoAWQ를 통해 폭넓게 지원된다. 활성화 인식이라는 관점은 이후 SmoothQuant, QuIP, AQLM 등 후속 양자화 연구에 중요한 설계 원칙을 제공했다.",
    relatedFoundations: ["transformer", "gpt"],
    relatedPapers: [
      { id: "gptq", fieldId: "efficient", title: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", relation: "prior" },
      { id: "vllm", fieldId: "efficient", title: "Efficient Memory Management for Large Language Model Serving with PagedAttention", relation: "related" },
    ],
  },

  "speculative-decoding": {
    tldr: "작은 드래프트 모델이 여러 토큰을 빠르게 생성한 뒤, 큰 타겟 모델이 이를 한 번에 병렬 검증하는 추측적 디코딩을 제안했다. 이 방법은 타겟 모델의 출력 분포를 정확히 보존하면서 2-3배의 추론 속도 향상을 달성한다.",
    background: "자기 회귀적(autoregressive) 언어 모델의 추론은 본질적으로 순차적이다. 각 토큰 생성에 전체 모델의 순전파가 필요하며, 이는 배치 크기가 작을 때 GPU 계산 자원의 활용률이 매우 낮아지는(memory-bandwidth bound) 문제를 야기한다. 특히 대규모 모델(100B+)에서 단일 토큰 생성의 지연 시간(latency)이 길어 실시간 응용에 제약이 있었다. 모델 병렬화, 양자화 등의 최적화가 연구되었지만, 자기 회귀 디코딩의 순차적 특성 자체를 해결하는 방법은 부족했다.",
    keyIdea: "추측적 디코딩(speculative decoding)은 '대부분의 토큰은 예측하기 쉬우며, 작은 모델도 올바르게 생성할 수 있다'는 직관에 기반한다. 작은 드래프트 모델(draft model)이 K개의 토큰을 자기 회귀적으로 빠르게 생성하고, 큰 타겟 모델(target model)이 이 K개 토큰에 대해 단 한 번의 순전파로 각 위치의 확률 분포를 병렬 계산한다. 그런 다음 수정된 거부 샘플링(modified rejection sampling) 기법으로 드래프트 토큰을 앞에서부터 순서대로 승인하거나 거부한다. 수학적으로 이 과정은 타겟 모델에서 직접 샘플링한 것과 동일한 분포를 보장하므로, 출력 품질의 저하가 전혀 없다.",
    method: "각 디코딩 스텝에서 드래프트 모델 M_q가 현재 컨텍스트에서 gamma개의 토큰을 자기 회귀적으로 생성하며, 각 위치의 드래프트 분포 q(x)를 저장한다. 타겟 모델 M_p는 원본 컨텍스트에 드래프트 토큰을 붙인 시퀀스에 대해 한 번의 순전파를 수행하여 각 위치의 타겟 분포 p(x)를 계산한다. 각 드래프트 토큰 x에 대해 min(1, p(x)/q(x))의 확률로 승인한다. 거부된 위치에서는 수정 분포 max(0, p(x)-q(x))에서 재샘플링하고, 모든 토큰이 승인되면 추가로 p에서 한 토큰을 더 샘플링한다. 이를 통해 한 라운드에서 평균적으로 1/(1-alpha) + 1개의 토큰을 생성하며, alpha는 드래프트와 타겟 분포의 일치도이다.",
    results: "T5-XXL(11B) 모델에서 T5-Small을 드래프트 모델로 사용했을 때, 텍스트 요약과 번역 태스크에서 2-3배의 추론 지연 시간 감소를 달성했다. 출력 분포가 타겟 모델과 수학적으로 동일함을 이론적으로 증명하고 실험적으로 검증했다. 드래프트 길이 gamma의 최적값은 태스크와 모델 쌍에 따라 4-8 범위에서 결정되었다. 채팅과 같은 다양한 생성 태스크에서 일관된 속도 향상을 보여 방법의 범용성을 입증했다.",
    impact: "추측적 디코딩은 LLM 추론 최적화의 핵심 기법으로 자리잡아, Google(PaLM/Gemini 서빙), Meta, Anthropic 등 주요 AI 기업의 프로덕션 서빙 스택에 통합되었다. 이 아이디어는 Medusa(다중 헤드 드래프트), EAGLE(자기 드래프트), SpecInfer(트리 기반 검증) 등 다양한 변형을 촉발했다. 드래프트 모델 없이도 적용 가능한 self-speculative decoding, 트리 구조 검증(tree attention) 등으로 발전하고 있으며, vLLM 등 주요 서빙 프레임워크에서 기본 지원된다.",
    relatedFoundations: ["transformer"],
    relatedPapers: [
      { id: "vllm", fieldId: "efficient", title: "Efficient Memory Management for Large Language Model Serving with PagedAttention", relation: "related" },
      { id: "flash-attention", fieldId: "efficient", title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", relation: "related" },
    ],
  },

  "vllm": {
    tldr: "운영체제의 가상 메모리 및 페이징 기법에서 영감받은 PagedAttention을 제안하여 KV 캐시 메모리를 비연속적 블록으로 관리함으로써, LLM 서빙의 처리량을 기존 시스템 대비 2-4배 향상시켰다.",
    background: "LLM 서빙에서 KV 캐시(key-value cache)는 자기 회귀적 디코딩의 효율성을 위해 필수적이지만, 전체 GPU 메모리의 상당 부분(최대 30% 이상)을 차지한다. 기존 서빙 시스템(FasterTransformer, Orca 등)은 각 요청의 KV 캐시를 연속적(contiguous) 메모리 공간에 할당했는데, 이는 세 가지 심각한 비효율을 초래했다: (1) 최대 시퀀스 길이를 기준으로 사전 할당하여 사용되지 않는 메모리가 많고(내부 단편화), (2) 요청 간 빈 공간이 발생하며(외부 단편화), (3) 빔 서치 등에서 KV 캐시를 복제해야 해 메모리 낭비가 발생한다. 이러한 비효율로 실제 활용률은 20-40%에 불과했다.",
    keyIdea: "PagedAttention은 KV 캐시를 고정 크기의 비연속적 블록(page)으로 분할하여 관리한다. 운영체제의 가상 메모리 시스템처럼, 논리적으로 연속인 KV 캐시를 물리적으로 비연속적인 블록에 저장하고 블록 테이블(page table)로 매핑한다. 이를 통해 (1) 내부 단편화를 마지막 블록에만 한정하고, (2) 외부 단편화를 완전히 제거하며, (3) copy-on-write 메커니즘으로 빔 서치나 병렬 샘플링에서 KV 캐시를 물리적으로 공유할 수 있다. 블록 크기는 보통 16 토큰으로 설정되며, 이는 OS의 4KB 페이지 크기에 해당하는 역할을 한다.",
    method: "각 어텐션 헤드의 KV 캐시를 고정 블록 크기(B 토큰)의 물리 블록으로 분할하고, 요청별 블록 테이블로 논리 블록과 물리 블록의 매핑을 관리한다. 새 토큰 생성 시 현재 블록에 여유가 있으면 추가하고, 없으면 새 물리 블록을 할당한다. 어텐션 계산 시 블록 테이블을 참조하여 비연속적 물리 블록에서 KV를 fetch하는 맞춤 CUDA 커널을 구현했다. 병렬 샘플링에서 같은 프롬프트의 KV 캐시를 공유할 때는 참조 카운트를 증가시키고, 수정 시에만 복사하는 copy-on-write를 적용한다. 이 시스템을 vLLM이라는 오픈소스 서빙 엔진으로 구현했다.",
    results: "vLLM은 동일한 지연 시간(latency) 조건에서 HuggingFace Transformers 대비 최대 24배, FasterTransformer 대비 최대 3.5배의 처리량(throughput) 향상을 달성했다. KV 캐시 메모리 낭비를 기존 60-80%에서 4% 미만으로 줄였다. 병렬 샘플링에서는 KV 캐시 공유를 통해 55%의 메모리 절감을 보였다. 빔 서치에서도 copy-on-write를 통해 메모리 사용량이 빔 수에 비례하여 증가하지 않았다. OPT-13B, OPT-175B 등 다양한 모델 크기에서 일관된 성능 향상을 확인했다.",
    impact: "vLLM은 오픈소스 LLM 서빙의 사실상 표준 프레임워크로 자리잡아, 학술 연구와 산업 프로덕션 모두에서 가장 널리 사용되는 LLM 추론 엔진이 되었다. PagedAttention의 아이디어는 TensorRT-LLM, TGI(Text Generation Inference) 등 다른 서빙 프레임워크에도 채택되었다. 지속적 배칭(continuous batching), 추측적 디코딩, 양자화(GPTQ/AWQ) 등과의 통합을 통해 LLM 서빙의 전체 스택을 커버하는 생태계로 발전하고 있다. SOSP 2023에서 발표되어 시스템 커뮤니티와 ML 커뮤니티의 교류를 촉진했다.",
    relatedFoundations: ["transformer", "attention"],
    relatedPapers: [
      { id: "flash-attention", fieldId: "efficient", title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", relation: "related" },
      { id: "speculative-decoding", fieldId: "efficient", title: "Fast Inference from Transformers via Speculative Decoding", relation: "related" },
      { id: "gptq", fieldId: "efficient", title: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", relation: "related" },
    ],
  },

  "flash-attention-2": {
    tldr: "FlashAttention의 알고리즘을 GPU의 병렬화와 작업 분할 측면에서 재설계하여, 비인과적 어텐션에서 이론적 최대 FLOPS의 약 70%를 달성하고, 원래 FlashAttention 대비 약 2배의 속도 향상을 이루었다.",
    background: "FlashAttention이 IO-aware 어텐션의 실용성을 입증한 후, GPU 하드웨어 활용률을 더 높이기 위한 최적화가 필요했다. FlashAttention v1은 A100 GPU의 이론적 최대 FLOPS의 약 25-35%만 활용하고 있었는데, 이는 GPU 스레드 블록 간 작업 분배의 비효율, 불필요한 공유 메모리 읽기/쓰기, 워프(warp) 간 동기화 오버헤드 등에 기인했다. 특히 인과적(causal) 마스킹이 적용된 경우 삼각 형태의 불균등한 워크로드로 인해 GPU 점유율이 더욱 낮았다.",
    keyIdea: "FlashAttention-2는 세 가지 핵심 최적화를 도입한다. 첫째, 알고리즘의 루프 구조를 변경하여 외부 루프를 Q 블록에, 내부 루프를 K/V 블록에 대해 순회하도록 재구성한다. 이를 통해 각 스레드 블록이 하나의 Q 블록에 대한 전체 어텐션 출력을 독립적으로 계산하므로, 스레드 블록 간 통신(synchronization)이 불필요해진다. 둘째, 워프 내 작업 분할을 최적화하여 공유 메모리(shared memory) 접근과 동기화를 최소화한다. 기존에는 K를 워프 간에 분할했지만, FlashAttention-2는 Q를 워프 간에 분할하여 리덕션(reduction) 단계의 공유 메모리 읽기/쓰기를 제거한다. 셋째, 시퀀스 길이 차원에서의 병렬화를 추가하여, 배치 크기나 헤드 수가 적을 때도 GPU SM(Streaming Multiprocessor)을 충분히 활용한다.",
    method: "Q를 외부 루프로, K/V를 내부 루프로 하는 타일링을 적용한다. 각 스레드 블록은 하나의 Q 블록을 담당하여 모든 K/V 블록을 순회하면서 부분 출력을 축적한다. 워프 수준에서 Q를 4개 워프에 분할하되 K/V는 모든 워프가 공유하여, softmax 통계(최대값, 합)의 워프 간 동기화를 register-level warp shuffle로 처리한다. 인과적 마스킹에서는 유효한 블록만 계산하도록 조기 종료를 적용하여 불필요한 계산을 제거한다. 시퀀스 길이가 길 때는 Q를 추가로 분할하여 더 많은 스레드 블록을 생성하고, 최종 결과를 별도 커널에서 병합한다.",
    results: "A100 80GB GPU에서 FlashAttention-2는 비인과적 어텐션에서 최대 230 TFLOPS를 달성하여 이론적 최대(312 TFLOPS)의 약 73%에 도달했다. 이는 FlashAttention v1 대비 약 2배의 속도 향상이다. 인과적 어텐션에서도 유사한 비율의 향상을 보였으며, 특히 긴 시퀀스(4K, 8K, 16K)에서 더 큰 속도 이득을 달성했다. GPT 스타일 모델의 end-to-end 학습에서 FlashAttention v1 대비 1.3-1.5배, 표준 어텐션 대비 5-8배의 학습 속도 향상을 기록했다.",
    impact: "FlashAttention-2는 현재 대부분의 LLM 학습 및 추론 프레임워크의 기본 어텐션 구현으로 사용되고 있다. PyTorch의 SDPA(Scaled Dot-Product Attention), HuggingFace Transformers, DeepSpeed, Megatron-LM 등에 통합되어 사실상 산업 표준이 되었다. GPU 하드웨어에 대한 깊은 이해를 바탕으로 한 시스템 수준 최적화가 알고리즘 수준 개선 못지않게 중요함을 재확인시켰다. 이후 FlashAttention-3(Hopper 아키텍처 최적화), FlashDecoding(추론 특화) 등으로 지속적으로 발전하고 있다.",
    relatedFoundations: ["transformer", "attention"],
    relatedPapers: [
      { id: "flash-attention", fieldId: "efficient", title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", relation: "prior" },
      { id: "vllm", fieldId: "efficient", title: "Efficient Memory Management for Large Language Model Serving with PagedAttention", relation: "related" },
    ],
  },

  // ============================================================
  // Graph Field (graph) - 7 Papers
  // ============================================================

  "mpnn": {
    tldr: "기존의 다양한 그래프 신경망(GCN, Gated Graph Neural Network, Interaction Networks 등)을 메시지 전달(message passing)이라는 통합된 프레임워크로 정리하고, 이를 양자 화학의 분자 성질 예측에 적용하여 최고 성능을 달성했다.",
    background: "분자와 같은 그래프 구조 데이터에 대한 딥러닝 방법이 다양하게 제안되었으나(Convolutional Networks on Graphs, Gated Graph Neural Networks, Deep Tensor Neural Networks 등), 이들 사이의 관계와 공통 구조가 명확하지 않았다. 양자 화학에서 분자의 성질(에너지, 쌍극자 모멘트, 진동 주파수 등)을 예측하는 것은 약물 발견과 재료 과학에서 핵심적인 과제였으며, DFT(밀도 함수 이론) 계산은 정확하지만 매우 느려 기계 학습 기반 대안이 필요했다.",
    keyIdea: "MPNN(Message Passing Neural Network)은 그래프 신경망을 메시지 전달(message passing)과 리드아웃(readout)이라는 두 단계로 추상화하는 통합 프레임워크를 제안한다. 메시지 전달 단계에서 각 노드는 이웃 노드로부터 메시지를 수집하고, 이를 자신의 은닉 상태와 결합하여 업데이트한다. 구체적으로 메시지 함수 M_t, 업데이트 함수 U_t, 리드아웃 함수 R의 세 가지 학습 가능한 함수로 구성된다. 기존의 GCN, GGNN, DTNN 등이 모두 이 프레임워크의 특수한 경우임을 보이고, 새로운 변형(edge network, set2set readout 등)을 제안하여 QM9 데이터셋에서 11개 화학 성질 중 11개에서 DFT 정확도에 근접하는 성능을 달성한다.",
    method: "T회의 메시지 전달 라운드에서 각 노드 v의 은닉 상태 h_v를 업데이트한다: m_v^{t+1} = sum_{w in N(v)} M_t(h_v^t, h_w^t, e_{vw}), h_v^{t+1} = U_t(h_v^t, m_v^{t+1}). 여기서 M_t는 메시지 함수, U_t는 업데이트 함수, e_{vw}는 엣지 특징이다. 그래프 수준 출력은 리드아웃 함수 R로 계산한다. 저자들은 edge network(엣지 특징을 신경망으로 처리), virtual graph elements(가상 노드/엣지), set2set readout(LSTM 기반 집합 인코딩) 등 여러 변형을 제안한다. QM9 분자 데이터셋(134K 분자, 13개 성질)에서 평가했다.",
    results: "제안된 MPNN 변형(enn-s2s)은 QM9 데이터셋의 13개 분자 성질 중 11개에서 기존 최고 성능을 달성했으며, 여러 성질에서 화학적 정확도(chemical accuracy)에 도달했다. 기존 방법 대비 평균 50% 이상의 오차 감소를 보였다. 특히 edge network와 set2set readout의 조합이 가장 효과적이었으며, 가상 그래프 요소의 도입도 성능 향상에 기여했다.",
    impact: "MPNN은 그래프 신경망 분야의 이론적 토대를 마련한 기념비적 논문이다. 메시지 전달이라는 통합 프레임워크는 이후 거의 모든 GNN 연구에서 참조되는 표준 용어가 되었으며, GIN의 표현력 분석, GraphSAGE의 인덕티브 학습, GAT의 어텐션 메커니즘 등이 모두 MPNN 프레임워크 내에서 이해될 수 있다. 양자 화학과 분자 과학에서 GNN 적용의 선구적 연구로서 SchNet, DimeNet, PaiNN 등 후속 분자 GNN의 직접적 기반이 되었다.",
    relatedFoundations: ["backpropagation"],
    relatedPapers: [
      { id: "schnet", fieldId: "graph", title: "SchNet: A continuous-filter convolutional neural network for modeling quantum interactions", relation: "related" },
      { id: "gin", fieldId: "graph", title: "How Powerful are Graph Neural Networks?", relation: "successor" },
      { id: "egnn", fieldId: "graph", title: "E(n) Equivariant Graph Neural Networks", relation: "successor" },
    ],
  },

  "schnet": {
    tldr: "원자 간 거리에 따라 연속적으로 변하는 필터를 학습하는 연속 필터 합성곱(continuous-filter convolution)을 도입하여, 불규칙한 3D 공간의 분자 구조에서 양자 화학적 성질을 높은 정확도로 예측하는 SchNet을 제안했다.",
    background: "분자의 양자 역학적 성질(에너지, 힘, 쌍극자 모멘트 등)을 예측하는 것은 약물 설계와 재료 과학의 핵심 과제이다. 기존의 기계 학습 접근법은 수작업으로 설계된 분자 기술자(descriptor)에 의존하거나, 고정된 그리드 기반 합성곱을 사용했는데, 분자는 불규칙한 3D 공간에 원자가 분포하므로 표준 합성곱을 직접 적용할 수 없었다. MPNN이 그래프 기반 접근을 제안했지만, 원자 간 거리의 연속적 특성을 충분히 활용하지 못했다.",
    keyIdea: "SchNet의 핵심 혁신은 연속 필터 합성곱(continuous-filter convolution)이다. 기존 CNN의 합성곱 필터가 고정된 격자점에 정의되는 것과 달리, SchNet의 필터는 원자 간 상대 위치(거리)의 연속 함수로 생성된다. 구체적으로, 원자 간 거리를 방사 기저 함수(radial basis function, RBF)로 확장한 뒤, 신경망(filter-generating network)으로 필터 가중치를 생성한다. 이를 통해 임의의 원자 배치에 대해 적응적인 상호작용을 모델링할 수 있으며, 회전 및 이동 불변성(invariance)을 자연스럽게 보장한다. 상호작용 블록(interaction block)을 여러 겹 쌓아 다체(many-body) 상호작용을 포착한다.",
    method: "각 원자의 초기 임베딩은 원소 종류에 따른 학습 가능한 벡터로 시작한다. 상호작용 블록에서 (1) 원자 간 거리를 가우시안 RBF로 확장하고, (2) dense 레이어로 필터 가중치를 생성하며, (3) element-wise 곱과 합으로 연속 필터 합성곱을 수행한다. 출력은 원자별 에너지 기여를 합산하거나 직접 분자 성질을 예측한다. 에너지 보존 SchNet 변형에서는 에너지의 좌표에 대한 해석적 미분으로 원자별 힘을 계산하여 에너지-힘 일관성을 보장한다. cutoff 함수로 원거리 상호작용을 부드럽게 제거한다.",
    results: "QM9 데이터셋에서 SchNet은 13개 분자 성질 중 대부분에서 화학적 정확도를 달성하여 기존 DTNN 및 MPNN과 경쟁력 있는 성능을 보였다. MD17 분자 동역학 벤치마크에서 에너지와 힘의 동시 예측에서 최고 성능을 기록했다. 에너지 보존 변형은 물리 법칙과 일관된 예측을 보장하여 분자 동역학 시뮬레이션에 직접 사용 가능한 수준의 정확도를 달성했다. 학습 데이터 효율성도 우수하여 소규모 데이터셋에서도 효과적이었다.",
    impact: "SchNet은 3D 분자 그래프에서의 등방성 메시지 전달(invariant message passing)의 표준 방법론을 확립했다. 연속 필터 합성곱이라는 아이디어는 이후 DimeNet(각도 정보 추가), PaiNN(등변 메시지 전달), MACE(다체 상호작용) 등 거의 모든 분자 GNN의 기초가 되었다. SchNetPack이라는 오픈소스 라이브러리로 제공되어 원자론적 시뮬레이션 커뮤니티에서 널리 사용되고 있으며, AlphaFold의 구조 모듈에도 유사한 거리 기반 상호작용 설계가 반영되었다.",
    relatedFoundations: ["backpropagation"],
    relatedPapers: [
      { id: "mpnn", fieldId: "graph", title: "Neural Message Passing for Quantum Chemistry", relation: "prior" },
      { id: "egnn", fieldId: "graph", title: "E(n) Equivariant Graph Neural Networks", relation: "successor" },
    ],
  },

  "gin": {
    tldr: "그래프 신경망(GNN)의 표현력을 이론적으로 분석하여, 대부분의 GNN이 1차 Weisfeiler-Leman(WL) 그래프 동형 테스트만큼 강력하지 않음을 밝히고, WL 테스트와 동등한 표현력을 가진 GIN(Graph Isomorphism Network)을 설계했다.",
    background: "GCN, GraphSAGE, GAT 등 다양한 GNN 아키텍처가 제안되었지만, 이들의 표현력(어떤 그래프 구조를 구분할 수 있는지)에 대한 이론적 이해가 부족했다. 실험적으로는 많은 태스크에서 좋은 성능을 보였지만, 어떤 GNN이 왜 더 나은지, 어떤 구조적 정보를 포착할 수 있는지에 대한 원칙적 분석이 없었다. 그래프 이론에서 WL 테스트는 그래프 동형성을 효율적으로 판별하는 고전적 알고리즘으로, GNN과의 관계가 주목되었다.",
    keyIdea: "이 논문의 핵심 기여는 GNN의 표현력을 WL 테스트와의 관계를 통해 엄밀하게 분석한 것이다. 주요 이론적 결과는 다음과 같다: (1) 임의의 MPNN 기반 GNN은 1-WL 테스트보다 강력할 수 없다(상한). (2) 이웃 집합의 집계 함수가 단사(injective)이면 GNN은 1-WL 테스트와 동등한 구별력을 가진다. (3) GCN의 mean 집계와 GraphSAGE의 max 집계는 단사가 아니므로 특정 멀티셋을 구분하지 못하여 표현력이 제한된다. 이 분석을 바탕으로 MLP + sum 집계로 구성된 GIN을 설계하여, 이론적으로 가장 강력한 MPNN임을 증명한다.",
    method: "GIN의 업데이트 규칙은 h_v^{(k)} = MLP^{(k)}((1 + epsilon^{(k)}) * h_v^{(k-1)} + sum_{u in N(v)} h_u^{(k-1)})이다. 여기서 epsilon은 학습 가능한 파라미터이고, sum 집계는 멀티셋에 대한 단사 함수를 구현한다. MLP는 범용 근사 정리(universal approximation theorem)에 의해 임의의 연속 함수를 근사할 수 있으므로, 전체 업데이트가 단사 함수가 된다. 그래프 수준 표현은 모든 레이어의 노드 표현을 합산(sum)한 뒤 연결(concatenation)하여 구성한다. mean 집계를 사용하는 GIN-mean, max 집계를 사용하는 GIN-max 변형도 비교 실험을 위해 구현했다.",
    results: "9개의 그래프 분류 벤치마크(MUTAG, PTC, PROTEINS, COLLAB, IMDB-BINARY 등)에서 GIN은 기존 GNN(GCN, GraphSAGE, GAT) 및 WL 서브트리 커널과 비교하여 대부분의 데이터셋에서 최고 또는 동등한 성능을 달성했다. 특히 GIN-sum이 GIN-mean과 GIN-max를 일관되게 능가하여 이론적 분석을 실험적으로 뒷받침했다. 합성 데이터에서 GIN이 GCN/GraphSAGE가 구분하지 못하는 그래프 쌍을 올바르게 구분함을 확인했다.",
    impact: "GIN은 GNN의 표현력에 대한 이론적 기초를 확립하여, 이후 GNN 연구의 방향에 지대한 영향을 미쳤다. WL 테스트와의 연결은 더 강력한 GNN(k-WL, k>1에 대응) 설계를 위한 연구(3WLGNN, k-IGN 등)를 촉발했으며, 집계 함수의 설계 원칙(sum이 mean/max보다 표현력이 높음)은 이후 GNN 아키텍처 설계의 기본 지침이 되었다. OGB 벤치마크에서 GIN은 기본 기준선(baseline)으로 널리 사용되고 있으며, GNN 이론 연구의 출발점으로서 수천 회 인용되었다.",
    relatedFoundations: ["backpropagation"],
    relatedPapers: [
      { id: "mpnn", fieldId: "graph", title: "Neural Message Passing for Quantum Chemistry", relation: "prior" },
      { id: "ogb", fieldId: "graph", title: "Open Graph Benchmark: Datasets for Machine Learning on Graphs", relation: "successor" },
      { id: "graphgps", fieldId: "graph", title: "Recipe for a General, Powerful, Scalable Graph Transformer", relation: "successor" },
    ],
  },

  "ogb": {
    tldr: "기존 그래프 ML 벤치마크의 한계(작은 규모, 비현실적 분할, 제한된 태스크 다양성)를 극복하기 위해 다양한 도메인과 규모를 아우르는 OGB(Open Graph Benchmark)를 제안하고, 통일된 평가 프로토콜과 리더보드를 제공했다.",
    background: "그래프 머신러닝이 빠르게 발전했지만, 표준 벤치마크의 부재로 인해 공정한 비교와 실질적 진보의 측정이 어려웠다. 기존에 널리 사용되던 벤치마크(Cora, Citeseer, PPI, TU 데이터셋 등)는 여러 문제가 있었다: (1) 너무 작은 규모(수천 노드)로 실제 응용과 동떨어짐, (2) 랜덤 분할이 비현실적인 평가를 유도, (3) 실험 프로토콜의 비표준화로 재현성이 떨어짐, (4) 분자, 소셜 네트워크, 지식 그래프 등 다양한 도메인을 포괄하지 못함.",
    keyIdea: "OGB는 세 가지 핵심 원칙으로 설계되었다. 첫째, 실제 응용에서 수집된 다양한 도메인(생물학, 화학, 소셜 네트워크, 지식 그래프)의 대규모 그래프를 포함하여 현실적 도전을 제시한다. 둘째, 각 데이터셋에 태스크 특성에 맞는 의미 있는 데이터 분할(시간 기반, 종 기반, 구조 기반 등)을 제공하여, 모델의 진정한 일반화 능력을 평가한다. 셋째, 데이터 로딩, 전처리, 평가를 자동화하는 Python 패키지와 공개 리더보드를 제공하여 재현 가능하고 공정한 비교를 가능하게 한다. 데이터셋은 노드 수준(ogbn), 링크 수준(ogbl), 그래프 수준(ogbg) 태스크로 분류된다.",
    method: "OGB는 세 가지 태스크 카테고리를 제공한다. ogbn(노드 분류): ogbn-products(240만 노드, Amazon 상품 네트워크), ogbn-proteins(13만 노드, 단백질 상호작용), ogbn-arxiv(17만 노드, 논문 인용) 등. ogbl(링크 예측): ogbl-ppa(57만 노드, 단백질 연관), ogbl-collab(23만 노드, 학술 협업), ogbl-citation2(295만 노드, 논문 인용) 등. ogbg(그래프 분류/회귀): ogbg-molhiv(4만 그래프, HIV 활성 예측), ogbg-molpcba(43만 그래프, 생물 활성), ogbg-ppa(16만 그래프, 종 간 단백질 기능) 등. 각 데이터셋에 도메인 전문가와 협력하여 설계한 의미 있는 데이터 분할과 적절한 평가 메트릭을 제공한다.",
    results: "GCN, GraphSAGE, GIN 등 기존 GNN을 OGB에서 체계적으로 평가한 결과, 기존의 작은 벤치마크에서의 순위가 대규모 현실적 설정에서 변동됨을 확인했다. 예를 들어, ogbn-products에서 단순한 MLP도 GNN과 비교 가능한 성능을 보여 GNN의 우위가 자명하지 않음을 밝혔다. ogbg-molhiv에서는 랜덤 분할 대비 scaffold 분할에서 모든 모델의 성능이 크게 하락하여, 현실적 분할의 중요성을 입증했다. 리더보드를 통해 새로운 방법들의 점진적 진보가 투명하게 추적되었다.",
    impact: "OGB는 그래프 ML 커뮤니티의 표준 벤치마크로 자리잡아, NeurIPS, ICML, ICLR 등 주요 학회의 GNN 논문 대다수가 OGB 결과를 보고하게 되었다. 공정하고 재현 가능한 평가를 통해 분야의 건전한 발전을 촉진했으며, OGB-LSC(대규모 챌린지)로 확장되어 KDD Cup 등에서 경쟁 벤치마크로 활용되고 있다. 그래프 ML에서 '벤치마크 주도 연구'의 모범 사례로, 이후 Long Range Graph Benchmark, TGB(Temporal Graph Benchmark) 등 유사한 벤치마크 구축에 영감을 주었다.",
    relatedFoundations: ["backpropagation"],
    relatedPapers: [
      { id: "gin", fieldId: "graph", title: "How Powerful are Graph Neural Networks?", relation: "prior" },
      { id: "graphgps", fieldId: "graph", title: "Recipe for a General, Powerful, Scalable Graph Transformer", relation: "related" },
      { id: "graphmae", fieldId: "graph", title: "GraphMAE: Self-Supervised Masked Graph Autoencoders", relation: "related" },
    ],
  },

  "egnn": {
    tldr: "구면 조화 함수(spherical harmonics)나 고차 텐서 없이도 좌표 업데이트를 통해 E(n)(유클리드 이동, 회전, 반사) 등변성을 달성하는 단순하면서도 효과적인 EGNN을 제안하여, 분자 동역학 및 N-body 시뮬레이션에서 우수한 성능을 보였다.",
    background: "물리 시스템과 분자 구조는 공간의 이동, 회전, 반사에 대해 대칭적이다. 즉, 분자를 회전시켜도 그 성질은 변하지 않으며(불변), 힘과 같은 벡터량은 함께 회전해야 한다(등변). TFN(Tensor Field Networks), SE(3)-Transformers 등 기존 등변 GNN은 구면 조화 함수와 Clebsch-Gordan 계수를 사용하여 등변성을 보장했으나, 구현이 복잡하고 계산 비용이 높았다. 불변 모델(SchNet 등)은 거리 정보만 사용하므로 방향 정보를 충분히 활용하지 못했다.",
    keyIdea: "EGNN의 핵심 혁신은 노드 좌표를 명시적으로 업데이트하면서도 E(n) 등변성을 유지하는 간단한 메커니즘을 제안한 것이다. 기존 등변 GNN이 고차 텐서(구면 조화)를 사용한 것과 달리, EGNN은 두 노드 간의 상대 위치 벡터(x_i - x_j)에 스칼라 가중치를 곱하여 좌표를 업데이트한다. 상대 위치 벡터 자체가 이미 등변적이므로, 스칼라 가중치만 불변이면 전체 업데이트가 등변적이 된다. 이 스칼라 가중치는 두 노드의 특징과 거리에 의존하는 학습 가능한 함수로 계산된다. 이를 통해 구면 조화, Wigner-D 행렬 등의 복잡한 수학적 장치 없이도 등변성을 달성한다.",
    method: "각 레이어에서 세 가지 업데이트를 수행한다. (1) 메시지 계산: m_{ij} = phi_e(h_i, h_j, ||x_i - x_j||^2, a_{ij}), 여기서 a_{ij}는 엣지 속성. (2) 좌표 업데이트: x_i = x_i + C * sum_j (x_i - x_j) * phi_x(m_{ij}), 여기서 phi_x는 스칼라 가중치를 출력하는 MLP. (3) 노드 특징 업데이트: h_i = h_i + phi_h(h_i, sum_j m_{ij}). phi_e, phi_x, phi_h는 모두 간단한 MLP이다. 좌표 업데이트가 상대 위치 벡터에 스칼라를 곱하는 형태이므로, 회전/이동에 대한 등변성이 자동으로 보장된다. 반사 등변성도 마찬가지이다.",
    results: "N-body 시뮬레이션(하전 입자 궤적 예측)에서 EGNN은 SE(3)-Transformer, TFN, Radial Field 등 기존 등변 모델을 MSE 기준으로 크게 능가했다. QM9 분자 성질 예측에서 SchNet, DimeNet 등 불변 모델과 경쟁력 있는 성능을 보이면서 계산 비용은 크게 낮았다. 모델 그래프 오토인코더(graph autoencoder)에서 분자 3D 구조 생성에도 적용하여, 등변 생성 모델의 가능성을 시연했다. TFN/SE(3)-Transformer 대비 학습 속도가 수배 빨랐다.",
    impact: "EGNN은 등변 GNN의 민주화에 기여했다. 구면 조화 없이도 등변성을 달성할 수 있다는 것을 보여줌으로써, 등변 모델의 구현 및 활용 장벽을 크게 낮추었다. 이후 GVP(Geometric Vector Perceptrons), SEGNN, PaiNN 등 다양한 간소화된 등변 아키텍처의 발전에 직접적 영감을 주었다. 분자 생성(EDM, GeoDiff), 단백질 구조 예측, 점군(point cloud) 처리 등 다양한 3D 과학 응용에서 기본 구성 요소로 채택되고 있으며, DiffDock 등 분자 도킹 모델의 기초가 되었다.",
    relatedFoundations: ["backpropagation"],
    relatedPapers: [
      { id: "mpnn", fieldId: "graph", title: "Neural Message Passing for Quantum Chemistry", relation: "prior" },
      { id: "schnet", fieldId: "graph", title: "SchNet: A continuous-filter convolutional neural network for modeling quantum interactions", relation: "prior" },
      { id: "graphgps", fieldId: "graph", title: "Recipe for a General, Powerful, Scalable Graph Transformer", relation: "related" },
    ],
  },

  "graphgps": {
    tldr: "위치/구조 인코딩(PE/SE), 로컬 MPNN 어텐션, 글로벌 Transformer 어텐션을 하나의 프레임워크로 결합하는 GraphGPS를 제안하여, 그래프 Transformer의 설계 공간을 체계적으로 정리하고 다양한 벤치마크에서 최고 수준의 성능을 달성했다.",
    background: "Transformer가 NLP와 비전에서 큰 성공을 거둔 후, 그래프 데이터에 Transformer를 적용하려는 시도가 활발해졌다. 그러나 그래프 Transformer에는 고유한 도전이 있었다: (1) 그래프에는 자연스러운 위치 개념이 없어 위치 인코딩 설계가 어렵고, (2) 모든 노드 쌍에 대한 전역 어텐션은 O(N^2) 비용이 들어 대규모 그래프에 비효율적이며, (3) 로컬 이웃 정보를 명시적으로 활용하는 MPNN의 귀납적 편향(inductive bias)을 어떻게 통합할지 불명확했다. SAN, Graphormer 등 초기 그래프 Transformer가 제안되었지만, 통일된 설계 원칙이 부재했다.",
    keyIdea: "GraphGPS(General, Powerful, Scalable Graph Transformer)는 그래프 Transformer를 세 가지 모듈의 조합으로 구조화하는 레시피를 제안한다. (1) 위치/구조 인코딩(PE/SE): 랜덤 워크 기반(RWSE), 라플라시안 고유벡터(LapPE), 또는 학습 가능한 구조 인코딩을 노드/엣지 특징에 추가하여 위치 정보를 제공한다. (2) 로컬 메시지 전달: GCN, GIN, GINE 등 기존 MPNN으로 로컬 이웃 정보를 처리한다. (3) 글로벌 어텐션: Transformer 셀프 어텐션으로 장거리 의존성을 포착한다. 각 레이어에서 로컬 MPNN과 글로벌 Transformer의 출력을 결합하여, 두 접근의 장점을 동시에 취한다.",
    method: "각 GPS 레이어는 다음과 같이 구성된다: (1) 노드 특징에 PE/SE를 추가, (2) MPNN 서브레이어(GCN, GIN, GINE, PNA 등 선택 가능)로 로컬 이웃 정보 처리, (3) 전역 Transformer 셀프 어텐션(또는 효율적 변형인 Performer, BigBird 등), (4) 두 출력의 합산 또는 연결, (5) FFN + 잔차 연결 + 정규화. 대규모 그래프에서는 전역 어텐션을 선형 복잡도 변형으로 대체할 수 있다. 위치 인코딩으로는 SignNet을 사용한 고유벡터 기반 LapPE와 RWSE를 권장한다. 하이퍼파라미터 탐색으로 각 데이터셋에 최적의 조합을 찾는다.",
    results: "GraphGPS는 ZINC(분자 성질 예측), PATTERN/CLUSTER(패턴 인식), MolPCBA/MolHIV(분자 활성), PCQM4Mv2(양자 화학) 등 11개 벤치마크에서 평가되었다. ZINC에서 MAE 0.070을 달성하여 기존 GNN 및 그래프 Transformer를 크게 능가했다. PCQM4Mv2에서도 GCN+Transformer 조합이 단독 MPNN이나 단독 Transformer보다 우수한 성능을 보였다. 특히 로컬 MPNN과 글로벌 어텐션의 결합이 단독 사용보다 일관되게 좋은 결과를 보여, 두 접근의 상호 보완성을 입증했다.",
    impact: "GraphGPS는 그래프 Transformer 설계의 체계적 가이드라인을 제공하여, 이후 연구들이 비교하고 발전시킬 수 있는 통일된 프레임워크를 확립했다. 로컬+글로벌 어텐션의 결합, PE/SE의 중요성 등의 발견은 이후 Exphormer, GRIT, TokenGT 등 후속 그래프 Transformer 설계에 직접 반영되었다. GraphGPS 프레임워크의 오픈소스 구현은 그래프 Transformer 연구의 재현성과 접근성을 크게 높였으며, GNN과 Transformer의 융합이라는 연구 방향의 표준 참조점이 되었다.",
    relatedFoundations: ["transformer", "attention"],
    relatedPapers: [
      { id: "gin", fieldId: "graph", title: "How Powerful are Graph Neural Networks?", relation: "prior" },
      { id: "mpnn", fieldId: "graph", title: "Neural Message Passing for Quantum Chemistry", relation: "prior" },
      { id: "graphmae", fieldId: "graph", title: "GraphMAE: Self-Supervised Masked Graph Autoencoders", relation: "related" },
    ],
  },

  "graphmae": {
    tldr: "그래프 데이터에 마스크 오토인코딩(masked autoencoding)을 적용하여, 대조 학습(contrastive learning) 없이도 그래프 자기지도학습에서 경쟁력 있는 성능을 달성하는 GraphMAE를 제안했다. 스케일드 코사인 오차와 리마스킹 전략으로 기존 생성적 방법의 한계를 극복했다.",
    background: "그래프 자기지도학습은 대부분 대조 학습(GraphCL, GRACE, GCA 등)에 의존해왔다. 대조 학습은 양성/음성 쌍 구성을 위한 데이터 증강 전략 설계가 태스크/도메인에 민감하며, 대규모 배치나 메모리 뱅크가 필요하다는 실용적 한계가 있었다. 반면 NLP(BERT)와 비전(MAE, BEiT)에서 마스크 기반 생성적 사전학습이 큰 성공을 거두었으나, 그래프에서의 마스크 오토인코딩은 노드 특징의 저차원성, 불규칙한 구조, 적절한 재구성 손실 선택의 어려움 등으로 성능이 제한적이었다.",
    keyIdea: "GraphMAE는 그래프에서 마스크 오토인코딩이 효과적으로 작동하기 위한 세 가지 핵심 설계를 제안한다. 첫째, 스케일드 코사인 오차(scaled cosine error, SCE)를 재구성 손실로 사용한다. MSE는 노드 특징의 분산에 민감하고, 크로스 엔트로피는 이산화가 필요한데, SCE는 특징 벡터의 방향 유사도를 측정하여 이러한 문제를 회피한다. 둘째, 디코더에 단일 GNN 레이어와 리마스킹(re-masking) 전략을 적용한다. 인코더 출력에서 마스크된 노드의 표현을 다시 마스크 토큰으로 교체한 뒤 디코더에 입력함으로써, 디코더가 인코더의 출력을 그대로 복사하는 지름길(shortcut)을 방지한다. 셋째, 균일 랜덤 마스킹보다 높은 마스킹 비율(50-75%)이 더 도전적인 사전학습 목표를 제공하여 학습 효과를 높인다.",
    method: "GNN 인코더(GCN, GAT, GIN 등)가 마스크되지 않은 노드의 특징을 입력받아 전체 그래프에 대한 노드 표현을 생성한다. 마스크된 노드는 학습 가능한 마스크 토큰으로 대체된다. 인코더 출력에서 마스크된 노드 위치의 표현을 다시 마스크 토큰으로 교체한 뒤(리마스킹), 단일 GNN 레이어 디코더를 통해 원래 노드 특징을 재구성한다. 손실은 마스크된 노드에서만 계산: L = 1/|M| sum_{v in M} (1 - cos(x_v, x_v_hat)) / gamma, 여기서 gamma는 스케일링 팩터. 사전학습 후 인코더의 노드 표현을 다운스트림 태스크에 활용한다.",
    results: "21개의 그래프/노드 분류 벤치마크에서 GraphMAE를 평가한 결과, 대조 학습 기반 방법(GraphCL, GRACE, GCA, BGRL 등)과 동등하거나 우수한 성능을 달성했다. Cora, Citeseer, PubMed 노드 분류에서 기존 최고 자기지도학습 방법을 능가했으며, 분자 분류(MUTAG, PROTEINS, NCI1 등)에서도 경쟁력 있는 결과를 보였다. 특히 데이터 증강 전략 설계 없이 범용적으로 적용 가능하다는 실용적 장점을 입증했다. 소거 실험(ablation)에서 SCE 손실과 리마스킹 전략 각각의 기여를 확인했다.",
    impact: "GraphMAE는 그래프 자기지도학습에서 생성적 방법의 부활을 이끌었다. 대조 학습의 복잡한 증강 전략 없이도 강력한 성능을 달성할 수 있음을 보여, 이후 GraphMAE2, MaskGAE, S2GAE 등 마스크 기반 그래프 사전학습 연구의 활발한 발전을 촉발했다. MAE 패러다임이 NLP, 비전에 이어 그래프 도메인에서도 유효함을 입증한 연구로서, 도메인 간 방법론 전이의 성공적 사례이다. KDD 2022에서 발표되어 그래프 ML 커뮤니티에서 높은 관심을 받았으며, OGB 벤치마크에서의 사전학습 방법으로도 활용되고 있다.",
    relatedFoundations: ["bert", "transformer"],
    relatedPapers: [
      { id: "gin", fieldId: "graph", title: "How Powerful are Graph Neural Networks?", relation: "prior" },
      { id: "ogb", fieldId: "graph", title: "Open Graph Benchmark: Datasets for Machine Learning on Graphs", relation: "related" },
      { id: "graphgps", fieldId: "graph", title: "Recipe for a General, Powerful, Scalable Graph Transformer", relation: "related" },
    ],
  },
};
