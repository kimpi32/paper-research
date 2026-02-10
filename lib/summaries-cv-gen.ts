import type { PaperSummary } from "./paper-summaries";

export const cvGenSummaries: Record<string, PaperSummary> = {
  // ============================================================
  // CV Field (cv)
  // ============================================================

  "yolo": {
    tldr: "이미지를 그리드로 나눠 바운딩 박스와 클래스 확률을 한 번의 순전파로 동시에 예측하는 실시간 객체 검출 모델이다.",
    background: "기존 객체 검출 방법(R-CNN 계열)은 영역 제안(region proposal) 생성과 분류를 별도로 수행하는 2단계 파이프라인을 사용했다. 이로 인해 추론 속도가 느려 실시간 적용이 어려웠으며, 각 구성 요소를 독립적으로 최적화해야 하는 문제가 있었다. 통합된 단일 네트워크로 객체 검출을 수행하려는 시도가 필요했다.",
    keyIdea: "YOLO는 객체 검출을 단일 회귀 문제로 재정의한다. 입력 이미지를 S x S 그리드로 분할하고, 각 셀이 B개의 바운딩 박스와 해당 신뢰도, 그리고 C개 클래스의 조건부 확률을 동시에 예측한다. 전체 이미지를 한 번만 보고(You Only Look Once) 모든 객체를 검출하므로, 기존 방법 대비 수십 배 빠른 추론 속도를 달성한다. 또한 이미지 전체의 맥락 정보를 활용하기 때문에 배경을 객체로 잘못 검출하는 비율(false positive)이 낮다.",
    method: "448x448 입력 이미지를 7x7 그리드로 나누고, 각 셀에서 2개의 바운딩 박스(x, y, w, h, confidence)와 20개 클래스 확률을 예측한다. 24개 합성곱 레이어와 2개 완전연결 레이어로 구성되며, ImageNet 사전학습 후 검출 태스크로 파인튜닝한다. 학습 시에는 좌표, 객체 유무, 클래스 분류에 대한 다중 항 손실 함수를 사용한다.",
    results: "Pascal VOC 2007에서 63.4 mAP를 달성하면서 45 FPS의 실시간 속도를 보였다. Fast YOLO 변형은 155 FPS까지 가능했다. 당시 최고 정확도 모델(Faster R-CNN 등)보다 정확도는 다소 낮았지만, 속도 면에서 압도적 우위를 보였다.",
    impact: "실시간 객체 검출의 가능성을 입증하며 자율주행, 감시 시스템, 로봇 비전 등 실시간 응용 분야에 큰 영향을 미쳤다. YOLO 시리즈(v2~v8)로 이어지며 지속적으로 발전했고, 단일 단계(single-stage) 검출기라는 새로운 패러다임을 확립했다. 산업계에서 가장 널리 사용되는 객체 검출 프레임워크 중 하나가 되었다.",
    relatedFoundations: ["alexnet", "resnet"],
    relatedPapers: [
      { id: "mask-rcnn", fieldId: "cv", title: "Mask R-CNN", relation: "successor" },
      { id: "detr", fieldId: "cv", title: "DETR", relation: "successor" },
      { id: "sam", fieldId: "cv", title: "Segment Anything", relation: "successor" },
    ],
  },

  "unet": {
    tldr: "인코더-디코더 구조에 스킵 연결을 결합하여 적은 학습 데이터로도 정밀한 의료 영상 분할을 가능하게 한 네트워크이다.",
    background: "의료 영상 분석에서 픽셀 수준의 정밀한 분할(segmentation)은 핵심 과제였지만, 레이블링된 데이터가 극히 부족한 것이 큰 제약이었다. 기존 슬라이딩 윈도우 방식은 느리고 중복 계산이 많았으며, 전체 이미지 수준의 분류 네트워크는 위치 정보를 잃어 세밀한 분할이 어려웠다.",
    keyIdea: "U-Net은 수축 경로(인코더)와 확장 경로(디코더)가 대칭적 U자 형태를 이루는 구조를 제안했다. 핵심 혁신은 인코더의 각 단계에서 추출한 고해상도 특징 맵을 디코더의 대응 단계에 직접 연결하는 스킵 연결(skip connection)이다. 이를 통해 디코더가 업샘플링 과정에서 잃어버리는 공간적 세부 정보를 복원할 수 있다. 또한 데이터 증강(탄성 변형, 회전 등)을 적극 활용하여 적은 수의 학습 이미지로도 우수한 성능을 달성한다. 경계 영역에서의 분할 정확도를 높이기 위해 가중치 맵을 사용한 손실 함수도 제안했다.",
    method: "인코더는 3x3 합성곱과 2x2 맥스 풀링을 반복하며 특징을 추출하고, 디코더는 2x2 업 컨볼루션으로 해상도를 복원한다. 각 디코더 단계에서 인코더의 대응 특징 맵을 잘라(crop) 이어 붙인다. 최종 1x1 합성곱으로 픽셀별 클래스를 예측한다.",
    results: "ISBI 2012 EM 분할 챌린지에서 기존 최고 성능을 큰 차이로 넘어섰으며, ISBI 2015 세포 추적 챌린지에서도 1위를 달성했다. 단 30장의 학습 이미지로도 뛰어난 분할 성능을 보여 데이터 효율성을 입증했다.",
    impact: "의료 영상 분할의 사실상 표준 아키텍처가 되었으며, 세포 분할, 장기 분할, 병변 검출 등 다양한 바이오메디컬 태스크에 광범위하게 적용되었다. U-Net의 인코더-디코더 + 스킵 연결 설계 패턴은 이후 분할 네트워크(V-Net, Attention U-Net, nnU-Net 등)의 기본 템플릿이 되었으며, 의료 분야를 넘어 위성 영상, 자율주행 등에서도 활용된다.",
    relatedFoundations: ["alexnet", "resnet"],
    relatedPapers: [
      { id: "mask-rcnn", fieldId: "cv", title: "Mask R-CNN", relation: "related" },
      { id: "sam", fieldId: "cv", title: "Segment Anything", relation: "successor" },
    ],
  },

  "mask-rcnn": {
    tldr: "Faster R-CNN에 픽셀 수준 마스크 예측 브랜치를 추가하고 RoIAlign을 도입하여 인스턴스 분할을 통합적으로 수행하는 프레임워크이다.",
    background: "객체 검출(바운딩 박스)과 시맨틱 분할(픽셀 분류)은 각각 발전해왔지만, 개별 객체 인스턴스를 픽셀 수준으로 구분하는 인스턴스 분할은 여전히 어려운 문제였다. Faster R-CNN은 강력한 2단계 검출기였지만 바운딩 박스만 출력했으며, RoI Pooling의 양자화 오류가 정밀한 공간 정보 보존을 방해했다.",
    keyIdea: "Mask R-CNN은 Faster R-CNN의 각 관심 영역(RoI)에 대해 바운딩 박스 회귀 및 분류와 병렬로 바이너리 마스크를 예측하는 브랜치를 추가한다. 핵심적으로 기존 RoI Pooling의 양자화 문제를 해결하는 RoIAlign을 제안했는데, 이는 이중선형 보간(bilinear interpolation)을 사용하여 입력 특징과 추출된 특징 사이의 정확한 공간 정렬을 보장한다. 마스크 예측은 클래스별로 독립적인 바이너리 마스크를 생성하여 클래스 간 경쟁을 없앴으며, 이를 통해 검출과 분할을 우아하게 통합했다.",
    method: "ResNet-FPN 백본으로 특징을 추출한 뒤, Region Proposal Network(RPN)이 후보 영역을 생성한다. 각 RoI에 대해 RoIAlign으로 고정 크기 특징을 추출하고, 분류/회귀 헤드와 마스크 헤드가 병렬로 작동한다. 마스크 헤드는 작은 FCN(Fully Convolutional Network)으로 28x28 해상도의 마스크를 예측한다.",
    results: "COCO 인스턴스 분할에서 35.7 AP를 달성하며 당시 최고 성능을 기록했다. 객체 검출에서도 Faster R-CNN을 능가했으며, 키포인트 검출 등 다른 태스크로의 확장도 효과적이었다. 추론 속도는 약 5 FPS로 실용적인 수준이었다.",
    impact: "인스턴스 분할의 표준 프레임워크로 자리잡았으며, Detectron/Detectron2 등 주요 검출 라이브러리의 핵심이 되었다. RoIAlign은 이후 거의 모든 영역 기반 검출기에 채택되었으며, Mask R-CNN의 모듈식 설계 철학은 이후 Panoptic FPN, PointRend 등 후속 연구에 큰 영향을 미쳤다.",
    relatedFoundations: ["resnet"],
    relatedPapers: [
      { id: "yolo", fieldId: "cv", title: "YOLO", relation: "prior" },
      { id: "detr", fieldId: "cv", title: "DETR", relation: "successor" },
      { id: "sam", fieldId: "cv", title: "Segment Anything", relation: "successor" },
    ],
  },

  "detr": {
    tldr: "트랜스포머를 객체 검출에 최초로 적용하여 앵커, NMS 등 수작업 구성 요소 없이 엔드투엔드로 검출을 수행하는 모델이다.",
    background: "기존 객체 검출기(Faster R-CNN, YOLO 등)는 앵커 박스 설계, NMS(Non-Maximum Suppression), 양성/음성 샘플 할당 규칙 등 사람이 설계한 복잡한 구성 요소에 크게 의존했다. 이러한 수작업 컴포넌트는 하이퍼파라미터 튜닝이 번거롭고 최적화가 어려웠다. 트랜스포머가 NLP에서 큰 성공을 거두면서 비전 태스크에도 적용하려는 움직임이 시작되고 있었다.",
    keyIdea: "DETR은 객체 검출을 집합 예측(set prediction) 문제로 재정의한다. CNN 백본으로 추출한 특징에 트랜스포머 인코더-디코더를 적용하고, 학습 가능한 객체 쿼리(object query)를 통해 고정된 수의 예측을 병렬로 생성한다. 헝가리안 알고리즘 기반의 이분 매칭(bipartite matching) 손실로 예측과 정답을 일대일 대응시켜, NMS 없이도 중복 없는 검출이 가능하다. 트랜스포머의 전역 어텐션(global attention)을 활용하여 이미지 전체의 맥락을 파악하므로, 큰 객체 검출에 특히 강점을 보인다.",
    method: "ResNet 백본에서 특징 맵을 추출한 후 1x1 합성곱으로 차원을 줄이고, 위치 인코딩을 더해 트랜스포머 인코더에 입력한다. 디코더는 N개의 학습 가능한 객체 쿼리를 입력받아 N개의 예측(클래스 + 바운딩 박스)을 출력한다. 학습 시 이분 매칭으로 최적 할당을 구한 뒤, 분류 손실과 박스 회귀 손실(L1 + GIoU)을 계산한다.",
    results: "COCO 데이터셋에서 Faster R-CNN과 동등한 42 AP를 달성했으며, 특히 큰 객체에서 우수한 성능을 보였다. 학습 수렴이 느린 편(500 에포크)이었으나, 앵커나 NMS 없이 간결한 파이프라인으로 경쟁력 있는 성능을 입증했다.",
    impact: "비전 분야에서 트랜스포머 기반 검출의 시대를 열었으며, Deformable DETR, DAB-DETR, DINO-DETR 등 수많은 후속 연구를 촉발했다. 앵커 프리 + NMS 프리 설계는 검출 파이프라인의 단순화라는 새로운 방향을 제시했다. 또한 집합 예측 프레임워크는 Panoptic Segmentation, 행동 검출 등 다른 태스크에도 확장되었다.",
    relatedFoundations: ["transformer", "resnet", "vit"],
    relatedPapers: [
      { id: "mask-rcnn", fieldId: "cv", title: "Mask R-CNN", relation: "prior" },
      { id: "yolo", fieldId: "cv", title: "YOLO", relation: "prior" },
      { id: "sam", fieldId: "cv", title: "Segment Anything", relation: "successor" },
    ],
  },

  "sam": {
    tldr: "포인트, 박스, 텍스트 등 다양한 프롬프트로 어떤 이미지의 어떤 객체든 분할할 수 있는 범용 분할 파운데이션 모델이다.",
    background: "NLP에서 GPT-3 등 파운데이션 모델이 제로샷으로 다양한 태스크를 수행하는 데 반해, 컴퓨터 비전의 분할(segmentation) 분야는 여전히 태스크별 모델에 의존하고 있었다. 또한 고품질 분할 데이터셋 구축에는 막대한 비용이 필요했고, 기존 데이터셋은 특정 도메인에 한정되어 범용성이 부족했다.",
    keyIdea: "SAM(Segment Anything Model)은 세 가지 혁신을 제시한다. 첫째, 프롬프트 기반 분할이라는 태스크를 정의하여 포인트, 바운딩 박스, 마스크, 텍스트 등 다양한 입력으로 분할을 유도한다. 둘째, 이미지 인코더(ViT-H), 프롬프트 인코더, 경량 마스크 디코더로 구성된 효율적 아키텍처를 설계했다. 셋째, 모델-인-더-루프 방식으로 SA-1B 데이터셋(11억 개 마스크, 1100만 이미지)을 구축하는 데이터 엔진을 개발했다. 이미지 인코더는 한 번만 실행하고, 프롬프트가 바뀔 때마다 경량 디코더만 재실행하여 실시간 상호작용이 가능하다.",
    method: "ViT-H(MAE 사전학습)를 이미지 인코더로 사용하여 이미지 임베딩을 추출한다. 프롬프트 인코더는 포인트/박스를 위치 인코딩으로, 마스크를 합성곱으로, 텍스트를 CLIP으로 인코딩한다. 마스크 디코더는 변형된 트랜스포머 블록 2개로 구성되며, 모호한 프롬프트에 대해 3개의 유효한 마스크를 동시에 출력한다.",
    results: "23개의 다양한 분할 데이터셋에서 제로샷 평가를 수행한 결과, 대부분의 벤치마크에서 완전 지도학습 모델과 경쟁하거나 능가하는 성능을 보였다. 특히 학습 데이터에 포함되지 않은 새로운 도메인(의료, 수중, 항공 등)에서도 강건한 분할 능력을 입증했다.",
    impact: "컴퓨터 비전 분할 분야의 GPT-3 모먼트로 평가받으며, 범용 분할 파운데이션 모델의 가능성을 입증했다. SA-1B는 역대 최대 규모의 분할 데이터셋이 되었다. 이미지 편집, 3D 재구성, 비디오 추적, 의료 영상 등 수많은 다운스트림 응용에서 핵심 구성 요소로 활용되고 있으며, SAM 2(비디오 확장) 등 후속 연구도 활발하다.",
    relatedFoundations: ["vit", "resnet"],
    relatedPapers: [
      { id: "mask-rcnn", fieldId: "cv", title: "Mask R-CNN", relation: "prior" },
      { id: "detr", fieldId: "cv", title: "DETR", relation: "prior" },
      { id: "unet", fieldId: "cv", title: "U-Net", relation: "prior" },
    ],
  },

  // ============================================================
  // Generative Field (generative)
  // ============================================================

  "wgan": {
    tldr: "GAN의 판별자를 크리틱(critic)으로 대체하고 바서슈타인 거리를 최적화하여 모드 붕괴와 학습 불안정성을 근본적으로 해결한 모델이다.",
    background: "기존 GAN은 생성자와 판별자 사이의 미니맥스 게임을 통해 학습하지만, Jensen-Shannon 발산 기반의 목적 함수는 여러 근본적 문제를 야기했다. 모드 붕괴(mode collapse)로 생성 다양성이 떨어지고, 판별자가 너무 잘 학습되면 생성자의 기울기가 소실되며, 학습 안정성과 생성 품질 사이의 균형을 맞추기 매우 어려웠다.",
    keyIdea: "WGAN은 JS 발산 대신 Earth Mover's Distance(바서슈타인-1 거리)를 목적 함수로 사용한다. 이 거리는 두 분포의 지지(support)가 겹치지 않아도 의미 있는 기울기를 제공하므로, 생성자가 안정적으로 학습할 수 있다. 판별자를 1-립시츠(1-Lipschitz) 함수로 제한하기 위해 가중치 클리핑을 적용하며, 이를 통해 크리틱이라 불리는 새로운 역할을 수행하게 된다. 크리틱 손실값이 생성 품질과 직접적으로 상관관계를 가져, 학습 진행 상황을 모니터링할 수 있게 된 것도 큰 장점이다.",
    method: "크리틱은 실제와 생성 데이터 간 바서슈타인 거리를 추정하도록 학습되며, 생성자는 이 거리를 줄이도록 학습된다. 크리틱의 가중치를 [-c, c] 범위로 클리핑하여 립시츠 제약을 근사한다. 크리틱을 생성자보다 더 많이(5:1 비율) 업데이트하여 거리 추정의 정확성을 보장한다. 옵티마이저는 RMSProp를 사용하며, 모멘텀 기반 옵티마이저는 불안정성을 유발할 수 있어 피한다.",
    results: "기존 GAN 대비 학습이 훨씬 안정적이며, 모드 붕괴 현상이 크게 줄었다. 크리틱 손실과 생성 품질 사이의 상관관계가 명확하여 하이퍼파라미터 탐색이 용이해졌다. 후속 연구인 WGAN-GP에서 가중치 클리핑을 그래디언트 페널티로 대체하여 성능을 더욱 개선했다.",
    impact: "GAN 학습의 이론적 이해를 크게 발전시키고, 최적 수송(Optimal Transport) 이론과 생성 모델링을 연결하는 중요한 가교 역할을 했다. 이후 거의 모든 GAN 학습에서 바서슈타인 거리 기반 목적 함수나 그 변형이 활용되었으며, WGAN-GP, Spectral Normalization 등 후속 안정화 기법의 기반이 되었다.",
    relatedFoundations: ["gan"],
    relatedPapers: [
      { id: "stylegan", fieldId: "generative", title: "StyleGAN", relation: "successor" },
    ],
  },

  "stylegan": {
    tldr: "매핑 네트워크와 적응적 인스턴스 정규화를 통해 스타일 기반으로 이미지를 계층적으로 제어 생성하는 GAN 아키텍처이다.",
    background: "기존 GAN 생성자는 잠재 벡터 z를 직접 입력으로 받아 이미지를 생성했지만, 잠재 공간의 얽힘(entanglement) 때문에 생성 이미지의 특정 속성만 독립적으로 제어하기 어려웠다. 고해상도 얼굴 생성의 품질은 꾸준히 향상되고 있었지만, 생성 과정에 대한 직관적 이해와 세밀한 제어가 부족했다.",
    keyIdea: "StyleGAN은 두 가지 핵심 혁신을 도입한다. 첫째, 8층 MLP로 구성된 매핑 네트워크가 잠재 벡터 z를 중간 잠재 공간 W로 변환하여 얽힘을 해소한다. 둘째, W 공간의 벡터를 적응적 인스턴스 정규화(AdaIN)를 통해 생성자의 각 레이어에 주입하여 서로 다른 해상도에서 서로 다른 시각적 속성(포즈, 얼굴형, 피부색, 머리카락 질감 등)을 제어한다. 또한 확률적 변동을 위한 노이즈 입력을 각 레이어에 추가하여 모공, 머리카락 배치 등 미세한 디테일의 다양성을 확보한다. 전통적 입력 레이어를 학습 가능한 상수로 대체한 것도 특징적이다.",
    method: "생성자의 합성 네트워크는 4x4부터 1024x1024까지 점진적으로 해상도를 키운다. 각 해상도의 합성곱 레이어마다 매핑 네트워크 출력으로부터 계산된 스케일/바이어스가 AdaIN으로 적용된다. 스타일 믹싱(style mixing) 정규화를 통해 서로 다른 잠재 벡터의 스타일을 섞어 레이어 간 상관관계를 줄인다. Progressive growing 기반으로 학습한다.",
    results: "FFHQ 데이터셋에서 FID 4.40을 달성하며 당시 최고 수준의 얼굴 생성 품질을 보였다. Perceptual Path Length, Linear Separability 등 새로운 평가 지표를 제안하여 잠재 공간의 품질을 정량적으로 측정했다. 스타일 믹싱을 통한 직관적인 속성 제어 능력을 시각적으로 입증했다.",
    impact: "고품질 이미지 생성의 새로운 기준을 세웠으며, StyleGAN2, StyleGAN3 등으로 발전하며 GAN 기반 생성 모델의 정점을 이루었다. 디핑페이크 논쟁을 촉발하는 등 사회적 영향도 컸으며, 이미지 편집, 도메인 적응, 데이터 증강 등 다양한 응용에 활용되었다. FFHQ 데이터셋은 얼굴 생성 연구의 표준 벤치마크가 되었다.",
    relatedFoundations: ["gan", "batch-normalization"],
    relatedPapers: [
      { id: "wgan", fieldId: "generative", title: "Wasserstein GAN", relation: "prior" },
    ],
  },

  "ldm": {
    tldr: "사전학습된 오토인코더의 잠재 공간에서 확산 과정을 수행하여 고해상도 이미지 생성의 계산 비용을 획기적으로 줄인 모델(Stable Diffusion의 기반)이다.",
    background: "DDPM 등 확산 모델은 뛰어난 생성 품질을 보였지만, 고해상도 이미지의 픽셀 공간에서 직접 노이즈 제거를 수행하므로 막대한 계산 비용이 필요했다. 수백 스텝의 반복적 추론과 고해상도 텐서 연산으로 인해 학습과 생성 모두 GPU 자원이 크게 요구되어 접근성이 제한적이었다.",
    keyIdea: "LDM은 확산 과정을 고차원 픽셀 공간이 아닌 저차원 잠재 공간(latent space)에서 수행하는 2단계 접근법을 제안한다. 먼저 오토인코더를 학습하여 이미지를 압축된 잠재 표현으로 인코딩하고, 이 잠재 공간에서 확산 모델을 학습한다. 잠재 공간의 차원이 픽셀 공간보다 훨씬 작으므로(예: 4~16배 다운샘플링) 계산량이 대폭 줄어든다. 또한 크로스 어텐션(cross-attention) 메커니즘을 통해 텍스트, 레이아웃, 시맨틱 맵 등 다양한 조건 입력을 유연하게 처리할 수 있는 범용 조건부 생성 프레임워크를 제공한다.",
    method: "1단계에서 KL-정규화 또는 VQ-정규화가 적용된 오토인코더를 학습한다. 2단계에서 잠재 공간의 U-Net 기반 확산 모델을 학습하며, 조건 입력은 도메인별 인코더(CLIP 텍스트 인코더 등)로 변환한 후 U-Net의 크로스 어텐션 레이어에 주입한다. Classifier-free guidance를 사용하여 조건 부합도를 높인다.",
    results: "인페인팅, 초해상도, 텍스트-이미지 생성 등에서 픽셀 공간 확산 모델과 동등하거나 우수한 FID를 달성하면서, 학습 및 추론 비용을 대폭 절감했다. 특히 256x256 이상의 고해상도 생성에서 효율성 차이가 두드러졌다.",
    impact: "Stable Diffusion의 핵심 기술로서 텍스트-이미지 생성의 대중화를 이끌었다. 오픈소스로 공개되어 AI 이미지 생성 생태계(Midjourney, DALL-E 등과 경쟁)의 폭발적 성장에 기여했으며, ControlNet, LoRA 파인튜닝 등 수많은 확장 기법의 기반이 되었다. 잠재 공간 확산이라는 패러다임은 비디오, 3D, 오디오 생성 등으로도 확장되고 있다.",
    relatedFoundations: ["vae", "ddpm"],
    relatedPapers: [
      { id: "dalle2", fieldId: "generative", title: "DALL-E 2", relation: "related" },
      { id: "flow-matching", fieldId: "generative", title: "Flow Matching", relation: "successor" },
    ],
  },

  "dalle2": {
    tldr: "CLIP 텍스트 임베딩으로부터 확산 사전(prior)을 거쳐 이미지를 생성하는 계층적 텍스트-이미지 생성 모델이다.",
    background: "DALL-E 1이 텍스트에서 이미지를 생성할 수 있음을 보였지만, 생성 품질과 해상도에 한계가 있었다. 한편 CLIP은 텍스트와 이미지를 공유 임베딩 공간에 매핑하여 강력한 멀티모달 표현을 학습했고, GLIDE는 텍스트 조건부 확산 모델로 높은 품질의 이미지 생성을 입증했다. 이 두 갈래의 발전을 통합할 방법이 필요했다.",
    keyIdea: "DALL-E 2(unCLIP)는 두 단계의 생성 과정을 거친다. 먼저 '사전 모델(prior)'이 CLIP 텍스트 임베딩으로부터 대응하는 CLIP 이미지 임베딩을 생성한다. 그런 다음 '디코더'가 이 CLIP 이미지 임베딩을 조건으로 실제 이미지를 생성한다. CLIP의 공동 임베딩 공간을 활용함으로써 텍스트의 의미적 내용이 이미지 생성에 충실히 반영되며, 이미지 임베딩의 보간(interpolation)이나 변환을 통해 이미지 변형과 편집이 자연스럽게 가능해진다. 디코더로는 GLIDE 기반의 확산 모델을 사용하며, 업샘플러를 통해 고해상도 출력을 얻는다.",
    method: "사전 모델은 자기회귀 모델 또는 확산 모델로 구현하며(확산 버전이 성능이 더 우수), CLIP 텍스트 임베딩을 입력받아 CLIP 이미지 임베딩을 예측한다. 디코더는 64x64 해상도의 GLIDE 변형 모델이며, 이후 두 단계의 업샘플러(64→256→1024)로 최종 1024x1024 이미지를 생성한다. Classifier-free guidance를 적용하여 텍스트 부합도와 다양성의 균형을 조절한다.",
    results: "텍스트-이미지 생성에서 GLIDE를 능가하는 포토리얼리즘과 텍스트 부합도를 보였다. CLIP 임베딩 공간에서의 보간으로 시맨틱하게 의미 있는 이미지 블렌딩이 가능했고, 텍스트 차이를 이용한 이미지 편집도 시연했다. 다만 텍스트 렌더링이나 공간 관계 표현에는 한계를 보였다.",
    impact: "텍스트-이미지 생성의 품질을 한 단계 끌어올려 대중적 관심을 크게 불러일으켰다. CLIP을 생성 모델의 핵심 구성 요소로 활용하는 패러다임을 확립했으며, 이미지 편집과 변형에 있어 새로운 가능성을 열었다. Stable Diffusion, Imagen 등 동시대 모델들과 함께 AI 이미지 생성 혁명을 촉발했다.",
    relatedFoundations: ["ddpm", "vae"],
    relatedPapers: [
      { id: "ldm", fieldId: "generative", title: "Latent Diffusion Models", relation: "related" },
      { id: "clip", fieldId: "multimodal", title: "CLIP", relation: "prior" },
    ],
  },

  "flow-matching": {
    tldr: "확률 경로를 따라 노이즈를 데이터로 변환하는 연속 정규화 흐름을 시뮬레이션 없이 학습하는 간결하고 효율적인 생성 모델링 프레임워크이다.",
    background: "확산 모델(DDPM)은 높은 생성 품질을 보였지만, 이론적으로는 확률적 미분방정식(SDE)에 기반하여 복잡하고, 수백 단계의 반복적 샘플링으로 인해 느렸다. 연속 정규화 흐름(CNF)은 결정론적 ODE 기반으로 우아하지만, 기존 학습 방법은 ODE 시뮬레이션이 필요하여 계산 비용이 높고 확장성에 한계가 있었다.",
    keyIdea: "플로우 매칭은 CNF를 시뮬레이션 없이(simulation-free) 학습하는 프레임워크를 제안한다. 핵심 아이디어는 시간에 따라 노이즈 분포에서 데이터 분포로 이동하는 확률 경로(probability path)를 정의하고, 이 경로를 생성하는 벡터장(vector field)을 신경망으로 회귀하는 것이다. 조건부 플로우 매칭(Conditional Flow Matching)을 통해 개별 데이터 포인트를 조건으로 하는 간단한 벡터장을 학습 목표로 사용함으로써, ODE를 풀 필요 없이 단순한 MSE 손실로 학습할 수 있다. 특히 직선 경로(straight paths)를 사용하는 최적 수송 조건부 플로우 매칭은 경로가 가장 짧고 교차하지 않아 효율적인 샘플링이 가능하다.",
    method: "시간 t에서의 조건부 확률 경로를 가우시안으로 정의하고(예: x_t = (1-t)x_0 + t*x_1, x_0은 노이즈, x_1은 데이터), 이에 대응하는 조건부 벡터장 u_t(x|x_1) = x_1 - x_0을 계산한다. 신경망 v_θ(t, x)가 이 벡터장을 근사하도록 MSE 손실로 학습한다. 샘플링 시에는 노이즈에서 시작하여 학습된 벡터장을 따라 ODE를 적분한다.",
    results: "ImageNet에서 확산 모델과 동등하거나 우수한 FID를 달성하면서, 더 적은 샘플링 스텝(약 100~250 NFE)으로 고품질 이미지를 생성했다. 직선 경로가 곡선 경로보다 일관되게 우수한 성능을 보였으며, 학습 안정성도 확산 모델에 비해 개선되었다.",
    impact: "생성 모델링의 새로운 이론적 프레임워크를 확립하여 확산 모델의 대안으로 급부상했다. Meta의 후속 연구를 통해 Stable Diffusion 3, Flux 등 차세대 이미지 생성 모델의 핵심 학습 방법으로 채택되었다. 단순한 수학적 정식화와 유연한 확장성으로 비디오, 오디오, 단백질 구조 생성 등 다양한 분야로 빠르게 확산되고 있다.",
    relatedFoundations: ["ddpm"],
    relatedPapers: [
      { id: "ldm", fieldId: "generative", title: "Latent Diffusion Models", relation: "prior" },
      { id: "dalle2", fieldId: "generative", title: "DALL-E 2", relation: "prior" },
    ],
  },
};
