// Foundation paper → related field papers mapping
export const foundationToFieldLinks: Record<string, { fieldId: string; paperId: string; year: number; title: string }[]> = {
  "transformer": [
    { fieldId: "nlp", paperId: "t5", year: 2019, title: "T5: Text-to-Text Transfer Transformer" },
    { fieldId: "nlp", paperId: "xlnet", year: 2019, title: "XLNet" },
    { fieldId: "llm", paperId: "gpt4", year: 2023, title: "GPT-4" },
    { fieldId: "llm", paperId: "llama", year: 2023, title: "LLaMA" },
    { fieldId: "cv", paperId: "detr", year: 2020, title: "DETR" },
    { fieldId: "multimodal", paperId: "clip", year: 2021, title: "CLIP" },
  ],
  "bert": [
    { fieldId: "nlp", paperId: "elmo", year: 2018, title: "ELMo (선행)" },
    { fieldId: "nlp", paperId: "xlnet", year: 2019, title: "XLNet (후속)" },
    { fieldId: "nlp", paperId: "t5", year: 2019, title: "T5 (후속)" },
    { fieldId: "representation", paperId: "mae", year: 2022, title: "MAE" },
  ],
  "gpt": [
    { fieldId: "nlp", paperId: "instructgpt", year: 2022, title: "InstructGPT" },
    { fieldId: "llm", paperId: "gpt4", year: 2023, title: "GPT-4" },
    { fieldId: "llm", paperId: "llama", year: 2023, title: "LLaMA" },
  ],
  "gpt3": [
    { fieldId: "llm", paperId: "chinchilla", year: 2022, title: "Chinchilla" },
    { fieldId: "llm", paperId: "cot", year: 2022, title: "Chain-of-Thought" },
    { fieldId: "llm", paperId: "llama", year: 2023, title: "LLaMA" },
    { fieldId: "llm", paperId: "gpt4", year: 2023, title: "GPT-4" },
    { fieldId: "nlp", paperId: "instructgpt", year: 2022, title: "InstructGPT" },
  ],
  "resnet": [
    { fieldId: "cv", paperId: "mask-rcnn", year: 2017, title: "Mask R-CNN" },
    { fieldId: "cv", paperId: "detr", year: 2020, title: "DETR" },
    { fieldId: "representation", paperId: "simclr", year: 2020, title: "SimCLR" },
    { fieldId: "representation", paperId: "byol", year: 2020, title: "BYOL" },
  ],
  "alexnet": [
    { fieldId: "cv", paperId: "yolo", year: 2015, title: "YOLO" },
    { fieldId: "cv", paperId: "unet", year: 2015, title: "U-Net" },
    { fieldId: "efficient", paperId: "distillation", year: 2015, title: "Knowledge Distillation" },
  ],
  "gan": [
    { fieldId: "generative", paperId: "wgan", year: 2017, title: "WGAN" },
    { fieldId: "generative", paperId: "stylegan", year: 2019, title: "StyleGAN" },
  ],
  "vae": [
    { fieldId: "generative", paperId: "ldm", year: 2022, title: "Latent Diffusion (Stable Diffusion)" },
  ],
  "ddpm": [
    { fieldId: "generative", paperId: "ldm", year: 2022, title: "Latent Diffusion" },
    { fieldId: "generative", paperId: "dalle2", year: 2022, title: "DALL-E 2" },
    { fieldId: "generative", paperId: "flow-matching", year: 2023, title: "Flow Matching" },
  ],
  "vit": [
    { fieldId: "cv", paperId: "detr", year: 2020, title: "DETR" },
    { fieldId: "cv", paperId: "sam", year: 2023, title: "SAM" },
    { fieldId: "representation", paperId: "dino", year: 2021, title: "DINO" },
    { fieldId: "representation", paperId: "mae", year: 2022, title: "MAE" },
    { fieldId: "multimodal", paperId: "clip", year: 2021, title: "CLIP" },
  ],
  "attention-mechanism": [
    { fieldId: "nlp", paperId: "elmo", year: 2018, title: "ELMo" },
    { fieldId: "cv", paperId: "detr", year: 2020, title: "DETR" },
    { fieldId: "graph", paperId: "gat", year: 2017, title: "GAT" },
  ],
  "lstm": [
    { fieldId: "nlp", paperId: "elmo", year: 2018, title: "ELMo" },
    { fieldId: "rl", paperId: "dqn", year: 2013, title: "DQN" },
    { fieldId: "audio", paperId: "wavenet", year: 2016, title: "WaveNet" },
  ],
  "adam": [
    { fieldId: "optimization", paperId: "adamw", year: 2017, title: "AdamW" },
    { fieldId: "optimization", paperId: "lion", year: 2023, title: "Lion" },
  ],
  "dropout": [
    { fieldId: "efficient", paperId: "distillation", year: 2015, title: "Knowledge Distillation" },
    { fieldId: "efficient", paperId: "lora", year: 2021, title: "LoRA" },
  ],
  "batch-normalization": [
    { fieldId: "generative", paperId: "stylegan", year: 2019, title: "StyleGAN" },
  ],
  "scaling-laws": [
    { fieldId: "llm", paperId: "chinchilla", year: 2022, title: "Chinchilla" },
    { fieldId: "llm", paperId: "llama", year: 2023, title: "LLaMA" },
    { fieldId: "llm", paperId: "gpt4", year: 2023, title: "GPT-4" },
  ],
  "backpropagation": [
    { fieldId: "optimization", paperId: "adamw", year: 2017, title: "AdamW" },
  ],
  "seq2seq": [
    { fieldId: "nlp", paperId: "t5", year: 2019, title: "T5" },
    { fieldId: "audio", paperId: "whisper", year: 2022, title: "Whisper" },
  ],
  "lenet": [
    { fieldId: "cv", paperId: "yolo", year: 2015, title: "YOLO" },
    { fieldId: "cv", paperId: "unet", year: 2015, title: "U-Net" },
  ],
};
