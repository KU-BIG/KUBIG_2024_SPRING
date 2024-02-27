# 🦩 OpenFlamingo를 활용한 Multimodal task 구현
CV 2팀 : 18기 백성은 / 19기 강지윤
<br>
<br>


## Paper List

[Open-flamingo Paper](https://arxiv.org/abs/2308.01390)

[Prompt Engineering in Image Captioning Paper](https://arxiv.org/abs/2211.09699)

[Prompt Engineering in Hateful Memes](https://arxiv.org/abs/2308.01390) 

## Task Definition & Objective
Open-Flamingo를 활용하여 여러 Multimodal task를 구현해보고 논문의 성능과 비교해보는 reproducing 및 prompt engineering을 통해 기존보다 더 향상된 성능을 얻고자 시도했습니다.

구현한 Multimodal task는 다음과 같습니다.

- Image Captioning
- Visual Question Answering (VQA)
- Image Classification

Image Captioning은 Colab의 resource 한계로 qualitative evaluation으로 진행했습니다. Captioning과 Classification에서 prompt engineering을 수행했으며, VQA는 resource 한계로 구현만 진행했습니다.

## Dataset
- Image Captioning
  - MS-COCO
  
- Visual Question Answering (VQA)
  - Vizwiz
  - Textvqa

- Image Classification
  - Hateful Memes
