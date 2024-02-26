# NLP Team3 HansolDeco QA Competition
![image](https://github.com/KU-BIG/KUBIG_2024_SPRING/assets/104672441/1694a3fe-da66-46af-93c9-efdd070010d0)
시트, 마루, 벽면, 도배와 같은 건축의 핵심 자재들의 품질 관리와 하자 판단 과정을 더욱 정교하고 효율적으로 만들고, 이러한 자재들의 관리 및 운용의 질을 향상시키는 것을 목적으로  NLP(자연어 처리) 기반의 QA (질문-응답) 시스템을 통해 도배하자와 관련된 깊이 있는 질의응답 처리 능력을 갖춘 AI 모델을 개발합니다.  
도배하자와 관련된 다양한 질문과 상황에 대한 정확하고 신속한 응답을 제공하는 AI 모델을 개발하는 것을 목표로 하며 실제 현장에서 발생할 수 있는 복잡한 상황에 대응하고, 고객의 문의에 신속하고 정확하게 답변할 수 있는 시스템을 구축합니다.
# Project Structure
```
Team 3/
│
├── train.py - main script to start training
├── inference.py - make submission with trained models
└── modules/ - functions and classes required to operate the model
    ├── dataloader.py
    ├── trainer.py
    └── utils.py
```
# Model
- [kogpt2-base-v2](https://huggingface.co/skt/kogpt2-base-v2)
- [OPEN-SOLAR-KO-10.7B](https://huggingface.co/beomi/OPEN-SOLAR-KO-10.7B)
- [LDCC-SOLAR-10.7B](https://huggingface.co/LDCC/LDCC-SOLAR-10.7B)
- [roberta-large](https://huggingface.co/klue/roberta-large)
