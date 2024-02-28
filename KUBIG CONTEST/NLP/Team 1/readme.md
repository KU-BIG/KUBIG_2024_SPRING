# 한솔데코 시즌2 AI 경진대회
#### NLP 1팀 김나연, 김송성, 안영지, 이승준
## [주제]
도배 하자 질의 응답 AI 모델 개발
- 도배 하자 도메인에 대한 질의를 바탕으로 지능적인 응답을 생성하는 AI 모델 개발
  
https://dacon.io/competitions/official/236216/overview/description

## Dataset Info.

train.csv [파일]   
- id : 질문 - 답변 (QA) 샘플 고유 번호  
- 질문_1, 질문_2 : 샘플 별 동일한 내용으로 구성된 질문 2개  
- category : 질문 - 답변 (QA) 샘플의 도메인 세부 분야  
- 답변_1, 답변_2, 답변_3, 답변_4, 답변_5 : 샘플 별 질문에 대한 동일한 답변 Reference 5개   


test.csv [파일]  
- id : 평가 질문 샘플 고유 번호  
- 질문 : 평가 샘플의 질의 내용  


sample_submission.csv [파일] - 제출 양식  
- id : 평가 질문 샘플 고유 번호  
- vec_0, vec_1 ... vec_511 : 생성된 답변을 512 차원의 Embedding Vector로 표현된 결과  


## Model
- kogpt2-base-v2
- LDCC-SOLAR-10.7B
- llama-2-ko-7b

```
CUDA_VISIBLE_DEVICES=0,1,2,3 LOCAL_RANK=0,1,2,3 torchrun --master_port 12345 --nproc_per_node 4 main.py
```
