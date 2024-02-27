# 24-Spring KUBIG CONTEST - NLP 분반 4팀
## 팀원
17기 황민아 18기 진서연 최유민
## 주제 : 도배 하자 질의 응답 처리 
https://dacon.io/competitions/official/236216/overview/description
## 목표
본 프로젝트의 목표는 도배 하자에 대한 질문의 답변을 생성하는 것으로, 구체적으로는 다음과 같은 인사이트를 얻고자 하였다.

<img width="563" alt="image" src="https://github.com/KU-BIG/KUBIG_2024_SPRING/assets/138692039/c03d3001-c6c7-4153-9d4f-1f8a72ec36b0">

## 데이터 소개
### Train data
1. id : 샘플 고유 번호
2. 질문_1, 질문_2 : 샘플 별 동일한 내용으로 구성된 질문 2개
3. category : 질문 - 답변 (QA) 샘플의 도메인 세부 분야
4. 답변_1, 답변_2, 답변_3, 답변_4, 답변_5 : 샘플 별 질문에 대한 동일한 답변 Reference 5개

### Test data
1.  id : 샘플 고유 번호
2.  질문
## 모델
베이스 모델로는 Kogpt2, LLAMA-2-ko-7b, SOLAR-10.7B, Gemma-7B를 선택하였다.

더불어 프로젝트의 메인인 "RAG"모델도 활용하였다. 
