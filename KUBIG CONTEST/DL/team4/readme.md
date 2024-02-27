데이터 전처리 
Data Source: AI 허브에서 제공하는 8만여 개의 텍스트 데이터
Tokenizer: split(), Kiwi.analyze()
데이터 유형: TabularDataset

모델 & 아키텍처
Transformer Architecture: 인코더와 디코더의 Input으로 각각 표준어, 사투리 문장 입력

결과 및 평가
점수 산출 공식: nltk.translate.bleu_score 라이브러리의 corpus_bleu() 함수
