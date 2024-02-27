프로젝트 설명
=============

월간 데이콘 예술 작품 화가 분류 AI 경진대회   
    (https://dacon.io/competitions/official/236006/overview/description)   
예술 작품의 일부분만 주어지는 테스트 데이터셋에 대해 올바르게 화가를 분류해내는 Multilabel Classification CNN 태스크

* 학습 데이터셋은 대표적인 화가 50명에 대한 예술 작품 이미지 제공(5911개)
* 테스트 데이터셋은 대표적인 화가 50명에 대한 예술 작품 이미지의 일부분(약 1/4)만 제공(12670개)
* 학습에 활용할 수 있는 화가 50명에 대한 특징 정보(csv) 추가 제공

           


프로젝트 진행
=============

##EDA 및 전처리
* 50명의 화가에 대한 그림의 개수가 매우 다름 (반 고흐는 600개, 잭슨 폴록은 20개)
  * weighted random sampling을 통해 모든 클래스에 대해 비슷한 개수의 샘플을 추출
* 흑백 이미지가 섞여있음 (프란시스 고야의 작품이 가장 많았음)
  * 모델에 같은 차원의 이미지를 넣어주어야 하므로 .RGB를 통해 3차원으로 변경
* ID 3896과 3986 값이 서로 바뀌어 있음
  * 인덱스 수정

##Data Augmentation
각각의 augmentation 진행할 확률을 0.5로 설정
* RandomResizedCrop : 입력 이미지를 ¼ 크기로 crop
* HorizontalFlip : 좌우반전
* VerticalFlip : 상하반전
* ShiftScaleRotate : 이미지 무작위 이동 + 크기 조절 + 회전
* Normalize : R,G,B 평균값을 0으로. 각 pixel 값 – 평균 pixel 값

##Model
1. ResNet
2. EfficientNet b0 -> 이미지 resolution을 맞추기 위해 224로 학습
3. EfficientNet b3 -> 이미지 300으로 resize
4. EfficientNet b4 -> 이미지 380으로 학습하면 코랩 무료버전에서는 Cuda out of memory

최종적으로 EfficientNet b3 모델 선정




프로젝트 결과
=============

데이콘 리더보드 제출시 Public 0.7665(53등), Private 0.7601(31등)
