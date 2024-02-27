# 글자에 가려진 숫자 예측 프로젝트

## 24W DL Basic 2팀
### 18기 방서연, 18기 신인수, 19기 이동주, 19기 정종락(팀장)

## 소개글

Dacon의 CV 분류 예측 프로젝트 중 하나인 '글자에 가려진 숫자 예측 프로젝트'를 주제로 선정하여, 제공된 변형 EMNIST 데이터셋을 활용하여 CNN 모델 기반의 분류예측을 시도하였습니다. 가장 기본적인 코드로만 학습 후 예측하였을 때의 정확도는 0.6에 불과했지만, 매주 조원들과 프로젝트 회의 등을 통해 조금씩 정확도를 높이며 성능을 개선시켜왔습니다. 각 시도의 정확도를 기준으로 성능을 비교하면서 Data Augmentation, Max Pooling, Batch Normalization, Activation function 등에서 변화를 주었습니다. 또한 이 과정에서는 train dataset의 크기가 10배 가까이 적기에 robust한 모델을 만들자는 목표를 가졌습니다. 여러 모델을 기반으로 train을 거친 뒤, Dacon에 제출하면서 test 정확도를 확인한 결과, 'Batch Normalization + 데이터 증강 + SGD 옵티마이저' 조합의 CNN 모델이 0.86의 예측 정확도를 보이며 가장 좋은 성능을 보였습니다. CNN 모델 외에 추가적으로 GAN 모델 구현을 시도해보기도 하였습니다.

## 코드(노트북) 파일 구성

### 1. 24W_DL2_Adam.ipynb: Adam optimizer 기반
- Data Loading and EDA
- Baseline
- Batch Normalization
- BN + Data Augmentation
- BN + Pooling
- BN + Pooling + Dropout
- BN + Data Augmentation + Pooling + Dropout
- Training Summary
- Test Summary

### 2. 24W_DL2_SGD.ipynb: SGD optimizer 기반
- Data Loading and EDA
- Baseline
- Batch Normalization
- BN + Data Augmentation
- BN + Pooling
- BN + Pooling + Dropout
- BN + Data Augmentation + Pooling + Dropout
- Training Summary
- Test Summary

### 3. 24W_DL2_GAN_CGAN.ipynb: GAN, CGAN 추가 시도 코드
- Setup Environment
- GAN
- CGAN




