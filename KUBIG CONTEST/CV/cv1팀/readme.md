# 인공지능 요가 선생
YOGIS팀(18기 원준혁, 19기 김은우, 이서현, 이소희)

<br>

## Objective
__Open pose__ 모델을 활용한 __yoga pose estimation__ 및 자세 __feedback__ 제공

<br>

## Model
Openpose model
- pose estimation에서 많이 활용되는 Openpose 모델의 api 활용
- COCO dataset 기반 총 18개의 keypoint 좌표 및 confidence 제공

<br>


## Pose estimation
yoga 데이터 set-> keypoint 추출 및 벡터 변환-> classifier 진행<br>
- Keypoint 적용 모델 : KNN, Random Forest, __MLP__
- Keypoint + 이미지 : __Resnet+MLP__

<br>

## Feedback
- 평가 지표 : __OKS__(거리 기반). __Cosine similarity__(각도 기반)
- Image-Image : keypoint의 거리 및 좌표를 활용한 __구체적인 피드백__ 제공
- Video-Image : skeleton 생성 및 __실시간 피드백__ 제공
