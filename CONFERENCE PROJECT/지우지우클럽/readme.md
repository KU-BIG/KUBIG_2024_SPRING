# 수제버거 추천 시스템 🍔
## Preview
<img width="612" alt="image" src="https://github.com/KU-BIG/KUBIG_2024_SPRING/assets/109716683/1629855b-25a6-4db6-8224-62e06df12eec">

[수제버거 추천시스템 링크](https://app-burgertest.streamlit.app/)

## Role
- **박세훈**: 협업 필터링 시스템 구현(ALS)
- **이승준**: 주제 선정 및 구체화, 메뉴 데이터 전처리, 포스터 최종 수합
- **임지우**: streamlit 구현, PPT 최종 수합
- **최주희**: 데이터 크롤링, 리뷰 데이터 전처리
- **최지우**: 콘텐츠 기반 필터링 시스템 구현(코사인 유사도 기반)
---

## Model Structure
<img width="1115" alt="image" src="https://github.com/KU-BIG/KUBIG_2024_SPRING/assets/109716683/a5ef0386-13ea-4e83-b8e2-84d4909e7c17">

### Data Crawling
메뉴 데이터()와 리뷰 데이터(유저 정보, 레스토랑 정보, 리뷰 텍스트 등) 각각 크롤링 진행, 추후에 추천시스템 사이트에 들어갈 버거 이미지 또한 크롤링 완료.  
