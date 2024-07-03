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
## 프로젝트 배경 및 필요성
### Problem
프리미엄 수제버거에 대한 한국인의 지속적이고 뜨거운 관심을 확인할 수 있다. 프리미엄 수제버거 시장이 커지고 수요가 생겨나고 있으나 사람들의 취식 경험은 저가형 프랜차이즈 버거에 머물러 있다. 이런 상황 속에서 **본인의 취향에 맞는 버거 가게, 메뉴를 선택함에 있어 어려움을 겪는다.**

### Solution
유저의 저가형 프랜차이즈 버거 취식경험을 입력으로 받아 **취향에 맞는 프리미엄 버거 가게 및 메뉴를 추천해준다.** 

## Dataset
### Data Crawling
- 메뉴 데이터: 콘텐츠 기반 필터링에 사용
  - 저가형 브랜드 버거: 가게 이름, 메뉴 이름, 가격, 한 줄 설명, 재료 
  - 프리미엄 수제 버거: 가게 이름, 평점, 위치, 방문자 리뷰 수, 블로그 리뷰 수, 메뉴 이름, 가격, 한 줄 설명, 재료(전처리로 제작)
- 리뷰 데이터: 협업 필터링에 사용
  - 가게 이름, 유저 이름, 유저 식별정보(리뷰 수, 사진 수, 팔로워 수), 별점, 방식(예약, 대기시간, 목적, 동행인), 리뷰 내용, 키워드, 방문 날짜, 방문 횟수

### Preprocessing
**메뉴 데이터**
1. 기본 데이터 전처리: '버거가 아닌 메뉴' 및 '세트 메뉴' 제거
2. 형태소 분리: KoNLPy 사용, 버거 도메인에 맞는 불용어 및 예외 단어 사전 구축, 햄버거 설명 문장 간의 유사도 비교를 위함
3. 패티 열 생성: GPT 4.0 프롬프팅. 소고기 패티, 새우 패티 등 콘텐츠 기반 필터링 시 1차 구분을 위함
4. 위경도 값 추출: 주소 데이터에 기반해 버거 가게 위경도 추출, 결과 구현 시 UI 내 가게 정보 노출을 위함


**리뷰 데이터**
1. 중복값 처리: 유저 정보로 groupby했을 때 동일한 사람이 다른 사람으로 인식되는 경우 전처리
2. 리뷰 텍스트 처리: 이모티콘 및 특수문자 제거
     

## Model Structure
<img width="1115" alt="image" src="https://github.com/KU-BIG/KUBIG_2024_SPRING/assets/109716683/a5ef0386-13ea-4e83-b8e2-84d4909e7c17">

### 콘텐츠 기반 필터링(cosine similarity)
- TF-IDF로 단어 임베딩 진행
- 벡터화된 아이템에 대해 코사인 유사도 행렬 생성
- 유사도 기반 추천 이전에 가격의 min, max 값 설정 + 동일 패티만 추천되도록 설정

**=> 선택한 프랜차이즈 버거와 가장 유사한 상위 5개 수제버거 추천**
```python
# wordlist 열을 TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(' '))
tfidf_matrix = tfidf_vectorizer.fit_transform(burger_data['wordlist'])


# 코사인 유사도 계산
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 코사인 유사도를 DataFrame으로 변환하여 확인
similarity_df = pd.DataFrame(cosine_sim, index=burger_data['menu'], columns=burger_data['menu'])
similarity_df.head()

# 최종 콘텐츠 기반 필터링 진행
filtered_recommendations = recommendations_df[
        (recommendations_df['class'] == 1) &
        (recommendations_df['price'] >= min) &
        (recommendations_df['price'] < max) &
        ((recommendations_df['visitor'] + recommendations_df['blog']) >= popularity_min) &
        (recommendations_df['patty'].str.contains(selected_burger_patty, na = False))
  ]

  filtered_recommendations = filtered_recommendations.drop_duplicates(subset='menu')

  # 최종 추천 점수를 계산하여 상위 5개 추천 확인
  final_recommendations = filtered_recommendations[['id', 'menu', 'name', 'price', 'score']].set_index('menu').sort_values(by='score', ascending=False).iloc[1:11]
  return final_recommendations
```

### 협업 필터링(ALS)
사용자와 아이템의 잠재 행렬을 순차적으로 최적화하는 matrix factorization인 ALS(Alternating Least Squares)을 사용하여 1순위로 추천된 버거가 속한 레스토랑과 가장 유사한 버거집 5개를 추가로 추천. 

```python
# 데이터셋 불러오기
train = pd.read_csv('1차 리뷰 전처리(원본).csv')
train = train[['username','restaurant']]
train.columns = ['user_id', 'rest_id']

# 데이터 <--> 인덱스 교환 딕셔너리
user2idx = {}
for i, l in enumerate(train['user_id'].unique()):
    user2idx[l] = i
    
rest2idx = {}
for i, l in enumerate(train['rest_id'].unique()):
    rest2idx[l] = i

idx2user = {}
for i, l in enumerate(train['user_id'].unique()):
    idx2user[i] = l

idx2rest = {}
for i, l in enumerate(train['rest_id'].unique()):
    idx2rest[i] = l

# 인덱스 생성
data = train.copy()
useridx = data['useridx'] = train['user_id'].apply(lambda x: user2idx[x]).values
restidx = data['restidx'] = train['rest_id'].apply(lambda x: rest2idx[x]).values
rating = np.ones(len(data))

# 희소 행렬(csr_matrix)
purchase_sparse = scipy.sparse.csr_matrix((rating, (useridx, restidx)), shape=(len(set(useridx)), len(set(restidx))))
# ALS 모델 초기화
als_model = ALS(factors=40, regularization=0.01, iterations=50)
# 모델 최적화
als_model.fit(purchase_sparse, show_progress=False)
    related = als_model.similar_items(rest2idx[unique_names[0]])
```
