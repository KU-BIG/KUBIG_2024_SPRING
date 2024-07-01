import numpy as np
import tqdm 
from implicit.als import AlternatingLeastSquares as ALS  
import pandas as pd
import scipy


# ALS 협업 필터링
zip_file = 'review_data.csv.zip'
csv_file = 'review_data.csv'

# ZIP 파일 열기
with zipfile.ZipFile(zip_file, 'r') as zipf:
    with zipf.open(csv_file) as file:
        train = pd.read_csv(file)



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
