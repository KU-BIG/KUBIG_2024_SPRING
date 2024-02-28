# 2024 봄 KUBIG CONTEST - ML 분반 1팀
---
## 팀원 
18기 심서현
19기 하진우 

## 주제: 부동산 경매 매물 플랫폼 (경매 낙찰가 예측)

https://dacon.io/competitions/official/17801/overview/description


## 목표

부동산 경매를 진행하는 입찰인은 일반적으로 진행 중인 경매 매물, 적정 입찰가액, 낙찰 확률, 낙찰가 등의 정보를 필요로 한다. 그러나, 이러한 정보들은 다른 사이트에 분산되어 있는 경우가 많으며, 낙찰 확률, 적정 입찰가액 등의 정보는 유료 서비스를 이용해야 하는 등, 정보에 대한 접근성이 다소 떨어지는 편이다. 이러한 문제를 고려하여, 본 프로젝트는 제공된 아파트 부동산의 정보를 토대로 하여 경매 낙찰가를 정확하게 예측해 소비자가 원하는 조건의 부동산 매물을 보다 저렴한 가격에 구매할 수 있는 서비스를 개발하는 것을 목표로 한다. 해당 서비스는 하나의 서비스로 경매 진행에 필요한 모든 정보 수집 및 경매가 계산을 가능하게 하여 정보 수집에 대한 입찰인의 피로도를 저하하고 빠르고 정확한 정보를 산출해내는 효과를 기대할 수 있다. 

## 변수 

    Auction_key                   경매 아파트 고유 키값
    
    Auction_class                  경매구분
    
    Bid_class                      입찰구분(일반/개별/일괄)
    
    Claim_price                    경매 신청인의 청구 금액  
    
    Appraisal_company             감정사
    
    Appraisal_date                 감정일자
    
    Auction_count                 총경매횟수
    
    Auction_miscarriage_count      총유찰횟수
    
    Total_land_gross_area          총토지전체면적
    
    Total_land_real_area            총토지실면적
    
    Total_land_auction_area        총토지경매면적
    
    Total_building_area            총건물면적
    
    Total_building_auction_area     총건물경매면적
    
    Total_appraisal_price           총감정가  
    
    Minimum_sales_price          최저매각가격
    
    First_auction_date             최초경매일
    
    Final_auction_date            최종경매일
    
    Final_result                   최종결과 
    
    Creditor                      채권자, 경매신청인
    
    addr_do                      주소_시도
    
    addr_si                       주소_시군구
    
    addr_dong                    주소_읍면동
    
    addr_li                      주소_리
    
    addr_san                    주소_산번지 여부
    
    Apartment_usage             건물(토지)의 대표 용도
    
    Preserve_regist_date           보존등기일
    
    Total_floor                    총층수
    
    Current_floor                  현재층수
    
    Share_auction_YorN           지분경매 여부
    
    Close_date                   종국일자
    
    Close_result                 종국결과
    
    point.y                      위도
    
    point.x                      경도
    
    Hammer_price                 낙찰가



## 프로젝트 진행 방식 

### 데이터 분석 및 전처리를 통해 타깃 변수인 낙찰가에 영향을 미치는 변수를 추출
### 파생 변수를 생성하여 성능을 개선 

* Apart : 상세 주소(addr_ect) 내 아파트 브랜드 명을 추출

* Ham / Min : 낙찰가 / 최소경매가 비율. 최소 경매가의 상관관계 지수가 너무 높아 결과가 편향되는 것을 방지. 

* Floor_rate : 해당 층수 / 아파트 전체 층수 비율. 아파트 층수 normalization

### 실거래가 데이터를 반영하여 예측 성능 개선

### 모델 선정 

* RandomForest

* XGBoost

* LightGBM 

### 성능 평가 

MSE, RMSE, R2 지수를 바탕으로 모델 성능 평가 진행 

