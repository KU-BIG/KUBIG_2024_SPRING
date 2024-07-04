# Load the dataset
data <- read.csv('D:/R/data_all2.csv')


# Re-categorize '노면상태'
data$노면상태 <- ifelse(data$노면상태 == '건조', '건조', '기타')

# Re-categorize '요일'
categorize_day <- function(day) {
  if (day %in% c('월요일', '화요일', '수요일', '목요일')) {
    return('월화수목')
  } else if (day == '금요일') {
    return('금')
  } else if (day %in% c('토요일', '일요일')) {
    return('주말')
  }
}

# Add new column for re-categorized '사고요일'
data$사고요일 <- sapply(data$요일, categorize_day)

# Re-categorize '도로형태'
categorize_road_type <- function(road_type) {
  if (grepl('단일로', road_type)) {
    return('단일로')
  } else if (grepl('교차로', road_type)) {
    return('교차로')
  } else {
    return('기타')
  }
}

# Add new column for re-categorized '도로형태'
data$도로형태 <- sapply(data$도로형태, categorize_road_type)

# 필요한 라이브러리 로드
library(dplyr)


# 기상상태 재범주화
data <- data %>%
  mutate(기상상태 = case_when(
    기상상태 %in% c('맑음','구름','흐림') ~ '맑음',
    기상상태 == '안개' ~ '안개',
    기상상태 %in% c('비', '눈') ~ '비',
    TRUE ~ '기타'  # 기타 상태 처리
  ))



# Re-categorize '사고시간대'
categorize_time_period <- function(time_period) {
  if (time_period %in% c('0시 ~ 3시', '3시 ~ 6시')) {
    return('심야')
  } else if (time_period %in% c('6시 ~ 9시', '9시 ~ 12시')) {
    return('아침')
  } else if (time_period %in% c('12시 ~ 15시', '15시 ~ 18시')) {
    return('오후')
  } else if (time_period %in% c('18시 ~ 21시', '21시 ~ 23시')) {
    return('저녁')
  }
}

# Add new column for re-categorized '사고시간대'
data$사고시간대 <- sapply(data$사고시간대, categorize_time_period)

# Convert '사고일시' to Date
data$사고일시 <- as.Date(data$사고일시)

# Function to assign season
get_season <- function(date) {
  month <- as.numeric(format(date, "%m"))
  if (month %in% c(12, 1, 2)) {
    return('겨울')
  } else if (month %in% c(3, 4, 5)) {
    return('봄')
  } else if (month %in% c(6, 7, 8)) {
    return('여름')
  } else if (month %in% c(9, 10, 11)) {
    return('가을')
  }
}

# Apply function to create '계절' column
data$사고계절 <- sapply(data$사고일시, get_season)

# Select needed columns
selected_columns <- c('가해운전자.성별', '가해운전자.연령', '가해운전자.차종',
                      '노면상태', '도로형태', '기상상태',
                      '사고요일', '사고시간대', '사고계절',
                      '법규위반', '피해운전자.상해정도','사망자수', '중상자수', '경상자수', '부상신고자수')

data <- data[selected_columns]

# 성별에서 '기타/불명' 제거
data <- data %>%
  filter(가해운전자.성별 != '기타불명')

# 종속 변수와 독립 변수 설정
data$피해운전자.상해정도 <- factor(data$피해운전자.상해정도, levels = c('상해없음', '부상신고', '경상', '중상', '사망'), labels = c(0, 0, 1, 2, 3), ordered = TRUE)
data$가해운전자.성별 <- as.factor(data$가해운전자.성별)
data$가해운전자.차종 <- as.factor(data$가해운전자.차종)
data$노면상태 <- as.factor(data$노면상태)
data$도로형태 <- as.factor(data$도로형태)
data$기상상태 <- as.factor(data$기상상태)
data$사고요일 <- as.factor(data$사고요일)
data$사고시간대 <- as.factor(data$사고시간대)
data$사고계절 <- as.factor(data$사고계절)
data$법규위반 <- as.factor(data$법규위반)


# Display the resulting data
print(head(data))

# Filter out non-elderly drivers
data_ne <- data[data$가해운전자.연령 < 65, ]
data_e <- data[data$가해운전자.연령 >= 65, ]


write.csv(data_ne, "D:/R/data_not_elder.csv", row.names = FALSE, fileEncoding = "UTF-8")
write.csv(data_e, "D:/R/data_elder2.csv", row.names = FALSE, fileEncoding = "CP949")

################################################비고령###########################################

install.packages("MASS")
install.packages("lmtest")

library(MASS)
library(lmtest)


# 순서형 로지스틱 회귀분석 모델 적합
model_ne <- polr(피해운전자.상해정도 ~ 가해운전자.성별 + 가해운전자.연령 + 가해운전자.차종 + 도로형태 + 기상상태 + 사고요일 + 사고시간대 + 사고계절 + 법규위반, data = data_ne, Hess = TRUE)
# 모델 요약
summary(model_ne)

# Wald 통계량 및 p-값 계산
wald_results <- coeftest(model_ne)

# 결과 출력
print(wald_results)
################################################고령###########################################


# 순서형 로지스틱 회귀분석 모델 적합
model_e <- polr(피해운전자.상해정도 ~ 가해운전자.성별 + 가해운전자.연령 + 가해운전자.차종 + 도로형태 + 기상상태 + 사고요일 + 사고시간대 + 사고계절 + 법규위반, data = data_e, Hess = TRUE)

# 모델 요약
summary(model_e)

# Wald 통계량 및 p-값 계산
wald_results <- coeftest(model_e)

# 결과 출력
print(wald_results)


################################################################################

########################################적합도검정####################################3
# 라이브러리 로드
library(MASS)
library(pROC)
library(pscl)
install.packages('pscl')


####################고령모델######################
# AIC 값 출력
model_aic <- AIC(model_e)
print(paste("AIC: ", model_aic))

# 로그우도 출력
model_logLik <- logLik(model_e)
print(paste("Log-likelihood: ", model_logLik))

# 오즈비 계산
exp(coef(model_e))

# Pseudo R-squared 계산
model_pseudoR2 <- pR2(model_e)
print(model_pseudoR2)


library(effects)
library(MASS)  # polr 함수가 포함된 패키지
# 예측 확률 계산

effect_plot <- allEffects(model_e)

# 예측 확률 시각화
plot(effect_plot)

# 특정 변수의 효과 추출
cross_effect <- effect("교차로_운행방법위반", model_e)

# 효과 요약 출력
summary(cross_effect)

# 효과 시각화
plot(cross_effect)
####################비비고령모델######################
# AIC 값 출력
model_aic <- AIC(model_ne)
print(paste("AIC: ", model_aic))

# 로그우도 출력
model_logLik <- logLik(model_ne)
print(paste("Log-likelihood: ", model_logLik))

# Pseudo R-squared 계산
model_pseudoR2 <- pR2(model_ne)
print(model_pseudoR2)


##################################################




# 순서형 로지스틱 회귀분석 모델 적합
model_e2 <- polr(피해운전자.상해정도 ~ 법규위반, data = data_e, Hess = TRUE)

# 모델 요약
summary(model_e2)

# Wald 통계량 및 p-값 계산
wald_results <- coeftest(model_e2)

# 결과 출력
print(wald_results)

