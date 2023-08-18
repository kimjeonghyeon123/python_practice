'''
회귀(Regression)
    - 지도학습
        - 분류 : 타깃을 0 과 1로 해두고 분류함
        - 회귀 : 타깃 값이 곧 예측 값임(사전적 의미X)

K-최근접 이웃
    - K-최근접 이웃 분류
        - 근처 타깃의 개수가 많은 쪽
    - K-최근접 이웃 회귀
        - 근처 타깃들의 평균
'''
import numpy as np

perch_length = np.array(
    [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0,
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5,
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5,
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0,
     36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0,
     40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
     )
perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
     1000.0, 1000.0]
     )

import matplotlib.pyplot as plt

plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 훈련 세트 준비
# 전부 1차원 배열로 나옴
# 훈련 데이터는 2차원, 타겟 데이터는 1차원

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42
)

train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
# 회귀 모델 훈련
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor()
knr.fit(train_input, train_target)
print(knr.score(test_input, test_target))

# R^2 = 1 - ((타깃 - 예측)^2 의 합 / (타깃 - 평균)^2 의 합)
from sklearn.metrics import mean_absolute_error

test_prediction = knr.predict(test_input)
# 테스트 타겟과 테스트 인풋으로 예측한 값의 차이
# 쉽게 말해 오차를 평균내서 전달해줌
mae = mean_absolute_error(test_target, test_prediction)
print(mae)

# 과대적합(overfitting)과 과소적합(underfitting)
# 보통은 훈련 데이터가 높게 나오고
# 테스트 데이터가 낮게 나옴
# 과소 적합 : 훈련 데이터 점수가 테스트 데이터 점수보다 더 낮게 나오는 경우
# 과대 적합 : 테스트 데이터 점수가 현저히 낮은 경우
print('5 일때')
print(knr.score(train_input, train_target))
print(knr.score(test_input, test_target))

# 이웃의 개수
# 이웃의 개수를 줄이면 과대적합
# 이웃의 개수를 높이면 과소적합
# 기본값은 5임
knr.n_neighbors = 3
print('3일 때')
print(knr.score(train_input, train_target))
print(knr.score(test_input, test_target))

