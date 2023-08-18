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

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0,
                29.7, 29.7, 30.0, 30.0, 30.7,
                31.0, 31.0, 31.5, 32.0, 32.0,
                32.0, 33.0, 33.0, 33.5, 33.5,
                34.0, 34.0, 34.5, 35.0, 35.0,
                35.0, 35.0, 36.0, 36.0, 37.0,
                38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0,
                450.0, 500.0, 390.0, 450.0, 500.0,
                475.0, 500.0, 500.0, 340.0, 600.0,
                600.0, 700.0, 700.0, 610.0, 650.0,
                575.0, 685.0, 620.0, 680.0, 700.0,
                725.0, 720.0, 714.0, 850.0, 1000.0,
                920.0, 955.0, 925.0, 975.0, 950.0]

import matplotlib.pyplot as plt

plt.scatter(bream_length, bream_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 훈련 세트 준비
# 전부 1차원 배열로 나옴
# 훈련 데이터는 2차원, 타겟 데이터는 1차원
import numpy as np
length = np.array(bream_length).reshape(-1, 1)
weight = np.array(bream_weight)

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    length, weight, random_state=42
)

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

# 과대적합이므로 이웃의 개수를 높여야 됨
knr.n_neighbors = 6
print('6일 때')
print(knr.score(train_input, train_target))
print(knr.score(test_input, test_target))

