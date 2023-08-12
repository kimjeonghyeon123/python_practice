# 도미
import numpy as np

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
# 빙어
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2,
                11.3, 11.8, 11.8, 12.0, 12.2,
                12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8,
                8.7, 10.0, 9.9, 9.8, 12.2,
                13.4, 12.2, 19.7, 19.9]

fish_length = bream_length + smelt_length
fish_weight = bream_weight + smelt_weight

# fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
# fish_target = [1] * 35 + [0] * 14

# 칼럼 스택
# np.full((2, 3), 9)
# 모든 항이 9인 2행 3열 행렬
fish_data = np.column_stack((fish_length, fish_weight))
print(fish_data)
# np.concatenate : 1열로 합쳐주는 함수
fish_target = np.concatenate((np.ones(35), np.zeros(14)))
print(fish_target)

# 훈련, 테스트 데이터 나눠줌
from sklearn.model_selection import train_test_split

# stratify : 편향이 생기지 않도록 타깃값을 전달해 훈련,테스트 데이터가 골고루 섞이게 함
# random_state : 섞이게 하는 값? 같은 거, 실제로는 필요없음
train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, stratify=fish_target, random_state=42
)

from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
print(kn.score(test_input, test_target))

print(kn.predict([[25, 150]]))

# 가장 가까운 5개의 점 거리와 인덱스 보여줌
distances, indexes = kn.kneighbors([[25, 150]])

import matplotlib.pyplot as plt

plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
# 스케일 맞추기
# plt.xlim((0, 1000))
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 표준점수로 바꾸기
# (특성 - 평균) / 표준편차

# 평균, 표준편차
mean = np.mean(train_input, axis=0)
# mean = [길이 평균, 몸무게 평균]
# axis = 1일 경우
# mean = [행의 평균, ..., ]
std = np.std(train_input, axis=0)
# std = [길이 표준편차, 몸무게 표준편차]
print(mean, std)
# 넘파이 브로드캐스팅
# 각 행마다 계산해줌
train_scaled = (train_input - mean) / std
print(train_scaled)

new = ([25, 150] - mean) / std

plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

kn.fit(train_scaled, train_target)
test_scaled = (test_input - mean) / std
print(kn.score(test_scaled, test_target))

print(kn.predict([new]))

distances, indexes = kn.kneighbors([new])

plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()