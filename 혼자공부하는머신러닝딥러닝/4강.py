'''
훈련데이터
- 입력
    - 특성들
- 타깃

- 지도학습
위와 같은 입력과 타깃이 있는 데이터를 사용해 학습하는 알고리즘
- 비지도학습
타깃이 없는 데이터를 사용해 학습하는 알고리즘
- 강화학습
행동 수행 후 피드백 받고 보상을 최대화하는 것
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
# 빙어
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2,
                11.3, 11.8, 11.8, 12.0, 12.2,
                12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8,
                8.7, 10.0, 9.9, 9.8, 12.2,
                13.4, 12.2, 19.7, 19.9]

length = bream_length + smelt_length
weight = bream_weight + smelt_weight

fish_data = [[l, w] for l, w in zip(length, weight)]
fish_target = [1] * 35 + [0] * 14

# 훈련 세트
# 슬라이싱
train_input = fish_data[:35]
train_target = fish_target[:35]

# 테스트 세트
test_input = fish_data[35:]
test_target = fish_target[35:]

# 테스트 세트에서 평가하기
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
# 훈련 세트로 훈련시키기
kn.fit(train_input, train_target)
# 테스트 세트로 평가하기
print(kn.score(test_input, test_target))

# 도미만 들어간 데이터로 훈련 후
# 빙어 데이터로 테스트 하면 안됨
# 도미와 빙어가 같이 들어간 테스트가 필요

# 넘파이 사용하기
import numpy as np

input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

# 49개의 샘플(행)
# 2개의 특성(열)
print(input_arr)

# 0~48까지 배열만들어줌
index = np.arange(49)
print(index)
# 섞어줌
np.random.shuffle(index)
print(index)
# 배열 슬라이싱
# a = np.array([5,6,7,8])
# a[[1,3]]
# => [6,7]

train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

import matplotlib.pyplot as plt

plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(test_input[:, 0], test_input[:, 1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

kn = kn.fit(train_input, train_target)
print(kn.score(test_input, test_target))



