# 도미와 빙어
# 2개의 클래스
# 분류(classification)
# 그래프로 그리기
# 산점도(scatter plot)
# 도미
import matplotlib.pyplot as plt

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

plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 도미와 빙어 합치기
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# 사이킷런에서 기대하는 데이터 형태
# [[12, 1], [13, 3], [14, 2], ..., [14, 1]]
# [12, 1] 가 샘플이라고 불림

fish_data = [[l, w] for l, w in zip(length, weight)]
# 1이 도미, 0이 빙어임을 알려줌
# 도미를 찾는 것이기 때문에 도미를 1로 둠
fish_target = [1] * 35 + [0] * 14

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
# 두 데이터를 가지고 머신러닝 모델 kn을 훈련
kn.fit(fish_data, fish_target)
# 정확도 출력
kn.score(fish_data, fish_target)

# 새로운 생선 예측
# 1이 출력되므로 도미
print(kn.predict([[30, 600]]))

kn49 = KNeighborsClassifier(n_neighbors=49)
kn49.fit(fish_data, fish_target)
# 정확도 출력
print(kn49.score(fish_data, fish_target))