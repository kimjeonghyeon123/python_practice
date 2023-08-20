import pandas as pd
import numpy as np
fish = pd.read_csv('https://bit.ly/fish_csv_data')
#print(fish.head()) # 위 5개 읽어옴

# 원하는 특성을 리스트로 만들어서 넣으면 됨
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42
)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

from sklearn.neighbors import KNeighborsClassifier

# n_neighbors 기본값 = 5, 가장 가까운 점 5개
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)

# '_' 가 붙어있는 값은 모델이 학습한 값
# 모든 target 값을 알려줌, 알파벳 순서
print(kn.classes_)
# 5개 샘플 뽑아서 테스트 함
print(kn.predict(test_scaled[:5]))

# 확률 출력
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))

# 로지스틱 회귀(분류 알고리즘)
# z = a*무게 + b*길이 + c*대각석 + d*높이 + e*두께 + f
# z범위 : -무한대~+무한대
# z범위를 0~1로 만들어주는 함수 : 시그모이드 함수(=로지스틱 함수)
# ∮ = 1 / (1 + e^(-z)) , 0~1 사이
# predict : z 값만 보고 판단
# predict_proba : ∮ 값 보고 판단

# 로지스틱 회귀(이진 분류)
# Bream, Smelt 만 True, 나머지는 False로 된 배열
# 알파벳 순서로 타겟 데이터가 bream이 0 smelt가 1 이 됨
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

print(lr.predict(train_bream_smelt[:5]))
print(lr.predict_proba(train_bream_smelt[:5]))

# 로지스틱 회귀 계수 확인하기
print(lr.coef_, lr.intercept_)

decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions) # z 값

# 시그마 값 계산해줌
from scipy.special import expit
print(expit(decisions))

# 로지스틱 회귀(다중 분류)
# max_iter(반복횟수) 기본값 = 100
# C 값이 올라가면 규제가 약해짐(alpha값과 반대)
# C 기본값 = 1
# 규제가 약해짐 = 좀 더 복잡한 모델을 만드는 것
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))

print(lr.coef_.shape, lr.intercept_.shape)
# (7,5) (7,)
# 7 : 클래스 개수, 5 : 특성 개수
# 클래스마다 z값이 만들어짐
# 하나의 클래스를 학습할 때 이진분류와 같은 방법으로 학습한다.
# 양성 클래스 하나, 나머지는 음성 클래스로 두고 학습
# 이런 방법으로 7번 반복함

# 소프트맥스 함수
# e_sum = e^(z0) + e^(z1) + e^(z2) +...+ e^(z6)
# s0 = e^(z0)/e_sum, s1 = e^(z1)/e_sum ...
# s1 + s2 + ... + s6 = 1
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))

from scipy.special import softmax

proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))