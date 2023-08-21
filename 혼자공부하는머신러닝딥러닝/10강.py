'''
1. 확률적 경사 하강법
    1) 데이터가 추가될 때마다 반복 학습
    2) 모델 업데이트 하기
        - 점진적 학습 : 점차 학습하는 것
        - 온라인 학습 : 서비스 도중 학습하는 것
    3) 딥러닝, 머신러닝 알고리즘 훈련방법임
    4) 확률적 경사 하강법
        - 가장 가파른 경사를 조금씩 내려가는 것
        - 훈련세트를 모두 다 사용했다면 1에포크 완료

    훈련세트에서 한개나 여러개씩 꺼내서 반복적으로 훈련한 후 훈련세트를
    다 사용했을 때 1에포크가 완료됐다고 한다

    5) 미니 배치 경사 하강법
        - 훈련세트에서 여러 개씩 꺼내는 것, 2의 배수개씩 꺼냄

    6) 손실함수
        - 나쁜 정도를 측정하는 함수
        - 낮을수록 좋음
        - 미분 가능해야 함
        - 로지스틱 손실 함수(크로스 엔트로피 손실 함수)
'''

import pandas as pd
import numpy as np
fish = pd.read_csv('https://bit.ly/fish_csv_data')
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

# 확률적 경사 하강법 분류 알고리즘
from sklearn.linear_model import SGDClassifier
# 로지스틱 손실 함수 적용
# max_iter = epoch
sc = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

# fit : 새로 학습하는 것
# partial_fit : 기존에 학습했던 것을 유지한채로 다시 학습하는 것
sc.partial_fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

# 과소적합일어남 따라서 epoch를 늘려야 됨
# 규제가 작아지면 과대적합 됨
# 규제가 커지면 둘다 정확성이 낮아짐
# epoch가 높아질수록 과대적합됨 낮으면 과소적합임
# 그래서 그 중간점을 찾아 종료하는 것을 조기종료라고 함

sc = SGDClassifier(loss='log_loss', random_state=42)
train_score = []
test_score = []

classes = np.unique(train_target)
for _ in range(0, 300):
    sc.partial_fit(train_scaled, train_target, classes=classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))

import matplotlib.pyplot as plt

plt.plot(np.arange(300), train_score)
plt.plot(np.arange(300), test_score)
plt.show()

sc = SGDClassifier(loss='log_loss', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))








