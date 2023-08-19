'''
무게 = a * 길이 + b
    - b : 계수, 가중치

1. 다중 회귀
    - multiple regression
    - 여러 개의 변수를 사용하는 회귀 (x, y, z, ...)

2. 특성 공학
    - 길이 높이 두께 가 있을 때 길이와 높이의 조합을 하는 것

3. 판다스로 데이터 준비 (데이터 프레임)
    - 엑셀 형식
    - 많이 쓰이고 널리 쓰임
    - csv 파일
    - 행 : 샘플
    - 열 : 특성

특성과의 조합으로 특성 개수를 늘려 테스트가 더 적합하도록 만들어줌
훈련 점수보다 테스트 점수가 더 높은 현상인 과소적합을 해결할 수 있음
특성이 많아질수록 훈련 데이터 점수가 올라감
하지만 너무 많은 특성을 만들경우 과대적합이 만들어질 확률이 높음
'''
import numpy as np
import pandas as pd

# csv 데이터를 가져와서 넘파이 배열로 만듦
# length, height, width
df = pd.read_csv('https://bit.ly/perch_csv')
perch_full = df.to_numpy()

perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
     1000.0, 1000.0]
     )

# 특성끼리 서로 곱하는 것
# PolynomialFeatures : 변환기, Transformer
#   - 메서드 : fit, transform
#       - 실제로 학습하는 것은 아님
# LinearRegression   : 추정기, Estimator
#   - 메서드 : fit -> predict -> score
from sklearn.preprocessing import PolynomialFeatures

# degree=2
# 제곱 함 디그리가 3이면 3제곱
poly = PolynomialFeatures()
poly.fit([[2, 3]])

# 1(bias), 2, 3, 2**2, 2*3, 3**2
print(poly.transform([[2, 3]]))

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    perch_full, perch_weight, random_state=42
)

poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
# (42, 9) 9개의 특성
print(train_poly.shape)
print(poly.get_feature_names_out())

test_poly = poly.transform(test_input)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))

# 더 많은 특성 만들기
poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

# (42, 55) 55개의 특성
print(train_poly.shape)

lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))

# 규제 : 과대적합된 모델을 완화하는 것
#       - 가중치(기울기)의 값을 작게 만드는 것
# 규제 전에 표준화 작업
# mean, std 구해서 작업한 것과 같음
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# 릿지 회귀 : 일반적으로 랏소보다 선호함, 규제가 더 잘됨
# - 가중치^2 : 모델 파라미터(모델이 알아내는 값)
# - L2 규제
# 기본 : Ridge(alpha = 1) alpha : 하이퍼파라미터(우리가 정해주는 값)
# alpha값이 커지면 강도가 세짐
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

# 적절한 규제 강도 찾기
# 일반적으로 10배씩 차이나게 함
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
train_score = []
test_score = []
for alpha in alpha_list:
    ridge = Ridge(alpha=alpha)
    ridge.fit(train_scaled, train_target)
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))

import matplotlib.pyplot as plt
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)

print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

# 라쏘 회귀
# - 가중치에 절대값을 씌움
# - L1 규제

from sklearn.linear_model import Lasso

lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))

lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))

# 기울기가 0인 것을 찾아 True로 만들고 그것의 개수를 찾음
# 사용하지 않는 특성의 개수를 의미함
# 55개 중 40개는 사용하지 않음
print(np.sum(lasso.coef_ == 0))