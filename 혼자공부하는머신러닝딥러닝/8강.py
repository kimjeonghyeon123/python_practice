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
'''

import pandas as pd

# csv 데이터를 가져와서 넘파이 배열로 만듦
df = pd.read_csv('https://bit.ly/perch_csv')
perch_full = df.to_numpy()
print(perch_full)

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







