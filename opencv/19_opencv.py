# 이미지 변형 (원근)
import cv2
import numpy as np

img = cv2.imread('newspaper.jpg')

width, height = 640, 240

# Input 4개 지점
src = np.array([[511, 352], [1008, 345], [1122, 584], [455, 594]], dtype=np.float32)
# Output 4개 지점
dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

# 변환 행렬 얻어옴
matrix = cv2.getPerspectiveTransform(src, dst)
# 변환한 결과를 얻어옴
result = cv2.warpPerspective(img, matrix, (width, height))

cv2.imshow('img', img)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()