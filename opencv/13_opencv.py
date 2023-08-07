# 보간법
# 1. cv2.INTER_AREA : 크기 줄일 때 사용
# 2. cv2.INTER_CUBIC : 크기 늘릴 때 사용(속도 느림, 퀄리티 좋음)
# 3. cv2.INTER_LINEAR : 크기 늘릴 때 사용(기본값)
import cv2

img = cv2.imread('img.jpg')

# width, height
#dst = cv2.resize(img, (400, 500))

# x, y 비율 정의
dst1 = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
dst2 = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

cv2.imshow('img', img)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)

cv2.waitKey(0)
cv2.destroyAllWindows()

