# 이미지 자르기
# 영역을 잘라서 새로운 윈도우 창에 표시

import cv2

img = cv2.imread('img.jpg')

# 세로 기준 100~200
# 가로 기준 300~400
crop = img[100:200, 200:400]
img[100:200, 400:600] = crop

cv2.imshow('img', img)
#cv2.imshow('crop', crop)

cv2.waitKey(0)
cv2.destroyAllWindows()