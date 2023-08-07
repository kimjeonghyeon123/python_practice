# 이미지 변형(흑백)

# import cv2
# 이미지를 흑백으로 읽음
# img = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)
#
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2

img = cv2.imread('img.jpg')
# 불러온 이미지를 흑백으로 변경
dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('img', img)
cv2.imshow('dst', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()