# 읽기 옵션
# 1. cv2.IMREAD_COLOR : 컬러 이미지, 투명 영역은 무시 (기본값)
# 2. cv2.IMREAD_GRAYSCALE : 흑백 이미지
# 3. cv2.IMREAD_UNCHANGED : 투명 영역까지 포함

import cv2

img_color = cv2.imread('img.jpg', cv2.IMREAD_COLOR)
img_gray = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)
img_unchanged = cv2.imread('img.jpg', cv2.IMREAD_UNCHANGED)

# 이미지의 세로, 가로, Channel 정보 표시
print(img_color.shape)

cv2.imshow('img_color', img_color)
cv2.imshow('img_gray', img_gray)
cv2.imshow('img_unchanged', img_unchanged)

cv2.waitKey(0)
cv2.destroyAllWindows()