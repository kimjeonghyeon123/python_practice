# import cv2
#
# # 흑백으로 이미지 불러오기
# img = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)
#
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# result = cv2.imwrite('img_save.jpg', img)
# print(result)

import cv2

img = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imwrite('img_save.png', img)