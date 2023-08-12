import cv2

img_basic = cv2.imread('img.jpg', cv2.IMREAD_COLOR)
cv2.imshow('img_basic', img_basic)
cv2.waitKey(0)

cv2.destroyAllWindows()

img_gray = cv2.cvtColor(img_basic, cv2.COLOR_BGR2GRAY)
cv2.imshow('img_gray', img_gray)
cv2.waitKey(0)