# 이미지 대칭, 회전
import cv2

img = cv2.imread('img.jpg')
# flipcode > 0 : 좌우 대칭
# flipcode = 0 : 상하 대칭
# flipcode < 0 : 상하좌우 대칭
flip_horizontal = cv2.flip(img, 1)
flip_vertical = cv2.flip(img, 0)
flip_both = cv2.flip(img, -1)

# 회전
rotate_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
rotate_180 = cv2.rotate(img, cv2.ROTATE_180)
rotate_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

cv2.imshow('img', img)
cv2.imshow('flip_horizontal', flip_horizontal)
cv2.imshow('flip_vertical', flip_vertical)
cv2.imshow('flip_both', flip_both)

cv2.imshow('rotate_90', rotate_90)
cv2.imshow('rotate_180', rotate_180)
cv2.imshow('rotate_270', rotate_270)

cv2.waitKey(0)
cv2.destroyAllWindows()