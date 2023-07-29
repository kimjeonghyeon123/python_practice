# 원

import cv2
import numpy as np

img = np.zeros((480, 640, 3), dtype=np.uint8)

COLOR = (255, 255, 0) # BGR 옥색
RADIUS = 50 # 원 둘레
THICKNESS = 3 # 두께

# 그릴 위치, 원의 중심점, 반지름, 색깔, 두께, 선 종류
# 속이 빈 원
cv2.circle(img, (200, 100), RADIUS, COLOR, THICKNESS, cv2.LINE_AA)

# 속이 꽉 찬 원
cv2.circle(img, (400, 100), RADIUS, COLOR, cv2.FILLED, cv2.LINE_AA)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()