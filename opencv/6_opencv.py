# 직선
# 1. cv2.LINE_4 : 상하좌우 4 방향으로 연결된 선
# 2. cv2.LINE_8 : 대각선을 포함한 8방향으로 연결된 선 (기본값)
# 3. cv2.LINE_AA : 부드러운 선 (anti-aliasing)

import cv2
import numpy as np

img = np.zeros((480, 640, 3), dtype=np.uint8)

COLOR = (0, 255, 255) # BGR : Yellow
THICKNESS = 3 # 두께

# (x1, y1)부터 (x2, y2) 까지
# 그릴 위치, 시작 점, 끝 점, 색깔, 두께, 선 종류
cv2.line(img, (50, 100), (400, 50), COLOR, THICKNESS, cv2.LINE_8)
cv2.line(img, (50, 200), (400, 150), COLOR, THICKNESS, cv2.LINE_4)
cv2.line(img, (50, 300), (400, 250), COLOR, THICKNESS, cv2.LINE_AA)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()