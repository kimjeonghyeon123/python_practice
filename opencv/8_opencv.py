import cv2
import numpy as np

# 세로 가로 채널
img = np.zeros((480, 640, 3), dtype=np.uint8)

COLOR = (0, 255, 0) # BGR 초록색
THICKNESS = 3 # 두께

cv2.rectangle(img, (100, 100), (200, 200), COLOR, THICKNESS)
# 그릴 위치, 왼쪽 위 좌표, 오른쪽 아래 좌표, 색깔, 두께

cv2.rectangle(img, (300, 100), (400, 300), COLOR, cv2.FILLED)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()