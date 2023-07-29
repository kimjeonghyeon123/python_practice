# 도형 그리기

import cv2
import numpy as np

# 세로 480 X 가로 640, 3 Channel [RGB] 에 해당하는 스케치북 만들기
img = np.zeros((480, 640, 3), dtype=np.uint8)

# [세로 영역, 가로 영역] = (B, G, R)
img[100:200, 200:300] = (255, 255, 255)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()