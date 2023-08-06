import cv2
import numpy as np

img = np.zeros((480, 640, 3), dtype=np.uint8)

COLOR = (0, 0, 255) # BGR 빨간색
THICKNESS = 3 # 두께

pts1 = np.array([[100, 100], [200, 100], [100, 200]])
pts2 = np.array([[200, 100], [300, 100], [300, 200]])

# cv2.polylines(img, [pts1], True, COLOR, THICKNESS, cv2.LINE_AA)
# cv2.polylines(img, [pts2], True, COLOR, THICKNESS, cv2.LINE_AA)
cv2.polylines(img, [pts1, pts2], True, COLOR, THICKNESS, cv2.LINE_AA)
# 그릴 위치, 그릴 좌표들, 닫힘 여부, 색깔, 두께, 선 종류

pts3 = np.array([[[100, 300], [200, 300], [100, 400]], [[200, 300], [300, 300], [300, 400]]])
cv2.fillPoly(img, pts3, COLOR, cv2.LINE_AA)
# 그릴 위치, 그릴 좌표들, 색깔, 선 종류

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()