# 마우스 이벤트 등록
import cv2

def mouse_handler(event, x, y, flags, param):
    pass

img = cv2.imread('poker.jpg')

# img 란 이름의 윈도우 먼저 만들어 두는 것
# 여기에 마우스 이벤트를 처리하기 위한 핸들러 적용
cv2.namedWindow('img')

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()