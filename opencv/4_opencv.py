# 카메라 출력

import cv2

# 0번째 카메라 장치 [Device ID]
cap = cv2.VideoCapture(0)

# 카메라가 잘 열리지 않은 경우
if not cap.isOpened():
    # 프로그램 종료
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('camera', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
