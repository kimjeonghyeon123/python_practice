import cv2
import numpy as np

# 웹캠을 사용할 경우
cap = cv2.VideoCapture(0)

# 눈 검출기 초기화
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    # 웹캠에서 프레임 가져오기
    success, image = cap.read()

    if not success:
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 눈 검출 수행
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (ex, ey, ew, eh) in eyes:
        eye_center = (ex + ew // 2, ey + eh // 2)
        cv2.circle(image, eye_center, 10, (0, 255, 0), -1)

    cv2.imshow("Eye Detection", image)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
