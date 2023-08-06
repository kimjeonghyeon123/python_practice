import cv2

cap = cv2.VideoCapture('video.mp4')

# 코덱 정의
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow('video', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()