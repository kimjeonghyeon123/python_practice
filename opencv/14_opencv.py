import cv2

cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    #frame_resize = cv2.resize(frame, (400, 500))
    frame_resize = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('video', frame_resize)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()