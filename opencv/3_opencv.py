# 동영상 출력
import cv2

# 동영상 파일 불러오기
cap = cv2.VideoCapture('video.mp4')

i = 0
# 동영상 파일이 올바르게 열렸는지?
while cap.isOpened():
    # ret : 성공 여부, frame : 받아온 이미지 (프레임)
    ret, frame = cap.read()
    if not ret:
        print('더 이상 가져올 프레임이 없어요')
        break

    cv2.imshow('video', frame)

    if cv2.waitKey(1) == ord('q'):
        print('사용자 입력에 의해 종료합니다')
        break

cap.release()
cv2.destroyAllWindows()