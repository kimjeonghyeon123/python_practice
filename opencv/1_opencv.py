import cv2

# 해당 경로의 파일 읽어오기
img = cv2.imread('img.jpg')

# img 라는 이름의 창에 img 를 표시
cv2.imshow('img', img)

# 지정된 시간 동안 사용자 키 입력 대기
# 0 : 무한정 대기
cv2.waitKey(0)

# 모든 창 닫기
cv2.destroyAllWindows()

