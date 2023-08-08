import cv2
import mediapipe as mp

# 웹캠을 사용할 경우
cap = cv2.VideoCapture(0)

# Face Detection 모듈 초기화
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

while True:
    # 웹캠에서 프레임 가져오기
    success, image = cap.read()

    if not success:
        break

    # 얼굴 감지 수행
    results = face_detection.process(image)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # 얼굴 영역에 박스 그리기
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", image)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()