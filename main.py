from ultralytics import YOLO
import cv2
import time

# YOLOv8 모델 로드
model = YOLO('yolo11n.pt')  # YOLOv8n 모델 사용

cap = cv2.VideoCapture(0)  # 웹캠 연결

# 10초 간격으로 결과를 표시하기 위한 변수
last_shown_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8을 사용하여 이미지에 대한 예측 수행
    results = model(frame)  # 모델을 호출하여 예측

    # 현재 시간과 마지막으로 결과를 표시한 시간 차이를 계산
    current_time = time.time()
    if current_time - last_shown_time >= 10:  # 10초마다 결과 표시
        results[0].show()  # 첫 번째 결과에 대해 show 호출
        last_shown_time = current_time  # 마지막 결과 표시 시간 갱신

    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()