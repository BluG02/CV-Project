# 필요한 라이브러리를 가져옵니다.
import torch  # 파이토치: 딥러닝 프레임워크
import cv2  # OpenCV: 컴퓨터 비전 라이브러리
import numpy as np  # NumPy: 다차원 배열 및 행렬 연산을 위한 라이브러리
import json  # JSON 처리를 위한 라이브러리
from datetime import datetime  # 날짜 및 시간 관련 라이브러리
import time

# 로컬에 저장된 YOLOv5 모델을 로드합니다.
model = torch.hub.load('C:/Users/ckseh/OneDrive/바탕 화면/공모전/학술제/yolov5-python3.6.9-jetson', 'custom', path='4th.pt', source='local', force_reload=True)

# 분석할 비디오 파일의 경로를 지정합니다.
video_path = '/home/jetson/yolov5-python3.6.9-jetson/testVideo_AI4th.mp4'
cap = cv2.VideoCapture(video_path)

# 결과를 저장할 JSON 파일 경로
output_json_path = 'test.json'

# 비디오의 한 프레임을 처리하여 좌석별 점유 상태를 반환하는 함수
def process_frame(frame, polygons):
    """현재 프레임을 처리하여 각 좌석의 점유 상태를 반환합니다."""
    results = model(frame)
    current_objects = {key: "0" for key in polygons.keys()}

    for polygon_name, polygon_coordinates in polygons.items():
        if polygon_coordinates and polygon_coordinates[0] != (0, 0):
            cv2.polylines(frame, [np.array(polygon_coordinates)], True, (0, 255, 0), 2)

    for obj in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = obj
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center_x = torch.as_tensor(center_x).cpu().item()
        center_y = torch.as_tensor(center_y).cpu().item()
        detected_class = model.names[int(cls)]

        label = f'Class: {detected_class}'
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        if detected_class == "ChairFull":
            closest_polygon = None
            min_distance = float('inf')
            for polygon_name, polygon_coordinates in polygons.items():
                if not polygon_coordinates or polygon_coordinates[0] == (0, 0):
                    continue
                
                polygon_np = np.array(polygon_coordinates)
                poly_center_x, poly_center_y = np.mean(polygon_np, axis=0)
                distance = np.linalg.norm(np.array([center_x, center_y]) - np.array([poly_center_x, poly_center_y]))

                if distance < min_distance:
                    min_distance = distance
                    closest_polygon = polygon_name
            
            if closest_polygon:
                current_objects[closest_polygon] = "1"
    
    return current_objects

# 좌석 상태를 JSON 파일로 저장하는 함수
def save_state_to_json(state_data, file_path):
    """주어진 데이터를 JSON 파일로 저장합니다."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, ensure_ascii=False, indent=4)
        print(f"좌석 상태가 {file_path} 파일에 저장되었습니다.")
    except IOError as e:
        print(f"파일 쓰기 오류: {e}")

# 감지할 좌석 및 감지하지 않을 좌석 영역 정의
polygons_detected = {
    "left_3": [(590, 230), (610, 220), (630, 220), (620, 280)], "left_4": [(570, 190), (600, 190), (620, 220), (590, 230)],
    "left_5": [(530, 150), (560, 150), (570, 170), (550, 170)], "left_6": [(510, 120), (540, 120), (560, 140), (530, 150)],
    "left_7": [(480, 100), (510, 100), (520, 110), (500, 110)], "left_8": [(470, 80), (490, 80), (510, 100), (480, 100)],
    "center_right_1": [(220, 240), (300, 240), (300, 315), (220, 315)], "center_right_2": [(230, 200), (300, 200), (300, 240), (230, 240)],
    "center_right_3": [(240, 170), (300, 170), (300, 200), (240, 200)], "center_right_4": [(240, 130), (300, 130), (300, 170), (240, 170)],
    "center_right_5": [(250, 110), (300, 110), (300, 130), (250, 130)], "center_right_6": [(260, 90), (300, 90), (300, 110), (260, 110)],
    "center_right_7": [(270, 70), (300, 70), (300, 90), (260, 90)], "center_left_1": [(310, 240), (400, 240), (400, 310), (310, 310)],
    "center_left_2": [(310, 200), (390, 200), (390, 240), (310, 240)], "center_left_3": [(310, 170), (380, 170), (380, 200), (310, 200)],
    "center_left_4": [(310, 130), (380, 130), (380, 170), (310, 170)], "center_left_5": [(310, 110), (370, 110), (370, 130), (310, 130)],
    "center_left_6": [(310, 90), (360, 90), (360, 110), (310, 110)], "center_left_7": [(310, 70), (360, 70), (360, 90), (310, 90)],
    "right_3": [(10, 220), (40, 230), (10, 280), (0, 250)], "right_4": [(50, 170), (70, 180), (40, 230), (10, 220)],
    "right_5": [(80, 140), (100, 150), (70, 180), (50, 170)], "right_6": [(100, 110), (120, 120), (100, 150), (80, 140)],
    "right_7": [(110, 100), (130, 110), (110, 120), (100, 110)], "right_8": [(140, 80), (150, 90), (140, 110), (120, 100)],
    "right_9": [(160, 60), (170, 70), (150, 90), (140, 80)], "right_10": [(180, 40), (190, 50), (170, 70), (160, 60)],
}

polygons = {
    "left_1": [(0, 0)], "left_2": [(0, 0)],
    **polygons_detected,
    "right_1": [(0, 0)], "right_2": [(0, 0)],
    "right_11": [(0, 0)], "right_12": [(0, 0)],
}

seat_mapping = {name: i for i, name in enumerate(polygons.keys(), 1)}
previous_objects = {}
location_id = 1

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("비디오의 끝에 도달했거나 읽을 수 없습니다.")
            break

        current_objects = process_frame(frame, polygons)

        if current_objects != previous_objects:
            now = datetime.now()
            
            # JSON으로 출력할 데이터 구조 생성 (occupancy_logs 테이블 형식에 맞게)
            output_records = []
            for polygon_name, value in current_objects.items():
                record = {
                    "location_id": location_id,
                    "seat_number": seat_mapping[polygon_name],
                    "is_occupied": value == "1",
                    "date": now.strftime('%Y-%m-%d'),
                    "hour": now.hour,
                    "day_of_week": (now.weekday() + 1) % 7,
                    "timestamp": now.isoformat()
                }
                output_records.append(record)
            
            # 변경된 상태를 JSON 파일로 저장
            save_state_to_json(output_records, output_json_path)
        
        previous_objects = current_objects.copy()

        cv2.imshow('YOLOv5 Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # time.sleep(0.1)

cap.release()
cv2.destroyAllWindows()
