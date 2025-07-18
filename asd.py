# 필요한 라이브러리를 가져옵니다.
import torch
import cv2
import numpy as np
import json
from datetime import datetime
import time
import sys
import ffmpeg

# 로컬에 저장된 YOLOv5 모델을 로드합니다.
model = torch.hub.load('C:/Users/ckseh/OneDrive/바탕 화면/공모전/학술제/yolov5-python3.6.9-jetson', 'custom', path='4th.pt', source='local', force_reload=True)

# 분석할 비디오 파일의 경로를 지정합니다.
# video_path = '/home/jetson/yolov5-python3.6.9-jetson/testVideo_AI4th.mp4'
# cap = cv2.VideoCapture(video_path)

# --- RTMP 스트림 설정 ---
FFMPEG_EXE = r"C:\Users\ckseh\OneDrive\바탕 화면\공모전\학술제\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
RTMP_URL = "rtmp://34.47.71.66/live/cam1"
WIDTH, HEIGHT = 640, 360

# 결과를 저장할 JSON 파일 경로
output_json_path = 'test.json'

# --- 좌석 영역 좌표 불러오기 ---
try:
    # polygons_config.py에서 좌표 데이터를 불러옵니다.
    from polygons_config import polygons_detected
    polygons = polygons_detected

    if not polygons:
        print("오류: polygons_config.py가 비어있습니다. coordinate.py를 실행하여 좌석을 설정하세요.")
        sys.exit()

except ImportError:
    print("오류: polygons_config.py를 찾을 수 없습니다. coordinate.py를 실행하여 파일을 생성하세요.")
    sys.exit()

# 비디오의 한 프레임을 처리하여 좌석별 점유 상태를 반환하는 함수
def process_frame(frame, polygons):
    """현재 프레임을 처리하여 각 좌석의 점유 상태를 반환합니다."""
    results = model(frame)
    current_objects = {key: "0" for key in polygons.keys()}

    # 감지 영역(polygons)을 프레임에 그립니다.
    for polygon_name, polygon_coordinates in polygons.items():
        if polygon_coordinates:
            cv2.polylines(frame, [np.array(polygon_coordinates)], True, (0, 255, 0), 2)
            # 좌석 이름 표시
            text_pos = polygon_coordinates[0]
            cv2.putText(frame, polygon_name, (text_pos[0], text_pos[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # YOLOv5 감지 결과를 처리합니다.
    for obj in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = obj
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        detected_class = model.names[int(cls)]

        # 감지된 객체에 바운딩 박스를 그립니다.
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # 'ChairFull' 클래스만 처리
        if detected_class == "ChairFull":
            closest_polygon = None
            min_distance = float('inf')
            for polygon_name, polygon_coordinates in polygons.items():
                if not polygon_coordinates:
                    continue
                
                polygon_np = np.array(polygon_coordinates)
                # 폴리곤의 중심점을 계산합니다.
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
        # print(f"좌석 상태가 {file_path} 파일에 저장되었습니다.") # 너무 자주 출력되므로 주석 처리
    except IOError as e:
        print(f"파일 쓰기 오류: {e}")

# --- 메인 실행 로직 ---
seat_mapping = {name: i for i, name in enumerate(polygons.keys(), 1)}
previous_objects = {}
location_id = 1

# FFmpeg 프로세스 시작
process = (
    ffmpeg
    .input(RTMP_URL, fflags='nobuffer', flags='low_delay', rtbufsize='100M')
    .output('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{WIDTH}x{HEIGHT}')
    .run_async(pipe_stdout=True, cmd=FFMPEG_EXE)
)

print("RTMP 스트림을 받아옵니다. 'q' 키를 누르면 종료됩니다.")

with torch.no_grad():
    while True:
        in_bytes = process.stdout.read(WIDTH * HEIGHT * 3)
        if not in_bytes:
            print("비디오 스트림이 종료되었습니다.")
            break
        
        frame = np.frombuffer(in_bytes, np.uint8).reshape([HEIGHT, WIDTH, 3])

        current_objects = process_frame(frame, polygons)

        if current_objects != previous_objects:
            now = datetime.now()
            
            output_records = []
            for polygon_name, value in current_objects.items():
                record = {
                    "location_id": location_id,
                    "seat_number": seat_mapping[polygon_name],
                    "is_occupied": value == "1",
                    "date": now.strftime('%Y-%m-%d'),
                    "hour": now.hour,
                    "day_of_week": (now.weekday() + 1) % 7, # 월요일=1, 일요일=7
                    "timestamp": now.isoformat()
                }
                output_records.append(record)
            
            save_state_to_json(output_records, output_json_path)
        
        previous_objects = current_objects.copy()

        cv2.imshow('YOLOv5 Real-time Occupancy Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

process.stdout.close()
process.wait()
cv2.destroyAllWindows()