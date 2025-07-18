

import cv2
import numpy as np
import json
import ffmpeg

# --- 설정 변수 ---
# Connect_cam.py에서 가져온 설정
FFMPEG_EXE = r"C:\Users\ckseh\OneDrive\바탕 화면\공모전\학술제\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
RTMP_URL = "rtmp://34.47.71.66/live/cam1"
WIDTH, HEIGHT = 640, 360

# 저장할 JSON 파일 경로
JSON_FILE_PATH = "C:/Users/ckseh/OneDrive/바탕 화면/공모전/학술제/CV-Project/polygons.json"

# 좌표를 지정할 좌석 이름 리스트 (순서대로 진행)
SEAT_NAMES = [
    "left_1", "left_2", "left_3", "left_4", "left_5", "left_6", "left_7", "left_8",
    "center_left_1", "center_left_2", "center_left_3", "center_left_4", "center_left_5", "center_left_6", "center_left_7",
    "center_right_1", "center_right_2", "center_right_3", "center_right_4", "center_right_5", "center_right_6", "center_right_7",
    "right_1", "right_2", "right_3", "right_4", "right_5", "right_6", "right_7", "right_8", "right_9", "right_10", "right_11", "right_12"
]

# --- 전역 변수 ---
drawing = False
ref_point = []
polygons_data = {}
current_seat_index = 0
frame = None # 실시간 프레임을 저장할 변수

def select_area_callback(event, x, y, flags, param):
    global ref_point, drawing, frame, polygons_data, current_seat_index

    if current_seat_index >= len(SEAT_NAMES):
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        drawing = True

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        drawing = False

        x1, y1 = ref_point[0]
        x2, y2 = ref_point[1]
        start_x, start_y = min(x1, x2), min(y1, y2)
        end_x, end_y = max(x1, x2), max(y1, y2)

        four_points = [
            (start_x, start_y), (end_x, start_y),
            (end_x, end_y), (start_x, end_y)
        ]

        seat_name = SEAT_NAMES[current_seat_index]
        polygons_data[seat_name] = four_points
        print(f"[저장됨] {seat_name}: {four_points}")

        # 다음 좌석으로 이동
        current_seat_index += 1

def save_polygons_to_json(data, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"\n✅ 좌표 데이터가 {file_path} 파일에 성공적으로 저장되었습니다.")
    except IOError as e:
        print(f"❌ 파일 쓰기 오류: {e}")

# --- 메인 실행 코드 ---
# FFmpeg 프로세스 시작
process = (
    ffmpeg
    .input(RTMP_URL, fflags='nobuffer', flags='low_delay', rtbufsize='100M')
    .output('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{WIDTH}x{HEIGHT}')
    .run_async(pipe_stdout=True, cmd=FFMPEG_EXE)
)

cv2.namedWindow("Coordinate Setup")
cv2.setMouseCallback("Coordinate Setup", select_area_callback)

print("RTMP 스트림을 불러오는 중입니다...")

while True:
    in_bytes = process.stdout.read(WIDTH * HEIGHT * 3)
    if not in_bytes:
        print("🚫 스트림 수신 실패. 프로그램을 종료합니다.")
        break

    # 바이트를 이미지 프레임으로 변환
    frame = np.frombuffer(in_bytes, np.uint8).reshape([HEIGHT, WIDTH, 3])
    display_frame = frame.copy()

    # --- 화면에 안내 문구 및 도형 표시 ---
    # 현재 설정할 좌석 이름 표시
    if current_seat_index < len(SEAT_NAMES):
        seat_to_set = SEAT_NAMES[current_seat_index]
        msg = f"Set coordinates for: {seat_to_set}. Drag the area."
        cv2.putText(display_frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        msg = "All seats are set. Press 's' to save and 'q' to quit."
        cv2.putText(display_frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 이미 설정된 좌표들을 화면에 표시
    for name, points in polygons_data.items():
        cv2.polylines(display_frame, [np.array(points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(display_frame, name, (points[0][0], points[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow("Coordinate Setup", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        save_polygons_to_json(polygons_data, JSON_FILE_PATH)
    elif key == ord('q'):
        break
    elif key == ord('r'): # 'r' 키로 마지막 좌표 리셋
        if polygons_data:
            last_seat_name = SEAT_NAMES[current_seat_index - 1]
            del polygons_data[last_seat_name]
            current_seat_index -= 1
            print(f"[리셋] {last_seat_name}의 좌표를 다시 설정하세요.")

process.stdout.close()
process.wait()
cv2.destroyAllWindows()
    
print("\n프로그램이 종료되었습니다.")
