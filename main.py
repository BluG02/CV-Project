import time, threading, queue
from datetime import datetime
import torch, cv2, numpy as np
import json   # ← TXT 대신 JSON으로 저장하면 파싱이 편리합니다

# ───────────── 설정 ─────────────
# DISPLAY_SIZE   = (640, 360)
DISPLAY_SIZE=(1280, 720)
INFER_INTERVAL = 1.0           # 1 초마다 추론
MODEL_INPUT    = 640           # YOLOv5 입력 해상도
ON_DELAY_SEC   = 10.0          # 10 초 연속 감지 → 점유 ON
OFF_DELAY_SEC  = 2.0           # 2 초 연속 미감지 → 점유 OFF
LOG_PATH       = "occupancy_log.csv"         # 세션별 착‧퇴석 기록
SNAPSHOT_PATH  = "yolo_result.txt"           # 실시간 상태 + 전체 사람 수

# “사람”으로 셀 클래스 이름 목록 -- 모델 학습 시 이름과 맞춰 주세요
PERSON_CLASSES = {"person", "ChairFull", "StandingPerson", "WalkingPerson"}

# ────────── 모델 로드 ──────────
model = torch.hub.load(
    'C:/Users/User/Desktop/github프로젝트/yolov5',
    'custom',
    path='4th.pt',
    source='local'
)
if torch.cuda.is_available():
    model.to('cuda').half()
model.eval()

# ──────── 카메라 스레드 ────────
rtsp_url = "rtsp://sunghyun:sunghyun02@172.25.81.245/stream1"
cap = cv2.VideoCapture(rtsp_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_q = queue.Queue(maxsize=1)
def reader():
    while True:
        ret, f = cap.read()
        if not ret:
            break
        if frame_q.full():
            try:
                frame_q.get_nowait()
            except queue.Empty:
                pass
        frame_q.put(f)
threading.Thread(target=reader, daemon=True).start()

# ────────── ROI 정의 ──────────
polygons = {
    # ───── Left 열 ─────
    "Left_1":  [(3, 906),  (96, 927),  (171, 783), (81, 774)],
    "Left_2":  [(78, 768), (180, 780), (264, 642), (183, 636)],
    "Left_3":  [(198, 624), (288, 630), (378, 507), (291, 504)],
    "Left_4":  [(294, 501), (384, 507), (444, 432), (372, 429)],
    "Left_5":  [(375, 420), (453, 420), (486, 384), (426, 381)],
    "Left_6":  [(441, 372), (495, 384), (540, 330), (480, 324)],
    "Left_7":  [(492, 309), (546, 321), (579, 291), (522, 276)],
    "Left_8":  [(528, 270), (573, 291), (618, 246), (558, 243)],

    # ───── Center-Right 열 ─────
    "center_right_1": [(1038, 705), (1290, 678), (1350, 879), (1059, 921)],
    "center_right_2": [(1014, 561), (1242, 537), (1281, 666), (1035, 687)],
    "center_right_3": [(1011, 486), (1206, 477), (1218, 525), (1014, 552)],
    "center_right_4": [(1002, 441), (1167, 420), (1194, 468), (1011, 474)],
    "center_right_5": [(996, 375), (1005, 429), (1173, 408), (1143, 366)],
    "center_right_6": [(993, 303), (999, 360), (1152, 354), (1119, 300)],
    "center_right_7": [(999, 246), (990, 291), (1125, 291), (1095, 225)],

    # ───── Center-Left 열 ─────
    "center_left_1": [(786, 738), (1023, 726), (1050, 924), (759, 936)],
    "center_left_2": [(810, 618), (786, 726), (1020, 720), (1005, 603)],
    "center_left_3": [(825, 528), (993, 519), (1005, 591), (810, 603)],
    "center_left_4": [(843, 459), (990, 450), (993, 507), (831, 510)],
    "center_left_5": [(855, 396), (984, 387), (987, 438), (846, 444)],
    "center_left_6": [(861, 336), (981, 324), (981, 372), (861, 390)],
    "center_left_7": [(873, 285), (978, 276), (981, 315), (867, 327)],

    # ───── Right 열 ─────
    "Right_1": [(1815, 468), (1890, 558), (1911, 552), (1914, 456)],
    "Right_2": [(1662, 345), (1731, 414), (1815, 390), (1761, 336)],
    "Right_3": [(1596, 297), (1644, 339), (1743, 321), (1692, 288)],
    "Right_4": [(1524, 255), (1566, 285), (1665, 276), (1626, 246)],
    "Right_5": [(1482, 207), (1521, 252), (1611, 237), (1560, 204)],
}

# ──── ROI별 상태(타이머 + 시각) ────
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

states = {
    roi: {
        "occupied":   False,
        "accum_on":   0.0,
        "accum_off":  0.0,
        "last_ts":    time.time(),
        "first_hit":  None,   # 착석 최초 감지
        "last_seen":  None,   # 마지막 감지
        "start_time": None,   # 확정된 착석 시각
        "end_time":   None    # 확정된 이탈 시각
    } for roi in polygons
}

# ────────── 보조 함수 ──────────
def pick_roi_name(cx, cy, cls_name):
    if cls_name == "ChairFull":                 # 사람(착석) 클래스
        best, dmin = None, float('inf')
        for name, pts in polygons.items():
            for px, py in pts:
                d = (cx - px) ** 2 + (cy - py) ** 2
                if d < dmin:
                    best, dmin = name, d
        return best
    else:                                       # 사물
        for name, pts in polygons.items():
            if cv2.pointPolygonTest(np.array(pts, np.int32),
                                    (int(cx), int(cy)), False) >= 0:
                return name
    return None

def annotate(frame, results):
    img = frame.copy()
    for name, pts in polygons.items():
        color = (0, 0, 255) if states[name]["occupied"] else (0, 255, 0)
        cv2.polylines(img, [np.array(pts, np.int32)], True, color, 2)
    for *box, _, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        cls_name = model.names[int(cls)] if hasattr(model, "names") else model.module.names[int(cls)]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, cls_name, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return img

# ─────────── 메인 루프 ───────────
last_infer_t = time.time() - INFER_INTERVAL
last_annot   = None

with torch.no_grad():
    while True:
        # ① 프레임 받기
        try:
            frame = frame_q.get(timeout=1)
        except queue.Empty:
            print("⚠️  캡처 실패"); break

        # ② 추론 (1 초 간격)
        if time.time() - last_infer_t >= INFER_INTERVAL:
            results = model(frame, size=MODEL_INPUT)
            now = time.time()

            # ―― 이번 프레임에서 감지된 ROI / 사람 수 ――
            hits = set()
            people_count = 0

            for *box, conf, cls in results.xyxy[0]:
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                cls_name = model.names[int(cls)] if hasattr(model, "names") else model.module.names[int(cls)]

                # 전체 사람 수 카운트
                if cls_name in PERSON_CLASSES:
                    people_count += 1

                # ROI 매핑(착석만)
                roi = pick_roi_name(cx, cy, cls_name)
                if roi:
                    hits.add(roi)

            # ③ 타이머·시각 업데이트
            for name, st in states.items():
                elapsed = now - st["last_ts"]

                if name in hits:                      # 감지 O
                    if st["accum_on"] == 0:           # 처음 들어온 프레임
                        st["first_hit"] = now_str()
                    st["accum_on"]  += elapsed
                    st["accum_off"]  = 0.0
                    st["last_seen"]  = now_str()
                else:                                 # 감지 X
                    if st["accum_off"] == 0:
                        st["last_seen"] = now_str()   # 마지막 보인 시각
                    st["accum_off"] += elapsed
                    st["accum_on"]   = 0.0
                st["last_ts"] = now

                # ④ 상태 전환
                if (not st["occupied"]) and (st["accum_on"] >= ON_DELAY_SEC):
                    st["occupied"]   = True
                    st["start_time"] = st["first_hit"]  # 실제 착석 시각
                    st["end_time"]   = None
                elif st["occupied"] and (st["accum_off"] >= OFF_DELAY_SEC):
                    st["occupied"] = False
                    st["end_time"]  = st["last_seen"]   # 실제 이탈 시각

                    # ▶︎ 파일에 세션 기록 (append)
                    with open(LOG_PATH, "a", encoding="utf-8") as log:
                        log.write(f"{name},{st['start_time']},{st['end_time']}\n")

            # ⑤ 스냅샷 저장(JSON, 덮어쓰기)
            snapshot = {
                k: [
                    "1" if st["occupied"] else "0",
                    st["start_time"] or "",
                    st["end_time"]   or ""
                ] for k, st in states.items()
            }
            snapshot["_people_count"] = str(people_count)   # ← 추가 필드
            with open(SNAPSHOT_PATH, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False)

            last_annot   = annotate(frame, results)
            last_infer_t = now

        # ⑥ 화면 표시
        show = last_annot if last_annot is not None else frame
        cv2.imshow("YOLOv5 Occupancy Demo", cv2.resize(show, DISPLAY_SIZE))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
