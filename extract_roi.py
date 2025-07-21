# click4_roi_live.py ─ 4점 클릭으로 좌석 ROI 지정 (NumPy fix)
import cv2, time, numpy as np      # ← NumPy 추가

### ✂️  필요하면 여기만 고치세요 ──────────────────────────────
RTSP_URL      = "rtsp://sunghyun:sunghyun02@172.25.81.245/stream1"
DISPLAY_SIZE  = (640, 360)
### ──────────────────────────────────────────────────────────

cap = cv2.VideoCapture(RTSP_URL)
assert cap.isOpened(), "❌  스트림 열기 실패"

orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
sx, sy  = orig_w / DISPLAY_SIZE[0], orig_h / DISPLAY_SIZE[1]

rois, pts = [], []

def on_mouse(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))
        if len(pts) == 4:
            poly = [(int(px*sx), int(py*sy)) for px, py in pts]
            idx  = len(rois) + 1
            name = f"Seat_{idx}"
            rois.append((name, poly))
            print(f'"{name}": {poly},')
            pts = []

cv2.namedWindow("Click 4 pts (q:quit, ⌫:undo)")
cv2.setMouseCallback("Click 4 pts (q:quit, ⌫:undo)", on_mouse)

print("▶ 좌석의 4 꼭짓점을 시계(또는 반시계) 방향으로 클릭하세요.")
print("   • 4점 찍으면 자동 저장 + 터미널 출력")
print("   • 잘못 찍었을 때: 백스페이스/Del → undo | 종료: q")

while True:
    ret, frame = cap.read()
    if not ret: continue
    disp = cv2.resize(frame, DISPLAY_SIZE)

    # 완성 ROI
    for _, poly in rois:
        pts_disp = [(int(x/sx), int(y/sy)) for x, y in poly]
        cv2.polylines(disp, [np.array(pts_disp, np.int32)], True, (0,255,0), 2)

    # 진행 중 점·선
    for (px, py) in pts:
        cv2.circle(disp, (px, py), 4, (0,0,255), -1)
    if len(pts) > 1:
        cv2.polylines(disp, [np.array(pts, np.int32)], False, (0,0,255), 1)

    cv2.imshow("Click 4 pts (q:quit, ⌫:undo)", disp)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in (8, 127) and rois:
        rois.pop(); pts=[]
        print("↩️  마지막 ROI 삭제")

cap.release(); cv2.destroyAllWindows()

print("\n=== 최종 ROI 목록 (복사해서 polygons에 붙여넣기) ===")
for name, poly in rois:
    print(f'"{name}": {poly},')
