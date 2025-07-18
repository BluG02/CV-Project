#휴대폰 카메라 - 서버 연결 / rtmp형식 -> ffmpeg으로 처리 -> cv2 
import ffmpeg
import numpy as np
import cv2


# FFmpeg 실행 파일 경로
ffmpeg_exe = r"C:\Users\ckseh\OneDrive\바탕 화면\공모전\학술제\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"

# RTMP 스트림 주소 (Streamlabs 앱에서 송출 중)
rtmp_url = "rtmp://34.47.71.66/live/cam1"

# 예상 해상도 (앱 해상도에 따라 조정 가능)
width, height = 640, 360

# FFmpeg 파이프로 스트림 수신
# process = (
#     ffmpeg
#     .input(rtmp_url)
#     .output('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}')
#     .run_async(pipe_stdout=True, cmd=ffmpeg_path)
# )
process = (
    ffmpeg
    .input(rtmp_url, fflags='nobuffer', flags='low_delay', rtbufsize='100M')
    .output('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}')
    .run_async(pipe_stdout=True, cmd= ffmpeg_exe)
)


# 프레임 반복 수신
while True:
    in_bytes = process.stdout.read(width * height * 3)
    if not in_bytes:
        print("🚫 스트림 수신 실패")
        break

    # 바이트 → numpy 배열 → OpenCV 이미지
    frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
    cv2.imshow("📡 RTMP 실시간 스트림", frame)

    # 'q' 키로 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

process.stdout.close()
cv2.destroyAllWindows()
