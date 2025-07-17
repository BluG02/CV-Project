#단순 서버 테스트용, 나중에 지워도 됨

from flask import Flask, Response
import cv2
import numpy as np
import ffmpeg

app = Flask(__name__)

ffmpeg_path = "/usr/local/bin/ffmpeg"  # 또는 Windows 경로
rtmp_url = "rtmp://34.47.71.66/live/cam1"
width, height = 640, 360

process = (
    ffmpeg
    .input(rtmp_url, fflags='nobuffer', flags='low_delay', rtbufsize='100M')
    .output('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}')
    .run_async(pipe_stdout=True, cmd=ffmpeg_path)
)

def generate():
    while True:
        in_bytes = process.stdout.read(width * height * 3)
        if not in_bytes:
            break
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/video')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, threaded=True)
