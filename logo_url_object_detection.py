
import time
from flask import Flask, Response
import cv2

app = Flask(__name__)

prototxt = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

classes = {0:"background", 1:"aeroplane", 2:"bicycle",
           3:"bird", 4:"boat", 5:"bottle", 6:"bus",
           7:"car", 8: "cat", 9:"chair", 10:"cow",
           11:"diningtable", 12:"dog", 13: "horse", 14:"motorbike",
           15:"person", 16:"pottedplant", 17:"sheep", 18:"sofa",
           19:"train", 20:"tvmonitor"}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)  # intenta 30 FPS

def generate_frames():
    prev_time = 0
    target_fps = 25  # objetivo fps
    frame_time = 1 / target_fps

    while True:
        time_elapsed = time.time() - prev_time
        if time_elapsed < frame_time:
            # Para no saturar, espera un poco
            time.sleep(frame_time - time_elapsed)

        success, frame = cap.read()
        if not success:
            break

        prev_time = time.time()

        height, width, _ = frame.shape

        # Redimensionar para la red directamente a 300x300 (sin escala extra)
        frame_for_net = cv2.resize(frame, (300, 300))
        blob = cv2.dnn.blobFromImage(frame_for_net, 0.007843, (300, 300), (127.5, 127.5, 127.5))
        net.setInput(blob)
        detections = net.forward()

        # Dibujar cuadros en el frame original (para buena calidad)
        for detection in detections[0][0]:
            confidence = detection[2]
            if confidence > 0.45:
                class_id = int(detection[1])
                label = classes.get(class_id, "unknown")

                box = detection[3:7] * [width, height, width, height]
                x_start, y_start, x_end, y_end = map(int, box)

                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                cv2.putText(frame, label, (x_start, y_start - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.putText(frame, f"Conf: {confidence * 100:.2f}%", (x_start, y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8" />
    <title>Detección de Objetos - Presentación Militar</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: 'Courier New', Courier, monospace;
            background-color: #1b2a0e; /* verde oscuro militar */
            color: #cddc39; /* verde lima */
        }
        .container {
            max-width: 900px;
            margin: 20px auto;
            padding: 25px;
            background-color: #2e3b17; /* verde más oscuro */
            border: 4px solid #4a601d; /* verde oliva fuerte */
            border-radius: 8px;
            box-shadow: 0 0 12px #a5bf47;
            position: relative;
        }
        h1 {
            text-align: center;
            color: #a5bf47;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 3px;
            margin-bottom: 30px;
            text-shadow: 0 0 5px #a5bf47;
        }
        .video-wrapper {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            border: 3px solid #4a601d;
            border-radius: 6px;
            background-color: #182106;
            padding: 10px;
        }
        .video-wrapper img {
            max-width: 100%;
            height: auto;
            border: 2px solid #a5bf47;
            border-radius: 4px;
            box-shadow: 0 0 10px #a5bf47;
        }
        .logo {
            position: absolute;
            top: 15px;
            right: 15px;
            width: 90px;
            opacity: 0.8;
            filter: drop-shadow(0 0 3px #a5bf47);
        }
        @media (max-width: 600px) {
            .logo {
                width: 60px;
            }
        }
        .logo {
            position: absolute;
            top: 15px;
            right: 15px;
            width: 80px; /* 3 veces menos que 90px */
            opacity: 0.8;
            filter: drop-shadow(0 0 3px #a5bf47);
        }
    </style>
</head>
<body>
    <div class="container">
        <img src="/static/fie.png" alt="Descripción de la imagen" class="logo"/>
        <h1>Detección de Objetos en Vivo</h1>
        <div class="video-wrapper">
            <img src="/video" alt="Video en vivo de detección" />
        </div>
    </div>
</body>
</html>"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


