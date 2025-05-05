#============= PROGRAMA PARA DETECTAR OBJETOS CON UNA CÁMARA VERSION 1.0 =================
#Autor: MARCELO ANDRÉS ACUÑA
#AÑO: 2024

from flask import Flask, Response
import cv2

app = Flask(__name__)

# Cargar el modelo
prototxt = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

classes = {0:"background", 1:"aeroplane", 2:"bicycle",
           3:"bird", 4:"boat", 5:"bottle", 6:"bus",
           7:"car", 8: "cat", 9:"chair", 10:"cow",
           11:"diningtable", 12:"dog", 13: "horse", 14:"motorbike",
           15:"person", 16:"pottedplant", 17:"sheep", 18:"sofa",
           19:"train", 20:"tvmonitor"}

cap = cv2.VideoCapture(0) # Cambia el 0 por la URL de tu cámara IP si es necesario(en este caso por 1)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        height, width, _ = frame.shape
        frame_resized = cv2.resize(frame, (800,800))
        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5))
        net.setInput(blob)
        detections = net.forward()

        for detection in detections[0][0]:
            if detection[2] > 0.45:
                label = classes[int(detection[1])]
                box = detection[3:7]*[width, height, width, height]
                x_start, y_start, x_end, y_end = map(int, box)

                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                cv2.putText(frame, label, (x_start, y_start - 25), 1, 1.5, (0, 255, 255), 2)
                cv2.putText(frame,"Conf: {:.2f}%".format(detection[2]*100), (x_start, y_start - 5),1, 1.2, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "Ir a /video para ver el stream"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
