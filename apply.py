from flask import Flask, render_template, Response, session,redirect
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
import cv2
import math
import time
from ultralytics import YOLO
from flask import jsonify
import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" )
app = Flask(__name__)
app.config['SECRET_KEY'] = 'LucasTVS'
app.config['UPLOAD_FOLDER'] = 'static/files'

class_name_dict = {
    0: "QR code scanning",
    1: "spindle screw driver",
    2: "spindle screw passenger",
    3: "Go/No Go RHS",
    4: "Go/No Go LHS"
}
detected_objects = []

def video_detection(path_x):
    video_capture = path_x
    
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    model = YOLO(r"C:\Users\PMTC-ELE\Desktop\Yom_YG8\train\Yom\Yom_predict\best.pt") # device='cpu' /'gpu' / model.to('cuda')
    #model.to('cuda')
    classNames = [
        "QR code scanning",
        "spindle screw driver",
        "spindle screw passenger",
        "Go/No Go RHS",
        "Go/No Go LHS"
    ]

    start_time = time.time()
    end_time = start_time + 50

    while True:
        success, img = cap.read()
        results = model(img,imgsz=160)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
               
                label = f"{class_name}{conf}"
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, [0, 255, 0], -1, cv2.LINE_AA)
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                if conf > 0.5:
                    if class_name not in detected_objects:
                        detected_objects.append(class_name)

        yield img

    cv2.destroyAllWindows()
    #return detected_objects


def generate_frames_web(path_x):
    yolo_output= video_detection(path_x)
    
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('Updated web.html',detected_objects=detected_objects)


@app.route('/webapp')
def webapp():
    
    
    return Response(generate_frames_web(path_x= "rtsp://admin:Lucas123@192.168.3.64:554"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')#rtsp://admin:cctv@123@192.168.1.64:554



@app.route('/refresh',methods=['GET', 'POST'])
def refresh():
    return "hello"
    

@app.route('/section2_data_endpoint', methods=['GET'])
def get_section2_data():
    return render_template('Updated section2.html', class_name_dict=class_name_dict, detected_objects=detected_objects)


if __name__ == "__main__":
    app.run()#debug=True

