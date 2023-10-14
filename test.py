from flask import Flask, render_template, Response, session,redirect
from werkzeug.utils import secure_filename
import os
import cv2
import math
import time
from ultralytics import YOLO
from flask import jsonify
import torch
import webview

app = Flask(__name__,template_folder="./templates")
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
    model = YOLO("C:\\Users\\PMTC-ELE\\Desktop\\jaga_files\\interview\\best.pt")
    classNames = [
        "QR code scanning",
        "spindle screw driver",
        "spindle screw passenger",
        "Go/No Go RHS",
        "Go/No Go LHS"
    ]

    
    while True:
        success, img = cap.read()
        results = model.predict(img,imgsz=320,device="cpu")
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
                cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                if conf > 0.2:
                    if class_name not in detected_objects:
                        detected_objects.append(class_name)
                    

        yield img

    cap.release()
    cv2.destroyAllWindows()

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
    return render_template('web.html',detected_objects=detected_objects)


@app.route('/webapp')
def webapp():
    
    return Response(generate_frames_web(path_x=0),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/section2_data_endpoint', methods=['GET'])
def get_section2_data():
    data = render_template('section2.html', class_name_dict=class_name_dict, detected_objects=detected_objects)
    detected_objects.clear()
    return data





#webview.create_window("test",app)
   

if __name__ == "__main__":
    app.run(debug=True)
    #webview.start()

