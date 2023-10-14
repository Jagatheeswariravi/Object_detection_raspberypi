from flask import Flask, render_template, Response, session,redirect,request
import os
import cv2
import math
import time
from ultralytics import YOLO
from flask import jsonify
from pymodbus.client import ModbusTcpClient


app = Flask(__name__)
app.config['SECRET_KEY'] = 'LucasTVS'
app.config['UPLOAD_FOLDER'] = 'static/files'

host =  '169.254.168.95'  
port = 502
client = ModbusTcpClient(host, port)


detected_objects =[]
class_name_dict = {
    0: "QR code scanning",
    1: "spindle screw driver",
    2: "spindle screw passenger",
    3: "Go/No Go RHS",
    4: "Go/No Go LHS"
}

def generate_frames():
    cap = cv2.VideoCapture(0)
    model = YOLO("last.pt")
    classNames = [
        "QR code scanning",
        "spindle screw driver",
        "spindle screw passenger",
        "Go/No Go RHS",
        "Go/No Go LHS"
    ]
    global detected_objects

    while True:
        client.connect()
        ret,img = cap.read()
        results = model.predict(img)
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
                if conf > 0.1:
                    if class_name not in detected_objects:
                        detected_objects.append(class_name)
                temp = client.read_holding_registers(1000,1,unit=1)
                if (temp.registers[0]) == 1 :
                    detected_objects.clear()
        
        yield img
    
    cv2.destroyAllWindows()

def generate_frames_web():
    yolo_output= generate_frames()
    
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')






@app.route("/",methods=["GET","POST"])
def home():
    session.clear()
    return render_template("web.html")

@app.route('/webapp')
def webapp():
    return Response(generate_frames_web(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/section2_data_endpoint', methods=['GET'])
def get_section2_data():
    return render_template('section2.html', class_name_dict=class_name_dict, detected_objects=detected_objects)



if __name__ == "__main__":
    app.run(debug=True)