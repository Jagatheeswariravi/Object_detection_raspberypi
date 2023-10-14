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
from postprocessing import *

app = Flask(__name__)
app.config['SECRET_KEY'] = 'LucasTVS'
app.config['UPLOAD_FOLDER'] = 'static/files'

classNames = {
    0: "QR code scanning",
    1: "spindle screw driver",
    2: "spindle screw passenger",
    3: "Go/No Go RHS",
    4: "Go/No Go LHS"
}

detected_objects1=[]

def generate_frames_web():
    model = YOLO("last.pt")
    while True:
        video = cv2.VideoCapture(0)
        ret, frame = video.read()
        if ret :
                results = model.predict(frame)
                labeled_img,detected_objects = draw_box(frame, results, classNames)
                imgbytes = cv2.imencode('.jpg', labeled_img)[1].tobytes()
                detected_objects1=detected_objects
                yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + imgbytes + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('web.html',detected_objects=detected_objects1)


@app.route('/webapp')
def webapp():
    
    
    return Response(generate_frames_web(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/refresh',methods=['GET', 'POST'])
def refresh():
    return "hello"
    

@app.route('/section2_data_endpoint', methods=['GET'])
def get_section2_data():
    return render_template('section2.html', class_name_dict=classNames, detected_objects=detected_objects1)






   

if __name__ == "__main__":
    app.run(debug=True)

