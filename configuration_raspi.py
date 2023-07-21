import PySimpleGUI as sg
import cv2
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
from ultralytics import YOLO
import math

def capture_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        sg.popup_error('Failed to access the camera!')
        return

    ret, frame = cap.read()

    if ret:
        return frame

def yolo_detection(img):
    model = YOLO("C:\\Users\\PMTC-ELE\\Downloads\\best.pt")
    results = model.predict(img)

    classNames = {0:"ok",1:"not_ok"}

    for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]

                if class_name == "ok":
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    label = f"{class_name} - {conf}"
                    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3
                    cv2.rectangle(img, (x1, y1), c2, [255, 255, 255], -1, cv2.LINE_AA)
                    cv2.putText(img, label, (x1, y1 - 2), 0, 1, [0, 255, 0], thickness=2, lineType=cv2.LINE_AA)
                if class_name == "not_ok":
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    label = f"{class_name} - {conf}"
                    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3
                    cv2.rectangle(img, (x1, y1), c2, [255, 255, 255], -1, cv2.LINE_AA)
                    cv2.putText(img, label, (x1, y1 - 2), 0, 1, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)
    return img


def main():
    #img = capture_frames()
    img=cv2.imread("C:\\Users\\PMTC-ELE\\Documents\\black_ring\\ok\\frame_notok1320.jpg") 
    image = yolo_detection(img)
    cv2.imwrite("ring1.jpg",image)



main()