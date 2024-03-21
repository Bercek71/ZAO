import cv2 as cv
import numpy as np


window_name = 'Face Recognition'
cv.namedWindow(window_name)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
cl = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
while True:
    ret, frame = cap.read()
    cv.imshow(window_name, frame)
    dets = cl.detectMultiScale(frame, 1.2, 4)
    for d in dets:
        cv.rectangle(frame, d, (0, 0, 255), 4)
        cv.rectangle(frame, d, (0, 255, 0), 2)
    cv.imshow(window_name, frame)
    # print(f"""Detected {dets} faces""")
    if cv.waitKey(1) & 0xFF == ord('q'):
        break