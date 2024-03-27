import cv2 as cv
import numpy as np


window_name = 'Face Recognition'
cv.namedWindow(window_name)
window_name_eyes = 'Eyes Recognition'
cv.namedWindow(window_name_eyes)

cap = cv.VideoCapture("fusek_face_car_01.avi")
# cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
cl_face_front = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
cl_face_profile = cv.CascadeClassifier('haarcascades/haarcascade_profileface.xml')
cl_eqes = cv.CascadeClassifier('haarcascades/eye_cascade_fusek.xml')
while True:
    ret, frame = cap.read()
    if frame is None:
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue
    cv.imshow(window_name, frame)
    dets = cl_face_front.detectMultiScale(frame, 1.2, 3, minSize=(5, 5))
    if len(dets) == 0:
        dets = cl_face_profile.detectMultiScale(frame, 1.2, 3, minSize=(5, 5))
    for d in dets:
        cv.rectangle(frame, d, (0, 0, 255), 4)
        cv.rectangle(frame, d, (0, 255, 0), 2)
        cv.putText(frame, "Face", (d[0], d[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        face = frame[d[1]:d[1]+d[3], d[0]:d[0]+d[2]]
        eyes = cl_eqes.detectMultiScale(face, 1.2, 4, minSize=(5, 5))
        for e in eyes:
            cv.rectangle(face, e, (0, 0, 255), 4)
            cv.rectangle(face, e, (0, 255, 0), 2)
            cv.putText(face, "Eye", (e[0], e[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            eyesFrame = face[e[1]:e[1]+e[3], e[0]:e[0]+e[2]]

    

    # else:
        # dets = cl_face_profile.detectMultiScale(frame, 1.2, 4, minSize=(100, 100))

    if cv.waitKey(3) & 0xFF == ord('q'):
        break
    # print(f"""Detected {dets} faces""")