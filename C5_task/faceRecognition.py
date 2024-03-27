import cv2 as cv
import numpy as np


window_name = 'Face Recognition'
cv.namedWindow(window_name)
window_name_eyes = 'Eyes Recognition'
cv.namedWindow(window_name_eyes)
window_name_threshold = 'Threshold'
cv.namedWindow(window_name_threshold)

cap = cv.VideoCapture("fusek_face_car_01.avi")
# cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
cl_face_front = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
cl_face_profile = cv.CascadeClassifier('haarcascades/haarcascade_profileface.xml')
cl_smile = cv.CascadeClassifier('haarcascades/haarcascade_smile.xml')
cl_eqes = cv.CascadeClassifier('haarcascades/eye_cascade_fusek.xml')
template = cv.imread('templates/template1.png', 0)
# template = cv.resize(template, (80, 50))
while True:
    ret, frame = cap.read()
    if frame is None:
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue
    cv.imshow(window_name, frame)
    dets, _, levelWeight = cl_face_front.detectMultiScale3(frame, 1.2, 3, minSize=(200, 200), outputRejectLevels=True)
    # print(levelWeight)
    face_side = "Front"
    if len(dets) == 0:
        dets = cl_face_profile.detectMultiScale(frame, 1.2, 3, minSize=(200, 200))
        face_side = "Profile"
    for d in dets:
        cv.rectangle(frame, d, (0, 0, 255), 4)
        cv.rectangle(frame, d, (0, 255, 0), 2)
        cv.putText(frame, face_side + " face", (d[0], d[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        face = frame[d[1]:d[1]+d[3], d[0]:d[0]+d[2]]
        eyes, _, levelW = cl_eqes.detectMultiScale3(face, 1.2, 4, minSize=(80, 80), outputRejectLevels=True)
        # print(f"""2: {something2}""")
        # print(f"""1: {something}""")
        toRemove = []
        length = len(eyes)
        for i in range(length):
            if levelW[i] < 8.1:
                toRemove.append(i)
        eyes = np.delete(eyes, toRemove, axis=0)
        min_vals = []
        for e in eyes:
            cv.rectangle(face, e, (0, 0, 255), 4)
            cv.rectangle(face, e, (0, 255, 0), 2)
            cv.putText(face, "Eye", (e[0], e[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            eyesFrame = face[e[1]:e[1]+e[3], e[0]:e[0]+e[2]]
            # eyeFrame = cv.cvtColor(eyesFrame, cv.COLOR_RGB2HSV)
            # eyeFrame = cv.cvtColor(eyesFrame, cv.COLOR_BGR2HSV)
            # eyeFrame = cv.medianBlur(eyeFrame, 6)
            # eyeFrame = cv.GaussianBlur(eyesFrame, (17, 17), 0)
            # eyeFrame = cv.cvtColor(eyeFrame, cv.COLOR_BGR2HSV)
            # frame_threshold = cv.inRange(eyeFrame, (110, 20, 140), (130, 30, 150))
            # mean = np.mean(frame_threshold)
            # means.append(-mean)

            # print("Threshold: ", np.mean(frame_threshold))
            eyesFrame = cv.cvtColor(eyesFrame, cv.COLOR_BGR2GRAY)
            eyesFrame = cv.equalizeHist(eyesFrame)
            eyesFrame = cv.medianBlur(eyesFrame, 5)
            ret, eyesFrame = cv.threshold(eyesFrame, 40, 255, cv.THRESH_BINARY)
            eyesFrame = cv.bitwise_not(eyesFrame)
            (cnt, hie) = cv.findContours(eyesFrame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            # move contours to the right position in frame
            for c in cnt:
                c[:, :, 0] += e[0] + d[0]
                c[:, :, 1] += e[1] + d[1]
            eyesFrame = cv.resize(eyesFrame, (80, 50))
            templateMatching = cv.matchTemplate(eyesFrame, template, cv.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(templateMatching)
            min_vals.append(min_val)
            print("Contours: ", cnt)
            print("Hierarchy: ", hie)
            print("Template matching: ", min_val)
                
            cv.drawContours(frame, cnt, -1, (0, 255, 0), 3)
            # eyesFrame = cv.Canny(eyesFrame, 100, 200)
            # eyesFrame = cv.dilate(eyesFrame, (3, 3), iterations=3)
            # cv.imshow(window_name_threshold, frame_threshold)
            cv.imshow(window_name_eyes, eyesFrame)
            cv.imshow(window_name, frame)
        mouth , _, lvlWeight = cl_smile.detectMultiScale3(face, 1.2, 4, minSize=(200, 200), outputRejectLevels=True)
        # print("Smile: ", lvlWeight)
        isOpen = False
        print(min_vals)
        for min_val in min_vals:
            print(min_val)
            if min_val < 0.9:
                isOpen = True
                break

        res = "closed"
        if isOpen:
            res = "open"
        cv.putText(frame, "Eyes " + res, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        to_remove = []
        length = len(mouth)
        for i in range(length):
            if lvlWeight[i] < 1:
                to_remove.append(i)
        mouth = np.delete(mouth, to_remove, axis=0)
        for m in mouth:
            cv.rectangle(face, m, (0, 0, 255), 4)
            cv.rectangle(face, m, (0, 255, 0), 2)
            cv.putText(face, "Mouth", (m[0], m[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            # print("Smile detected")
        cv.imshow(window_name, frame)
    # else:
        # dets = cl_face_profile.detectMultiScale(frame, 1.2, 4, minSize=(100, 100))

    if cv.waitKey(50) & 0xFF == ord('q'):
        break
    # print(f"""Detected {dets} faces""")