 
import matplotlib
import cv2 as cv
import numpy as np


cap = cv.VideoCapture("03/kor-4-2.avi")
# cap = cv.VideoCapture(0)

windowName = "Live Video Feed"
windowName2 = "Thresholded Video Feed"
windowName3 = "Dilated Video Feed"
windowName4 = "Eroded Video Feed"
cv.namedWindow(windowName, 0)
cv.namedWindow(windowName2, 0)
cv.namedWindow(windowName3, 0)  
cv.namedWindow(windowName4, 0)

while True:
    ret, frame = cap.read()
    if frame is None:
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #120
    ret, frame_th = cv.threshold(frame_gray, 120, 255, cv.THRESH_BINARY, frame_gray)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (6, 6))
    frame_dil = cv.dilate(frame_th, kernel)
    kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (30, 30))
    frame_ero = cv.erode(frame_dil, kernel2)

    (cnt, hie) = cv.findContours(frame_ero, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    print(len(cnt))

    cv.drawContours(frame, cnt, -1, (0, 255, 0), 3)
    cv.imshow(windowName4, frame_ero)
    cv.imshow(windowName3, frame_dil)
    cv.imshow(windowName, frame)
    cv.imshow(windowName2, frame_th)
    if cv.waitKey(2) == ord('q'):
        break

