import cv2 as cv
import numpy as np

A = [4, 5, 6, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
B = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 15, 16, 17]
SSD = 0
SAD = 0
CC = 0

for i in range(len(A)):
    SSD = SSD + (A[i] - B[i]) ** 2
    SAD = SAD + np.abs(A[i] - B[i])
    CC = CC + A[i] * B[i]

print(f"SSD: {SSD}")
print(f"SAD: {SAD}")
print(f"CC: {CC}")

A = np.array(A, dtype=np.uint8)
B = np.array(B, dtype=np.uint8)

TM_SQDIFF = cv.matchTemplate(A, B, cv.TM_SQDIFF_NORMED)
print(f"TM_SQDIFF: {TM_SQDIFF}")

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

window_name = 'frame'
win_out_name = 'frame_out'
cv.namedWindow(window_name, cv.WINDOW_NORMAL)
cv.namedWindow(win_out_name, cv.WINDOW_NORMAL)
template = cv.imread('template.png', 0)
while True:
    ret, frame = cap.read()
    # cv.imwrite("cam.png", frame)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    tmp_out = cv.matchTemplate(frame_gray, template, cv.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(tmp_out)
    center = (min_loc[0] + template.shape[0] // 2, min_loc[1] + template.shape[1] // 2)
    cv.circle(frame, center, 30, (0, 255, 0), -1)
    cv.imshow(window_name, frame)
    cv.imshow(win_out_name, tmp_out)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# if __name__ == '__main__':
#     print_hi('PyCharm')
