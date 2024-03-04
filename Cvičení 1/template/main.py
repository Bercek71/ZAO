import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use("WebAgg")

print(cv2.__version__)


def cv_01():
    print("cv_01")
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    image = cv2.imread("img.png", 1)
    print(image.shape)
    cv2.startWindowThread()
    cv2.imshow("Image", image)
    print("showed")
    cv2.waitKey()


def cv_02():
    image = cv2.imread("img.png", 1)
    image_2 = cv2.imread("img-2.png", 1)
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
    image_2 = cv2.resize(image_2, (200, 200))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2)

    plt.show()


def cv_03():
    win_name = "window01"
    cv2.namedWindow(win_name, 0)
    img = np.ones([5, 5, 3], dtype=np.uint8)
    img[1, 4] = (255, 255, 0)
    print(img[1, 4], img.shape)
    cv2.imshow(win_name, img)
    cv2.waitKey()


def cv_04():
    win_name = "window01"

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(0)  # První parametr buďto id kamery (pr nb --- 1, jinak přidat path/to/vieo)
    # writer = cv2.VideoWriter("output.avi", cv2.VideoWriter.fourcc("M", "J", "P", "G"), cap.get(cv2.CAP_PROP_FPS),
    #                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), False)
    while (True):
        ret, frame = cap.read()
        edges = cv2.Canny(frame, 50, 100)
        # writer.write(edges)
        cv2.imshow(win_name, edges)
        if cv2.waitKey(1) == ord("q"):
            break

        # cv_02()


cv_04()
# cv_02()

# cv_01()
