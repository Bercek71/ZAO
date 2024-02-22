import cv2
import matplotlib

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


cv_01()


