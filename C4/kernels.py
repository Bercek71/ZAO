import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use("WebAgg")

# https://docs.opencv.org/4.x/d4/dbd/tutorial_filter_2d.html
# https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
# https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html


kernel_1 = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0] ])

kernel_2 = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1] ]) / 9

kernel_3 = np.ones([11, 11]) 
kernel_3 /= 11*11

dx_kernel = np.array([[-1, 0, 1]])
dy_kernel = dx_kernel.T

sobel_x = np.array([
    [ -1, 0, 1],
    [ -2, 0, 2],
    [ -1, 0, 1] ])
    
sobel_y = np.array([
    [ -1, -2, -1],
    [  0,  0,  0],
    [  1,  2,  1] ])


window1 = "Original"
window2 = "Filtered"

image =cv.imread("test2.jpg", 0)
cv.imshow(window1, image)

# filtered = cv.filter2D(image, -1, kernel_1) 
dx = cv.filter2D(image, cv.CV_32FC1, sobel_x)
dy = cv.filter2D(image, cv.CV_32FC1, sobel_y)
dx = cv.convertScaleAbs(dx) # convert to 8 bit
dy = cv.convertScaleAbs(dy) # convert to 8 bit
# connect dx and dy


final = cv.addWeighted(dx, 0.5, dy, 0.5, 1)
sobel_edges = cv.Sobel(image, cv.CV_64F, 1, 1, ksize=5)
cv.imshow(window2, final)
cv.imshow(window1, sobel_edges)
cv.waitKey(0)



def betterSobel(originalPicture, upper, lower):
    # Convert to gray
    # blur = cv.GaussianBlur(originalPicture, (5, 5), 0)
    # # blur = cv.filter2D(blur, -1, kernel_1)
    # dx = cv.filter2D(originalPicture, cv.CV_32FC1, sobel_x)
    # dy = cv.filter2D(originalPicture, cv.CV_32FC1, sobel_y)
    # dx = cv.convertScaleAbs(dx) # convert to 8 bit
    # dy = cv.convertScaleAbs(dy) # convert to 8 bit

    # # Put images together
    # sobel = cv.addWeighted(dx, 0.5, dy, 0.5, 1)
    # # sobel = cv.GaussianBlur(sobel, (15, 15), 0)
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    # sobel = cv.erode(sobel, kernel, iterations=1)

    # _, soble = cv.threshold(sobel, 120, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    
    # Compute gradients using Sobel operator
    # gradient_x = cv.Sobel(originalPicture, cv.CV_64F, 1, 0, ksize=3)
    # gradient_y = cv.Sobel(originalPicture, cv.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude and direction

        # Compute gradients using Sobel operator
    dx = cv.Sobel(originalPicture, cv.CV_64F, 1, 0, ksize=3)
    dy = cv.Sobel(originalPicture, cv.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(dx**2 + dy**2)
    # gradient_direction = np.arctan2(dy, dx)
    # Initialize output image
    output_image = np.zeros_like(originalPicture)
    
    # Edge candidate selection
    strong_edges = gradient_magnitude > upper
    weak_edges = (gradient_magnitude <= upper) & (gradient_magnitude >= lower)
    
    # Mark strong edge pixels
    output_image[strong_edges] = 255
    
    # Edge tracking
    for i in range(1, originalPicture.shape[0] - 1):
        for j in range(1, originalPicture.shape[1] - 1):
            if weak_edges[i, j]:
                if (strong_edges[i-1:i+2, j-1:j+2]).any():
                    output_image[i, j] = 255
    
    return output_image
