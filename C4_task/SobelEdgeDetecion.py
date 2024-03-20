import cv2 as cv
import numpy as np

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
    [ -10, 0, 10],
    [ -1, 0, 1] ])
    
sobel_y = np.array([
    [ -1, -10, -1],
    [  0,  0,  0],
    [  1,  10,  1] ])

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
    sobel = np.zeros_like(originalPicture)
    
    # Edge candidate selection
    
    # Edge tracking
    for i in range(1, originalPicture.shape[0] - 1):
        for j in range(1, originalPicture.shape[1] - 1):
            if gradient_magnitude[i, j] > upper:
                sobel[i, j] = 255
            elif lower < gradient_magnitude[i, j] <= upper:
                if np.max(gradient_magnitude[i-1:i+2, j-1:j+2]) > lower:
                    sobel[i, j] = 255
    
    return sobel


def Sobel(originalPicture, upper, lower):
    # Convert to gray
    blur = cv.GaussianBlur(originalPicture, (5, 5), 0)
    # blur = cv.filter2D(blur, -1, kernel_1)
    dx = cv.filter2D(blur, cv.CV_32FC1, sobel_x)
    dy = cv.filter2D(blur, cv.CV_32FC1, sobel_y)
    dx = cv.convertScaleAbs(dx) # convert to 8 bit
    dy = cv.convertScaleAbs(dy) # convert to 8 bit

    # Put images together
    sobel = cv.addWeighted(dx, 0.5, dy, 0.5, 1)
    # sobel = cv.GaussianBlur(sobel, (15, 15), 0)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    sobel = cv.erode(sobel, kernel, iterations=1)

    _, soble = cv.threshold(sobel, 120, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    
    return soble

