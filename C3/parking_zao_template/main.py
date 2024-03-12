#!/usr/bin/python

import sys
import cv2 as cv
import numpy as np
import math
import struct
from datetime import datetime
import glob

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, one_c):
    #https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    
    pts = [((float(one_c[0])), float(one_c[1])),
            ((float(one_c[2])), float(one_c[3])),
            ((float(one_c[4])), float(one_c[5])),
            ((float(one_c[6])), float(one_c[7]))]
    
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(np.array(pts))
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
	    [0, 0],
	    [maxWidth - 1, 0],
	    [maxWidth - 1, maxHeight - 1],
	    [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

    
def main(argv):

    pkm_file = open('parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []
   
    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)
      
    test_images = [img for img in glob.glob("test_images_zao/*.jpg")]
    test_images.sort()
    print(pkm_coordinates)
    print("********************************************************")     

    windowName = "Parking Map"
    cv.namedWindow(windowName, cv.WINDOW_NORMAL)
    window2 = "Warped Image"
    cv.namedWindow(window2, cv.WINDOW_NORMAL)
    window3 = "Diletate Image"
    cv.namedWindow(window3, cv.WINDOW_NORMAL)
    wondow4 = "Erode Image"
    cv.namedWindow(wondow4, cv.WINDOW_NORMAL)    

    parking_classificator = []

    for img_name in test_images:
        print(img_name)
        image = cv.imread(img_name, cv.IMREAD_COLOR)
        image_blackandWhite = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # cv.imshow(windowName, image)
        counter = 1
        for c in pkm_coordinates:
            one_image = four_point_transform(image_blackandWhite, c)
            one_image_car = cv.resize(one_image, (80, 80))
            # kernel =  cv.getStructuringElement(cv.MORPH_RECT, (50, 50))
            edges = cv.Canny(one_image_car, 100, 200)
            # erode = cv.erode(edges, kernel, iterations=1)
            # diletate = cv.dilate(erode, kernel, iterations=1)

            template = cv.imread('templates/template2.png', 0)
            matchingResult = cv.matchTemplate(one_image_car, template, cv.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(matchingResult)
            print("Matching: "+ str(min_val))

            color = None
            edges_average = np.mean(edges)
            print("Edges average: " + str(edges_average))
            if edges_average > 19 and min_val > 0.1:
                parking_classificator.append(1)
                print("Car detected")
                color = (0, 0, 255)
            elif edges_average > 30:
                parking_classificator.append(1)
                print("Car detected")
                color = (0, 0, 255)
            elif min_val > 0.14:
                parking_classificator.append(1)
                print("Car detected")
                color = (0, 0, 255)
            else:
                parking_classificator.append(0)
                print("Empty place detected")
                color = (0, 255, 0)
            print(edges_average)
            #put circle in the center of the parking place c has all coordinates in array
            # c[0] = x1, c[1] = y1, c[2] = x2, c[3] = y2, c[4] = x3, c[5] = y3, c[6] = x4, c[7] = y4
            center = (int(c[0]), int(c[1])) 
            points = [(int(c[i]), int(c[i+1])) for i in range(0, len(c), 2)]

# Calculate the average of x and y coordinates to find the center
            center_x = sum(point[0] for point in points) / len(points)
            center_y = sum(point[1] for point in points) / len(points)
            center = (int(center_x), int(center_y))
            cv.circle(image, center, 10, color, -1)
            (textCenterX, textCenterY) = center
            cv.putText(image, str(counter), (textCenterX + 10, textCenterY + 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            counter += 1
            cv.imshow(wondow4, edges)
            # cv.imshow(window3, diletate)
            cv.imshow(window2, one_image_car)
            # if cv.waitKey(0) == ord('q'):
            #     break
            #write text to image with 1 or 0
            # cv.putText(image, str(parking_classificator[-1]), bottomLeftOrigin=(c[0], c[1]))
        cv.imshow(windowName, image)
        cv.waitKey(0)
if __name__ == "__main__":
   main(sys.argv[1:])     