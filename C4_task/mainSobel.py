#!/usr/bin/python

import sys
import cv2 as cv
import numpy as np
import math
import struct
from datetime import datetime
import glob
from SobelEdgeDetecion import betterSobel

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

def getTotalAccuracy(values, res):
    totalCounter = 0
    correctCounter = 0
    for i in range(len(values)):
        for j in range(len(values[i])):
            if values[i][j] == res[i][j]:
                correctCounter += 1
            totalCounter += 1
    return correctCounter / totalCounter


def getTotalFScoreBinary(values, res):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for i in range(len(values)):
        for j in range(len(values[i])):
            if values[i][j] == res[i][j]:
                if values[i][j] == 1:
                    true_positives += 1
            else:
                if values[i][j] == 0 and res[i][j] == 1:
                    false_negatives += 1
                elif values[i][j] == 1 and res[i][j] == 0:
                    false_positives += 1

    if true_positives == 0:
        print(f"Precision F-score: Undefined")
        return
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        return 0
    fScore = 2 * (precision * recall) / (precision + recall)

    return fScore

def getAccuracy(values, res):
    correctCounter = 0
    for i in range(len(values)):
        if values[i] == res[i]:
            correctCounter += 1

    return correctCounter / len(values)



def getFScoreBinary(values, res):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for i in range(len(values)):
        if values[i] == res[i]:
            if values[i] == 1:
                true_positives += 1
        else:
            if values[i] == 0 and res[i] == 1:
                false_negatives += 1
            elif values[i] == 1 and res[i] == 0:
                false_positives += 1

    if true_positives == 0:
        print(f"Precision F-score: Undefined")
        return
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        return 0
    fScore = 2 * (precision * recall) / (precision + recall)

    return fScore

def templateMatchingSqdiffNormed(image, template):
    imageHeight, imageWidth = image.shape[:2]
    temlateHeight, templateWidth = template.shape[:2]

    result = np.zeros((imageHeight - temlateHeight + 1, imageWidth - templateWidth + 1), dtype=np.float32)

    for y in range(imageHeight - temlateHeight + 1):
        for x in range(imageWidth - templateWidth + 1):
            pixel = image[y:y+temlateHeight, x:x+templateWidth]
            diff = (pixel - template) ** 2
            normed_score = (np.sum(diff) / (temlateHeight * templateWidth)) / 1000
            result[y, x] = normed_score
    return result



def main(argv):

    pkm_file = open('parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []

    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)

    test_images = [(img, img.replace(".jpg", ".txt")) for img in glob.glob("test_images_zao/*.jpg")]
    print(test_images)
    test_results = [img for img in glob.glob("test_images_zao/*.txt")]
    test_results.sort()
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
    window5 = "Edges Image"
    cv.namedWindow(window5, cv.WINDOW_NORMAL)
    window6 = "Matching Image"
    cv.namedWindow(window6, cv.WINDOW_NORMAL)
    window7 = "Matching Horizontal Image"
    cv.namedWindow(window7, cv.WINDOW_NORMAL)
    totalPredictedResults = []
    for img_name, result_name in test_images:
        parking_classificator_results = []
        print(img_name)
        image = cv.imread(img_name, cv.IMREAD_COLOR)
        image_blackandWhite = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # cv.imshow(windowName, image)
        counter = 1
        avg = np.mean(image)
        for c in pkm_coordinates:
            one_image = four_point_transform(image_blackandWhite, c)
            one_image_car = cv.resize(one_image, (45, 45))
            if avg > 100:
                kernel =  cv.getStructuringElement(cv.MORPH_RECT, (8, 8))
                kernelErode = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
                erode = cv.erode(one_image_car, kernelErode, iterations=1)
                erode = cv.GaussianBlur(one_image_car, (3, 3), 0)
                edges =  betterSobel(erode, 100, 50)

                matchingKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (18, 17))
                matchingErode = cv.erode(one_image_car, matchingKernel, iterations=2)
                matchingDiletate = cv.dilate(matchingErode, matchingKernel, iterations=1)
                # cv.imshow(window6, matchingDiletate)


                # diletate = cv.dilate(erode, kernel, iterations=1)

                template = cv.imread('templates/template9.png', 0)
                template = cv.resize(template, (5, 45))
                # matchingResult = template_matching_sqdiff_normed(matchingDiletate, template)
                matchingResult = cv.matchTemplate(matchingDiletate, template, cv.TM_SQDIFF_NORMED)

                # cv.imshow(window7, matchingResult)
                min_val = np.min(matchingResult)
                templateHorizontal = cv.imread('templates/template10.png', 0)
                templateHorizontal = cv.resize(templateHorizontal, (45, 30))
                # horizontalResult = template_matching_sqdiff_normed(matchingDiletate, templateHorizontal)
                horizontalResult = cv.matchTemplate(matchingDiletate, templateHorizontal, cv.TM_SQDIFF_NORMED)
                min_val_horizontal = np.min(horizontalResult)
                # print("edges average: " + str(np.mean(edges)))
                color = None
                edges_average = np.mean(edges)
                # print("Edges average: " + str(edges_average))
                if edges_average > 60:
                        parking_classificator_results.append(1)
                        # print("Car detected")
                        color = (0, 0, 255)
                elif min_val > 0.8:
                    parking_classificator_results.append(1)
                    # print("Car detected")
                    color = (0, 0, 255)
                elif min_val_horizontal > 0.8:
                    parking_classificator_results.append(1)
                    # print("Car detected")
                    color = (0, 0, 255)
                elif min_val > 0.308 and edges_average > 10 and min_val_horizontal > 0.3208:
                    parking_classificator_results.append(1)
                    # print("Car detected")
                    color = (0, 0, 255)
                elif min_val > 0.17 and min_val_horizontal > 0.322 and edges_average > 21:
                    parking_classificator_results.append(1)
                    # print("Car detected")
                    color = (0, 0, 255)
                elif min_val > 0.33 and min_val_horizontal > 0.29 and edges_average > 17:
                    parking_classificator_results.append(1)
                    # print("Car detected")
                    color = (0, 0, 255)
                elif min_val > 0.269 and min_val_horizontal > 0.24 and edges_average > 20:
                    parking_classificator_results.append(1)
                    # print("Car detected")
                    color = (0, 0, 255)
                else:
                    parking_classificator_results.append(0)
                    # print("Empty place detected")
                    color = (0, 255, 0)
                # print(edges_average)
                center = (int(c[0]), int(c[1]))
                points = [(int(c[i]), int(c[i+1])) for i in range(0, len(c), 2)]

                center_x = sum(point[0] for point in points) / len(points)
                center_y = sum(point[1] for point in points) / len(points)
                center = (int(center_x), int(center_y))
                cv.circle(image, center, 10, color, -1)
                (textCenterX, textCenterY) = center
                cv.putText(image, str(counter), (textCenterX + 10, textCenterY + 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                counter += 1
                cv.imshow(wondow4, edges)
                cv.imshow(window2, one_image_car)
                cv.imshow(windowName, image)
                print("edges average: " + str(np.mean(edges)))
            else:
                kernel = cv.getStructuringElement(cv.MORPH_RECT, (8, 8))
                kernelErode = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
                # erode = cv.erode(one_image_car, kernelErode, iterations=1)
                # erode = cv.GaussianBlur(one_image_car, (3, 3), 0)
                edges = betterSobel(one_image_car, 100, 50)

                # diletate = cv.dilate(erode, kernel, iterations=1)


                cv.imshow(window7, matchingResult)

                color = None
                edges_average = np.mean(edges)
                # print("Edges average: " + str(edges_average))
                if edges_average > 50:
                        parking_classificator_results.append(1)
                        # print("Car detected")
                        color = (0, 0, 255)
                else:
                    parking_classificator_results.append(0)
                    # print("Empty place detected")
                    color = (0, 255, 0)
                # print(edges_average)
                center = (int(c[0]), int(c[1]))
                points = [(int(c[i]), int(c[i+1])) for i in range(0, len(c), 2)]

                center_x = sum(point[0] for point in points) / len(points)
                center_y = sum(point[1] for point in points) / len(points)
                center = (int(center_x), int(center_y))
                cv.circle(image, center, 10, color, -1)
                (textCenterX, textCenterY) = center
                cv.putText(image, str(counter), (textCenterX + 10, textCenterY + 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                counter += 1
                cv.imshow(wondow4, edges)
                cv.imshow(window2, one_image_car)
                print("edges average: " + str(np.mean(edges)))

            # cv.waitKey(0)

        print("********************************************************")
        print("Results for: " + result_name)
        print("Average: " + str(avg))



        file = open(result_name, 'r')
        lines = file.readlines()
        res = [int(x) for x in lines]
        accuracy = getAccuracy(parking_classificator_results, res)
        fscore =  getFScoreBinary(parking_classificator_results, res)
        
        cv.putText(image, f"Accuracy: {accuracy}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(image, f"F-score: {fscore}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imshow(windowName, image)
        totalPredictedResults.append(parking_classificator_results)
        # parking_classificator_results .clear()
        # cv.waitKey(0)

    totalResults = []
    for _, resultName in test_images:
        file = open(resultName, 'r')
        lines = file.readlines()
        res = [int(x) for x in lines]
        totalResults.append(res)
    print("********************************************************")
    print("Total accuracy: " + str(getTotalAccuracy(totalResults, totalPredictedResults)))
    print("Total F-score: " + str(getTotalFScoreBinary(totalResults, totalPredictedResults)))
    print("********************************************************")

if __name__ == "__main__":
   main(sys.argv[1:])