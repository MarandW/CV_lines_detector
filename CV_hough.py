import cv2 as cv
import numpy as np
import os
import re
import math


################################################################################################
#
# Mariusz Wisniewski KD-712
#
# Computer Vision
# 4606-ES-000000C-0128
#
# Measuring the position of the scale line on the camera image
#
# Edge detection with Hough line fitting
#
# 2023-06-07
#
################################################################################################



def measurements_from_filename(name):
    t = re.split('[_.]', name)
    if t[0] == "kreska":
        measurement_time, manual_position = t[1], (float(t[2]), float(t[3]), (float(t[2]) + float(t[3])) / 2.)
#        print(measurement_time, manual_position)
        return (measurement_time, manual_position)
    else:
        return None, None


def create_a_list_of_test_files_in_the_directory(dir_path):

    image_list = []

    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            # check only text files
            if path.endswith('.png'):
                image_list.append(path)
    print("total: ", len(image_list), "images")

    return image_list


def detektor_hough(imgraw, edge_ignore_width, max_line_width, only_middle_line, fname):

    sizewindow = 100
    maxsizeline = max_line_width
    cl = 50  # cut firsts lines

    measurement_time, manual_position = measurements_from_filename(fname)
    cropped_image = imgraw[0:480, cl:640]

    grey = cv.cvtColor(cropped_image, cv.COLOR_RGB2GRAY)
    greym = cv.medianBlur(src=grey, ksize=11)
    blur = cv.blur(greym, (5, 199))

    sobelx = cv.Sobel(blur, ddepth=cv.CV_64F, dx=1, dy=0, ksize=9, scale=0.0005, delta=0, borderType=cv.BORDER_DEFAULT)
    sobelxn = -sobelx

    sobelx8 = np.where(sobelx < 0, 0, sobelx)
    sobelx8n = np.where(sobelxn < 0, 0, sobelxn)
    sobelx8 = cv.convertScaleAbs(sobelx8)
    sobelx8n = cv.convertScaleAbs(sobelx8n)

    _, th_otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    _, th_sobelx = cv.threshold(sobelx8, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    _, th_sobelxn = cv.threshold(sobelx8n, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    lines = cv.HoughLines(th_sobelx, 1, 1, 150, None, 0, 0, min_theta=0, max_theta=0.1)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv.line(cropped_image, pt1, pt2, (0, 0, 255), 1, cv.LINE_AA)

    linesP = cv.HoughLines(th_sobelxn, 1, 1, 150, None, 0, 0, min_theta=0, max_theta=0.1)

    if linesP is not None:
        for i in range(0, len(linesP)):
            rho = linesP[i][0][0]
            theta = linesP[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv.line(cropped_image, pt1, pt2, (255, 0, 0), 1, cv.LINE_AA)

    if lines is not None and linesP is not None:
        linemed = []
        k = 0
        jmin = 0
        for i in range(0, len(lines)):
            distance = 2000   # something big
            for j in range(0, len(linesP)):
                if abs(lines[i][0][0] - linesP[j][0][0]) < distance and abs(lines[i][0][0] - linesP[j][0][0]) < maxsizeline:
                    distance = abs(lines[i][0][0] - linesP[j][0][0])
                    jmin = j
            if distance <= maxsizeline:
                linemed.append([int((lines[i][0][0] + linesP[jmin][0][0]) / 2), i, jmin])
                k += 1

        xlim1 = 640/2 - sizewindow - cl
        xlim2 = 640/2 + sizewindow - cl
        linefinal = []
        distance = 2000  # somethig big
        k = -1
        if len(linemed) > 1:
            for m in range(len(linemed)):
                if xlim1 < linemed[m][0] < xlim2 and abs(linemed[m][0]-640/2-cl) < distance:
                    distance = abs(linemed[m][0]-640/2-cl)
                    linefinal = linemed[m].copy()
                    k = m
            if k < 0:
                print("too far from the center")
            else:
                for n in range(len(linemed)):
                    if n != k and xlim1 < linemed[n][0] < xlim2 and abs(linemed[k][0]-linemed[n][0]) < maxsizeline:

                        a1 = lines[linefinal[1]][0][0]
                        a2 = linesP[linefinal[2]][0][0]
                        b1 = lines[linemed[n][1]][0][0]
                        b2 = linesP[linemed[n][2]][0][0]

                        if a1 < a2:
                            if a1 > b1:
                                linefinal[1] = linemed[n][1]
                            if a2 < b2:
                                linefinal[2] = linemed[n][2]

                        else:
                            if a1 < b1:
                                linefinal[1] = linemed[n][1]
                            if a2 > b2:
                                linefinal[2] = linemed[n][2]
                        linefinal[0] = int((lines[linefinal[1]][0][0] + linesP[linefinal[2]][0][0]) / 2)

        else:
            if len(linemed) > 0:
                if xlim1 < linemed[0][0] < xlim2:
                    linefinal = linemed[0].copy()
                    k = 1
#                    cv.line(cropped_image, (linemed[k][0], 0), (linemed[k][0], 480), (0, 255, 0), 3, cv.LINE_AA)
                else:
                    print("too far from the center")
            else:
                print("no line")
#        print(k)
        if k >= 0:
            print("final line", linefinal)
            cv.line(cropped_image, (linefinal[0], 0), (linefinal[0], 480), (0, 155, 255), 3, cv.LINE_AA)
            cv.line(cropped_image, (int(lines[linefinal[1]][0][0]), 0), (int(lines[linefinal[1]][0][0]), 480), (0, 0, 255), 3, cv.LINE_AA)
            cv.line(cropped_image, (int(linesP[linefinal[2]][0][0]), 0), (int(linesP[linefinal[2]][0][0]), 480), (255, 0, 0), 3, cv.LINE_AA)

            if manual_position:
                f = open("detekcja_hough.txt", "a")
                f.write(fname + " " + format(lines[linefinal[1]][0][0]+cl, '.1f') + " " + format(linesP[linefinal[2]][0][0]+cl, '.1f') + " " + format(linefinal[0]+cl, '.1f')
                        + " " + format(manual_position[0], '.1f') + " " + format(manual_position[1], '.1f') + " " + format(manual_position[2], '.1f')
                        + " " + format(linefinal[0]+cl - manual_position[2], '.1f') + "\n")
                f.close()

    sobelxc = cv.cvtColor(sobelx8, cv.COLOR_GRAY2BGR)
    sobelxnc = cv.cvtColor(sobelx8n, cv.COLOR_GRAY2BGR)
#    cv.imshow('blur', blur)
#    cv.imshow('th_sobelx', th_sobelx)
#    cv.imshow('th_sobelxn', th_sobelxn)
#    cv.imshow('Sobel X', sobelxc)
#    cv.imshow('Sobel Xn', sobelxnc)
#    cv.imshow('th_otsu', cropped_image)
    v_img1 = cv.vconcat([cv.cvtColor(blur, cv.COLOR_GRAY2BGR), cropped_image])
    v_img2 = cv.vconcat([sobelxc, sobelxnc])
    v_img = cv.hconcat([v_img1, v_img2])
    cv.imshow('v_img', v_img)
#    name = fname[:-4]
#    nfname = "report/" + name + "_hough.png"
#    cv.imwrite(nfname, v_img)
    cv.waitKey(200)

    return 0


######################################################
# main
######################################################
if __name__ == '__main__':
    #dir_path = r'C:\Users\maran\Documents\images\opisane'
    #dir_path = r'C:\Users\maran\Documents\selected'
    dir_path = './selected'

    ignorujkrawedzie = 100  # zaweza pole sprawdzania do centralnej czesci obrazu
    maxszerokosc = 110  # zalezna od spodziewanego rozmiaru kresek i skali
    tylkosrodkowa = True

    tkreski = create_a_list_of_test_files_in_the_directory(dir_path)

    for fname in tkreski:
        print(fname)
        tname = os.path.join(dir_path, fname)
        imgraw = cv.imread(tname)

        if imgraw is None:
            ret = False
            break
        else:
            ret = True

        if ret:
            kreski = detektor_hough(imgraw, ignorujkrawedzie, maxszerokosc, tylkosrodkowa, fname)
