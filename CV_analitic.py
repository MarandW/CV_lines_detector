import cv2 as cv
import numpy as np
import os
import re
import matplotlib.pyplot as plt


################################################################################################
#
# Mariusz Wisniewski KD-712
#
# Computer Vision
# 4606-ES-000000C-0128
#
# Measuring the position of the scale line on the camera image
#
# Contrast enhancement and analytical line position detection
#
# 2023-06-07
#
################################################################################################


def measurements_from_filename(name):
    # read position measured manually from file name
    t = re.split('[_.]', name)
    if t[0] == "kreska":
        measurement_time, manual_position = t[1], (float(t[2]), float(t[3]), (float(t[2]) + float(t[3])) / 2. )
#        print(measurement_time, manual_position)
        return ( measurement_time, manual_position )
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


def glue_lines(tresh, ignore):
    j = 0
    stan0 = 0
    stan1 = 0
    for i in range(ignore, len(tresh) - ignore):
        stan = tresh[i]
        if stan0 == 0:
            stan0 = stan
            stan1 = stan
        else:
            if stan != stan1:
                if stan == 0:
                    j += 1
                else:
                    if stan == stan0:
                        for k in range(i - j, i):
                            tresh[k] = stan0
                    stan0 = stan
                    j = 0
            else:
                if stan == 0:
                    j += 1
            stan1 = stan

    return tresh


def is_line_black(tresh, ignore):
    white_line = 0
    black_line = 0
    for i in range(ignore, len(tresh) - ignore):
        if tresh[i] == 1:
            white_line += 1
        if tresh[i] == -1:
            black_line += 1
    if black_line <= white_line :
        return True    # dark lines on a light background
    else:
        return False   # light lines on a dark background


def create_lines_table(tresh, ignore, blackline, max_line_width):
    tab = []
    if blackline:
        stan = 1
        line_beginning = ignore
        for i in range(ignore, len(tresh) - ignore):
            if tresh[i] != stan:
                if tresh[i] < stan and stan == 1:     #
                    line_beginning = i
                if tresh[i] > stan and tresh[i] == 1:
                    tab.append( [line_beginning, i, (i+line_beginning)/2.] )
            stan = tresh[i]
    else:
        stan = -1
        line_beginning = ignore
        for i in range(ignore, len(tresh) - ignore):
            if tresh[i] != stan:
                if tresh[i] > stan and stan == -1:
                    line_beginning = i
                if tresh[i] < stan and tresh[i] == -1:
                    tab.append( [line_beginning, i, (i+line_beginning)/2.])
            stan = tresh[i]

    if len(tab)>1:
        i = 0
        while i < len(tab)-1:
            if tab[i+1][0] - tab[i][1] < max_line_width:
                tab[i][1] = tab[i+1][1]
                tab.pop(i+1)
            else:
                i += 1
    for i in range(len(tab)):
        tab[i][2] = (tab[i][1] + tab[i][0])/2.

    return tab


def hysteresis_thresholding(hist_10, edge_ignore_width):

    min_ = np.min(hist_10)
    max_ = np.max(hist_10)

    min_val = max_
    for i in range(len(hist_10)):
        if edge_ignore_width < i < len(hist_10) - edge_ignore_width and min_val > hist_10[i]:
            min_val = hist_10[i]
    max_val = min_
    for i in range(len(hist_10)):
        if edge_ignore_width < i < len(hist_10) - edge_ignore_width and max_val < hist_10[i]:
            max_val = hist_10[i]

    amp = max_val - min_val

    treshold = np.zeros([len(hist_10)], dtype=int)
    for i in range(edge_ignore_width, len(hist_10) - edge_ignore_width):
        if hist_10[i] > max_val - amp * 0.3:
            treshold[i] = 1
        if hist_10[i] < min_val + amp * 0.3:
            treshold[i] = -1

    return (treshold, amp)


def find_the_line_in_the_middle(lines):
    image_center = 640 / 2
    dist = image_center * 2
    middle_line = -1
    for i in range(len(lines)):
        if dist > abs((lines[i][1] + lines[i][0]) / 2 - image_center):
            dist = abs((lines[i][1] + lines[i][0]) / 2 - image_center)
            middle_line = i
    if middle_line < 0:
        print("something wrong...")
    return middle_line


def lines_detector(imgraw, edge_ignore_width, max_line_width, only_middle_line, fname):
    pk = 0  # manual correction

    measurement_time, manual_position = measurements_from_filename(fname)

    imggray = cv.cvtColor(imgraw, cv.COLOR_BGR2GRAY)

    # kernel
    element = np.ones((29, 1), np.float32) / (29)

    # close/open operation
    morphC = cv.morphologyEx(imggray, cv.MORPH_CLOSE, element)
    morphO = cv.morphologyEx(imggray, cv.MORPH_OPEN, element)

    # averaging
    fil2dO = cv.filter2D(morphO, -2, element)
    fil2dC = cv.filter2D(morphC, -2, element)

    # Otsu thresholding
    _, th3_otsuO = cv.threshold(fil2dO, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    _, th3_otsuC = cv.threshold(fil2dC, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    #    cv.imshow("morphC", morphC)
    #    cv.imshow("morphO", morphO)
    #    cv.imshow("fil2dO", fil2dO)
    #    cv.imshow("fil2d", fil2dC)
    #    cv.imshow("th3_otsuO", th3_otsuO)
    #    cv.imshow("th3_otsuC", th3_otsuC)

    # convert to 1D signal
    histO = np.sum(th3_otsuO, axis=0)
    histC = np.sum(th3_otsuC, axis=0)

    # moving average
    kernel_size = 9
    kernel = np.ones(kernel_size) / kernel_size
    histO_10 = np.convolve(histO, kernel, mode='same')
    histC_10 = np.convolve(histC, kernel, mode='same')

    # hysteresis thresholding
    tresholdO, ampO = hysteresis_thresholding(histO_10, edge_ignore_width)
    tresholdC, ampC = hysteresis_thresholding(histC_10, edge_ignore_width)

    # gluing function
    glue_lines(tresholdO, edge_ignore_width)
    glue_lines(tresholdC, edge_ignore_width)

    # recognizing whether we are dealing with dark lines on a light background or light on a dark background
    black_lines_O = is_line_black(tresholdO, edge_ignore_width)
    black_lines_C = is_line_black(tresholdC, edge_ignore_width)

    # final table of lines
    lines_O = create_lines_table(tresholdO, edge_ignore_width, black_lines_O, max_line_width)
    lines_C = create_lines_table(tresholdC, edge_ignore_width, black_lines_C, max_line_width)

    # best result
    if ampC >= ampO:
        black_lines = black_lines_C
    else:
        black_lines = black_lines_O

    if black_lines:  #lines are black
        final_lines_list = lines_C
    else:  # lines are white
        final_lines_list = lines_O

    if only_middle_line:
        #  if we are looking for only single line
        if len(lines_O) > 0:
            middle_line_O = find_the_line_in_the_middle(lines_O)
            result_O = np.zeros([len(histO_10)], dtype=int)
            for i in range(len(result_O)):
                if i < lines_O[middle_line_O][0]:
                    result_O[i] = 0
                if i >= lines_O[middle_line_O][0] and i < lines_O[middle_line_O][1]:
                    result_O[i] = 1
                if i >= lines_O[middle_line_O][1]:
                    result_O[i] = 0

            if not black_lines:
                final_lines_list = lines_O[middle_line_O]
                result = result_O
                print("middle_line O", final_lines_list)

        if len(lines_C) > 0:
            middle_line_C = find_the_line_in_the_middle(lines_C)
            result_C = np.zeros([len(histC_10)], dtype=int)
            for i in range(len(result_C)):
                if i < lines_C[middle_line_C][0]:
                    result_C[i] = 0
                if i >= lines_C[middle_line_C][0] and i < lines_C[middle_line_C][1]:
                    result_C[i] = 1
                if i >= lines_C[middle_line_C][1]:
                    result_C[i] = 0
            if black_lines:
                final_lines_list = lines_C[middle_line_C]
                result = result_C
                print("middle_line C",final_lines_list)

        imgrawO = imgraw.copy()
        if len(lines_O) > 0:
            if black_lines:
                cv.line(imgrawO, (lines_O[middle_line_O][0] - 5 + pk, 0), (lines_O[middle_line_O][0] - 5 + pk, 480),
                         (255, 0, 0), 2)
                cv.line(imgrawO, (lines_O[middle_line_O][1] + 5 + pk, 0), (lines_O[middle_line_O][1] + 5 + pk, 480),
                         (255, 0, 0), 2)
            else:
                cv.line(imgrawO, (lines_O[middle_line_O][0] - 5 + pk, 0), (lines_O[middle_line_O][0] - 5 + pk, 480),
                         (0, 0, 255), 2)
                cv.line(imgrawO, (lines_O[middle_line_O][1] + 5 + pk, 0), (lines_O[middle_line_O][1] + 5 + pk, 480),
                         (0, 0, 255), 2)
        cv.imshow("imgrawO", imgrawO)

        imgrawC = imgraw.copy()
        if len(lines_C) > 0:
            if black_lines:
                cv.line(imgrawC, (lines_C[middle_line_C][0] - 5 + pk, 0), (lines_C[middle_line_C][0] - 5 + pk, 480),
                         (0, 0, 255), 2)
                cv.line(imgrawC, (lines_C[middle_line_C][1] + 5 + pk, 0), (lines_C[middle_line_C][1] + 5 + pk, 480),
                         (0, 0, 255), 2)
            else:
                cv.line(imgrawC, (lines_C[middle_line_C][0] - 5 + pk, 0), (lines_C[middle_line_C][0] - 5 + pk, 480),
                         (255, 0, 0), 2)
                cv.line(imgrawC, (lines_C[middle_line_C][1] + 5 + pk, 0), (lines_C[middle_line_C][1] + 5 + pk, 480),
                         (255, 0, 0), 2)
        cv.imshow("imgrawC", imgrawC)

    prepare_report = False

    if prepare_report:
        fig, axs = plt.subplots(2, 3, figsize=(19.5,9.5), sharex=True)
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.07, top=0.97)

        imgplot = axs[0,1].imshow(cv.cvtColor(imgrawC, cv.COLOR_BGR2RGB))
        imgplot = axs[1,1].imshow(cv.cvtColor(imgrawO, cv.COLOR_BGR2RGB))
        imgplot = axs[0,2].imshow(th3_otsuC, cmap="gray")
        imgplot = axs[1,2].imshow(th3_otsuO, cmap="gray")

        if len(lines_C) > 0:
            axs[0,0].plot((tresholdC + 1) / 2)
            if black_lines_C:
                axs[0,0].plot(1 - (result_C * result_C.max()), label='result_C')
            else:
                axs[0, 0].plot((result_C * result_C.max()), label='result_C')
        axs[0,0].plot(histC / histC_10.max(), label='histC_10')
        axs[0,0].plot(histC_10 / histC_10.max(), label='histC_10')
    #    axs[0,0].legend()
    #    plt.xlim(0, 640)

        if len(lines_O) > 0:
            axs[1,0].plot((tresholdO + 1) / 2)
            if black_lines_O:
                axs[1, 0].plot(1 - (result_O * result_O.max()), label='result_O')
            else:
               axs[1, 0].plot(result_O * result_O.max(), label='result_O')
        axs[1,0].plot(histO / histO_10.max(), label='histC_10')
        axs[1,0].plot(histO_10 / histO_10.max(), label='histO_10')
    #    axs[1,0].legend()
        plt.xlim(0, 640)

        plt.savefig('data.png')
#        plt.show()

    prepare_charts = False

    if prepare_charts:
        plt.xlim(0, 640)
        if black_lines:
            if len(lines_C) > 0:
                plt.plot((tresholdC + 1) / 2)
                plt.plot(1 - (result_C * result_C.max()), label='result_C')
            plt.plot(histC_10 / histC_10.max(), label='histC_10')
        else:
            if len(lines_O) > 0:
                plt.plot((tresholdO + 1) / 2)
                plt.plot(result_O * result_O.max(), label='result_O')
            plt.plot(histO_10 / histO_10.max(), label='histO_10')
        plt.legend()

        plt.show()

    key = cv.waitKey(200)

    if key == ord('s') and prepare_report:
        name = fname[:-4]
        nfname = "report/" + name + "_fig.png"

        try:
            os.rename("data.png", nfname)
        except:
            print("something wrong... ")

        pictures_separately = False

        if pictures_separately:
            v_img1 = cv.vconcat([imgraw, cv.cvtColor(imggray, cv.COLOR_GRAY2BGR)])
            v_img2 = cv.vconcat([morphC, morphO])
            v_img3 = cv.vconcat([fil2dC, fil2dO])
            v_img4 = cv.vconcat([th3_otsuC, th3_otsuO])
            v_img5 = cv.vconcat([imgrawC, imgrawO])

            nfname = "report/" + name + "_img1.png"
            cv.imwrite(nfname, v_img1)
            nfname = "report/" + name + "_img2.png"
            cv.imwrite(nfname, v_img2)
            nfname = "report/" + name + "_img3.png"
            cv.imwrite(nfname, v_img3)
            nfname = "report/" + name + "_img4.png"
            cv.imwrite(nfname, v_img4)
            nfname = "report/" + name + "_img5.png"
            cv.imwrite(nfname, v_img5)

#    plt.close()

    # saving the detection and manual pointing comparison
    if len(final_lines_list)>0:
        if manual_position:
            f = open("detekcja_analitic.txt", "a")
            f.write(fname + " " + format(final_lines_list[0], '.1f') + " " + format(final_lines_list[1], '.1f') + " " + format(final_lines_list[2], '.1f')
                    + " " + format(manual_position[0], '.1f') + " " + format(manual_position[1], '.1f') + " " + format(manual_position[2], '.1f')
                    + " " + format(final_lines_list[2] - manual_position[2], '.1f') + "\n")
            f.close()
        return final_lines_list
    else:
        return None


######################################################
# main
######################################################
if __name__ == '__main__':

    edge_ignore_width = 100
    max_line_width = 75  # depends on the scale of the image
    only_middle_line = True

    #dir_path = r'C:\Users\maran\Documents\images\opisane'
    #dir_path = r'C:\Users\maran\Documents\selected'
    dir_path = './selected'

    test_files = create_a_list_of_test_files_in_the_directory(dir_path)

    for fname in test_files:
        print(fname)
        tname = os.path.join(dir_path, fname)
        imgraw = cv.imread(tname)

        if imgraw is None:
            ret = False
            break
        else:
            ret = True

        if ret:
            lines = lines_detector(imgraw, edge_ignore_width, max_line_width, only_middle_line, fname)
