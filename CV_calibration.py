import cv2 as cv
import numpy as np
import os
import re
import matplotlib.pyplot as plt


def find_the_difference(base, curr, name):
    # convert to grayscale
    base_gray = cv.cvtColor(base, cv.COLOR_BGR2GRAY)
    curr_gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)

    # Applying the function
    base_orb = cv.ORB_create(nfeatures=1000)
    base_kp, baseDescriptors = base_orb.detectAndCompute(base_gray, None)
    curr_orb = cv.ORB_create(nfeatures=1000)
    curr_kp, currDescriptors = curr_orb.detectAndCompute(curr_gray, None)

    # Drawing the keypoints
    kp_image = cv.drawKeypoints(curr, curr_kp, None, color=(0, 255, 0), flags=0)

    cv.imshow('ORB', kp_image)
#    cv.imwrite(name, kp_image)
    cv.waitKey(300)

    # Initialize the Matcher for matching the keypoints and then match the keypoints
    matcher = cv.BFMatcher()
    matches = matcher.match(baseDescriptors, currDescriptors)

    # finding the humming distance of the matches and sorting them
    dmatches = sorted(matches, key=lambda x: x.distance)

    # extract the matched keypoints
    bf_final = np.float32([base_kp[m.queryIdx].pt for m in dmatches]).reshape(-1, 1, 2)
    cf_final = np.float32([curr_kp[m.trainIdx].pt for m in dmatches]).reshape(-1, 1, 2)

    # find perspective transformation using the arrays of corresponding points
    transformation, hom_stati = cv.findHomography(cf_final, bf_final, method=cv.RANSAC, ransacReprojThreshold=5.0)

    return transformation


######################################################
# main
######################################################

# positions measured by the interferometer
xint = []
f = open("../calibration_images/interferometr.txt", "r")
for x in f:
    xint.append(float(x))
print(len(xint))


# folder path
dir_path = r'calibration_images'
# files list
tkreski = []
# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        # check only text files
        if path.endswith('.png'):
            tkreski.append(path)
print(len(tkreski))

# read position measured manually from file name
kreski = []
for i in range(len(tkreski)):
    #  print(i, res[i])
    t = re.split('[_.]', tkreski[i])
    k = [i, float(t[2]), float(t[3]), (float(t[2]) + float(t[3])) / 2.]
    #  print(i, float(t[2]), float(t[3]))
    kreski.append(k)

base_name = 'calibration_images' + '/' + tkreski[0]
base = cv.imread(base_name)

ydet = []
ydet.append(0.0)
i = 1
for image in tkreski[1:]:
    curr_name = 'calibration_images' + '/' + image
    orb_name = 'test_orb' + '/' + image  #if we want to save them
    curr = cv.imread(curr_name)
    #  print(image)
    trans = find_the_difference(base, curr, orb_name)
    print(i, -(kreski[i][3] - kreski[0][3]), trans[0][2])
    ydet.append(trans[0][2])
    i += 1

y = []
for kreska in kreski:
    y.append(kreska[3] - kreski[0][3])

xdetf = []
ydetf = []
for i in range(len(kreski)):
    xdetf.append(xint[i])
    ydetf.append(-ydet[i])

#  dopasowanie do recznych pomiarow
poly, residuals, rank, singular_values, rcond = np.polyfit(y, xint, deg=1, full=True)
yint = np.polyval(poly, y)
print("manual:",poly, residuals, rank, singular_values, rcond)

# dopasowanie do automatycznych pomiarow
polydet, residuals, rank, singular_values, rcond = np.polyfit(ydetf, xdetf, deg=1, full=True)
ydetint = np.polyval(polydet, ydetf)
print("detected:",polydet, residuals, rank, singular_values, rcond)

y_s = y.copy()
errt1 = yint.copy()
errt2 = yint.copy()
ydetf_s = ydetf.copy()
for i in range(len(y_s)):
    y_s[i] = y[i] + 340
    ydetf_s[i] = ydetf[i] + 340
    errt1[i] = (yint[i] - xint[i]) / poly[0]
    errt2[i] = (ydetint[i] - xdetf[i]) / poly[0]
print("std manual detected:", np.std(errt1), np.std(errt2))

plt.scatter(y_s, (yint - xint) / poly[0], label='manual')
plt.scatter(ydetf_s, (ydetint - xdetf) / poly[0], label='detected')
plt.legend()
plt.xlim([0, 640])
plt.show()
