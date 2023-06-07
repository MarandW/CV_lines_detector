import cv2 as cv
import numpy as np
import os
import re
import glob
from tensorflow.lite.python.interpreter import Interpreter
import matplotlib.pyplot as plt
# tensorflow 2.8.0
# Downgrade the protobuf package to 3.20.x or lower.


# Script to run custom TFLite model on test images to detect objects
# Source: https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/TFLite_detection_image.py


def measurements_from_filename(name):
    t = re.split('[_.]', name)
    if t[0] == "kreska":
        measurement_time, manual_position = t[1], (float(t[2]), float(t[3]), (float(t[2]) + float(t[3])) / 2.)
#        print(measurement_time, manual_position)
        return (measurement_time, manual_position)
    else:
        return None, None


def create_a_list_of_test_files_in_the_directory(dir_path):

    images_list = []

    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            # check only text files
            if path.endswith('.png'):
                images_list.append(path)
#                print(path)
    print("total: ", len(images_list), "images")

    return images_list


# Define function for inferencing with TFLite model and displaying results
def tflite_detect_images(modelpath, imgpath, lblpath, min_conf=0.5, savepath="C:/Users/maran/Documents/images/output",
                         txt_only=False):

    # Grab filenames of all images in test folder
    images = glob.glob(imgpath + '/*.png')
    print(f for f in os.listdir(imgpath) if f.endswith('.png'))
    image_list = create_a_list_of_test_files_in_the_directory(imgpath)

    # Load the label map into memory
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the Tensorflow Lite model into memory
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    print(width, height)

    float_input = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    images_to_test = images

    # Loop over every image and perform detection
    for image_path, fname in zip(images_to_test, image_list):

        measurement_time, manual_position = measurements_from_filename(fname)
        print(image_path, fname, measurement_time, manual_position)

        # Load image and resize to expected shape [1xHxWx3]
        image = cv.imread(image_path)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        imH, imW, _ = image.shape
        image_resized = cv.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence of detected objects

        detections = []

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions,
                # need to force them to be within image using max() and min()
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

#                print(i, xmin, xmax)

                cv.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])]  # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
                labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
                cv.rectangle(image, (xmin, label_ymin - labelSize[1] - 10),
                             (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                             cv.FILLED)  # Draw white box to put label text in
                cv.putText(image, label, (xmin, label_ymin - 7), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                           2)  # Draw label text

                detections.append([object_name, scores[i], xmin, ymin, xmax, ymax, (xmin+xmax)/2])

        # only middle line
        distmin = 2000  # somthing big
        halfwidth = 640/2
        if len(detections) > 0:
            imin = 0
            for i in range(len(detections)):
                if abs(detections[i][6] - halfwidth) < distmin:
                    distmin = abs(detections[i][6] - halfwidth)
                    imin = i
            print(detections[imin])
            cv.line(image, (int(detections[imin][6]), 0), (int(detections[imin][6]), 480), (0, 0, 255), 3, cv.LINE_AA)

            if manual_position:
                f = open("detekcja_tf.txt", "a")
                f.write(fname + " " + format(detections[imin][2], '.1f') + " " + format(detections[imin][4], '.1f')
                        + " " + format(detections[imin][6], '.1f')
                        + " " + format(manual_position[0], '.1f') + " " + format(manual_position[1], '.1f')
                        + " " + format(manual_position[2], '.1f')
                        + " " + format(detections[imin][6] - manual_position[2], '.1f') + "\n")
                f.close()

        # All the results have been drawn on the image, now display the image
        if txt_only == False:
            # "text_only" controls whether we want to display the image results or just save them in .txt files

            # set the timer interval 5000 milliseconds
            fig = plt.figure()
            ax = fig.add_subplot(111)
            timer = fig.canvas.new_timer(interval=1000)
            timer.add_callback(plt.close)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#            plt.figure(figsize=(12, 16))
            ax.imshow(image)
            ax.plot([0, 0], [100, 100], color='blue', linewidth=2)
            timer.start()
            plt.show()

        # Save detection results in .txt files (for calculating mAP)
        elif txt_only == True:

            # Get filenames and paths
            image_fn = os.path.basename(image_path)
            base_fn, ext = os.path.splitext(image_fn)
            txt_result_fn = base_fn + '.txt'
            txt_savepath = os.path.join(savepath, txt_result_fn)

            # Write results to text file
            # (Using format defined by https://github.com/Cartucho/mAP, which will make it easy to calculate mAP)
            with open(txt_savepath, 'w') as f:
                for detection in detections:
                    f.write('%s %.4f %d %d %d %d\n' % (
                        detection[0], detection[1], detection[2], detection[3], detection[4], detection[5]))

    return


if __name__ == '__main__':

    # Set up variables for running user's model
    # Path to test images folder
    PATH_TO_IMAGES = './selected'
    #PATH_TO_IMAGES = 'C:/Users/maran/Documents/images/opisane'
    #PATH_TO_IMAGES = 'C:/Users/maran/Documents/selected'
    # Path to .tflite model file
    PATH_TO_MODEL = './custom_model_lite/detect.tflite'
    # Path to labelmap.txt file
    PATH_TO_LABELS = './custom_model_lite/labelmap.txt'
    # Confidence threshold (try changing this to 0.01 if you don't see any detection results)
    min_conf_threshold = 0.2

    # Run inferencing function!
    tflite_detect_images(PATH_TO_MODEL, PATH_TO_IMAGES, PATH_TO_LABELS, min_conf_threshold)
