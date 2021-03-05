from cv2 import imread
from cv2 import CascadeClassifier
import cv2
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed,random
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
import numpy as np
import cvlib as cv

dataset_home = 'images/images/'

captured_num = 0

for file in os.listdir(dataset_home):
    src = dataset_home + file
    # load the pre-trained model
    classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
    print(src)
    pixels = cv2.imread(src)
    print(pixels)
    # perform face detection
    bboxes = classifier.detectMultiScale(pixels)

    for box in bboxes:
        print(box)

        x, y, width, height = box
        x2, y2 = x + width, y + height
        # draw a rectangle over the pixels
        captured_num = captured_num + 1
        face_in_img = pixels[y:y2, x:x2,:]
        cv2.imwrite("C:\\Users\\LG\\Desktop\\newface\\" + str(captured_num) + '.jpg', face_in_img)  # 마스크 착용데이터
        captured_num = captured_num + 1