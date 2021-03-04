from cv2 import imread
from cv2 import CascadeClassifier
import cv2

# load the pre-trained model
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
pixels = imread('test2.jpg')

# perform face detection
bboxes = classifier.detectMultiScale(pixels)
# print bounding box for each detected face
for box in bboxes:
    print(box)

# perform face detection
bboxes = classifier.detectMultiScale(pixels)
# print bounding box for each detected face
for box in bboxes:
    print(box)

    x, y, width, height = box
    x2, y2 = x + width, y + height
    # draw a rectangle over the pixels
    cv2.rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)

# show the image
cv2.imshow('face detection', pixels)
# keep the window open until we press a key
cv2.waitKey(0)
# close the window
cv2.destroyAllWindows()

