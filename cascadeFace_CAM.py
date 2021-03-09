# import necessary packages
import cvlib as cv
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image
from cv2 import CascadeClassifier

model = load_model('model_final.h5')

# open webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

while webcam.isOpened():

    status, frame = webcam.read()

    if not status:
        print("Could not read frame")
        exit()

    # 얼굴 검출
    face, confidence = cv.detect_face(frame)

    classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
    bboxes = classifier.detectMultiScale(frame,1.3,5)


    # loop through detected faces
    for idx, f in enumerate(bboxes):

        (startX, startY ,endX, endY) = f
        endX += startX
        endY += startY

        if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[
            0] and 0 <= endY <= frame.shape[0]:

            face_region = frame[startY:endY, startX:endX]
            print(face_region)
            # preprocessing face
            re_face_region = cv2.resize(face_region, (224, 224), interpolation=cv2.INTER_AREA)
            x = img_to_array(re_face_region)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            prediction = model.predict(x)

            if prediction < 0.5:  # 50퍼센트 미만의 softmax score -> 미착용
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "No Mask ({:.2f}%)".format((1 - prediction[0][0]) * 100)
                cv2.putText(frame, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:  # 50퍼센트 이상의 softmax score -> 착용
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "Mask ({:.2f}%)".format(prediction[0][0] * 100)
                cv2.putText(frame, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # display output
    cv2.imshow("mask nomask classify", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()