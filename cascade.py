import numpy as np
import cv2
import cvlib as cv
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image

model = load_model('model_final.h5')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(gray, 1.3, 5)

    # for (x, y, w, h) in faces:
    #     img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #     roi_gray = gray[y:y + h, x:x + w]
    #     roi_color = img[y:y + h, x:x + w]
    #     eyes = eye_cascade.detectMultiScale(roi_gray)
    #     for (ex, ey, ew, eh) in eyes:
    #         cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # 얼굴 검출
    # face, confidence = cv.detect_face(frame)

    # loop through detected faces
    for idx, f in enumerate(face):

        (startX, startY) = f[0], f[1]
        (endX, endY) = f[0]+f[2], f[1]+f[3]

        if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[
            0] and 0 <= endY <= frame.shape[0]:

            face_region = frame[startY:endY, startX:endX]

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