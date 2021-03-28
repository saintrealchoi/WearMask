from flask import Blueprint, url_for, render_template, flash, request, session,g,Response
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import redirect

from wearmask import db
from wearmask.forms import UserCreateForm,UserLoginForm
from wearmask.models import User

from keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image
import cvlib as cv
import cv2
import numpy as np
from camera import Camera

bp = Blueprint('webcam', __name__, url_prefix='/')

model = None
video = None
def load_modelh5():
	global model
	model = load_model('wearmask/model.h5')



def gen(video):
    while True:
        success, image = video.read()

        if not success:
            print("Could not read frame")
            exit()

            # 얼굴 검출
        face, confidence = cv.detect_face(image)

        # loop through detected faces
        for idx, f in enumerate(face):

            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
            
            if 0 <= startX <= image.shape[1] and 0 <= endX <= image.shape[1] and 0 <= startY <= image.shape[
                0] and 0 <= endY <= image.shape[0]:

                face_region = image[startY:endY, startX:endX]

                # preprocessing face
                re_face_region = cv2.resize(face_region, (224, 224), interpolation=cv2.INTER_AREA)
                x = img_to_array(re_face_region)
                x = np.expand_dims(x, axis=0)

                prediction = model.predict(x)

                if np.argmax(prediction) == 0:
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    Y = startY - 10 if startY - 10 > 10 else startY + 10
                    text = "No Mask ({:.2f}%)".format((prediction[0][0]) * 100)
                    cv2.putText(image, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                else:
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    Y = startY - 10 if startY - 10 > 10 else startY + 10
                    text = "Mask ({:.2f}%)".format(prediction[0][1] * 100)
                    cv2.putText(image, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
@bp.route('/webcam')
def index():
    model = load_modelh5()
    global video
    return render_template('webcam.html')

@bp.route('/video_feed')
def video_feed():
    video = cv2.VideoCapture(0)
    return Response(gen(video), mimetype='multipart/x-mixed-replace; boundary=frame')

@bp.route('/video_closed')
def video_closed():
    # video.release()
    return redirect(url_for('main.index'))


@bp.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')
    if user_id is None:
        g.user = None
    else:
        g.user = User.query.get(user_id)