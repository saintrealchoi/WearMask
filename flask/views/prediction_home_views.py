# import the necessary packages
from flask import Blueprint,render_template,request
from werkzeug.utils import secure_filename

model = None

bp = Blueprint('prediction_home', __name__, url_prefix='/')




@bp.route('/prediction')
def index():
    return render_template('prediction.html')

@bp.route('/fileUpload',methods = ['GET','POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        return render_template('prediction_result.html', file = f)


@bp.route("/predict", methods=['GET','POST'])
def predict():

	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}


	# ensure an image was properly uploaded to our endpoint
	if request.method == "POST":
		if request.files.get('file'):
			# read the image in PIL format
			image = request.files['file'].read()
			image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
			image = prepare_image(image, target=(224, 224))

			# classify the input image and then initialize the list
			# of predictions to return to the client
			preds = model.predict(image)
			results = np.argmax(preds)
			preds = preds[0][results]
			# results = imagenet_utils.decode_predictions(preds)
			data["predictions"] = []

			# loop over the results and add them to the list of
			# returned predictions
			r = {"label" : int(results), "probability":float(preds)}
			data["predictions"].append(r)
			#
			# for (imagenetID, label, prob) in results[0]:
			# 	r = {"label": label, "probability": float(prob)}
			# 	data["predictions"].append(r)

			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return render_template('prediction_result.html', data = flask.jsonify(data))

if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_modelh5()
	bp.run()