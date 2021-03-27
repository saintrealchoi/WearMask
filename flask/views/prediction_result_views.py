from flask import Blueprint,render_template

bp = Blueprint('prediction_result', __name__, url_prefix='/prediction')

@bp.route('/detail')
def index(image):
    img = image
    return render_template('prediction_result.html')