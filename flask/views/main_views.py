from flask import Blueprint, url_for, render_template, flash, request, session,g
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import redirect

from wearmask import db
from wearmask.forms import UserCreateForm,UserLoginForm
from wearmask.models import User

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def index():
    return render_template('home.html')