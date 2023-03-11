from flask_api import FlaskAPI
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = FlaskAPI(__name__)
app.config.from_object('config')
app.config['SQLALCHEMY_DATABASE_URI'] = " "

db = SQLAlchemy(app)
migrate = Migrate(app, db)

from app.models import person
from app.routers import default