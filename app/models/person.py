from sqlalchemy import Column, Integer, String, Boolean, Date, DateTime
import pickle as pkl
from app import db
from datetime import datetime


class Person(db.Model):
    __tablename__ = "person"
    __table_args__ = {'extend_existing': True}
    cd_person = Column("cd_person", Integer, primary_key=True, autoincrement=True)
    name = Column("name", String)
    birthday = Column("birthday", Date)
    sex = Column("sex", String)
    phone = Column("phone", String)
    email = Column("email", String)
    instagram = Column("instagram", String)
    confirmed = Column("confirmed", Boolean)
    image_path = Column("path", String)
    face_attributes = Column("face_attributes", db.PickleType)
    dt_confirmed = Column("dt_confirmed", DateTime)
    dt_created = Column("dt_created", DateTime)
    dt_updated = Column("dt_updated", DateTime)

    def __init__(self, name, birthday, sex, phone, email, instagram, image_path, image):
        self.name = name
        self.birthday = birthday
        self.sex = sex
        self.phone = phone
        self.email = email
        self.instagram = instagram
        self.image_path = image_path
        self.face_attributes = pkl.dumps(image)
        self.confirmed = False
        self.dt_confirmed = datetime.now()
        self.dt_created = datetime.now()
