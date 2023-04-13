from app import app
from flask import request, make_response, Response
from flask import jsonify
from app.controllers.face_recognition import VGGFaceRecognizer
from app.models.person import Person
import cv2
import base64
import numpy as np
from mtcnn import MTCNN
import os
from PIL import Image
import logging
import pickle

LOGGER = logging.getLogger(__name__)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
mtcnn = MTCNN()
face_recognizer = VGGFaceRecognizer(model='senet50')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)


@app.route('/api/face/recognizer', methods=['POST'])
def recognizer():
    if request.method == "POST":
        imgb64 = request.data['image']
        image = np.fromstring(base64.b64decode(imgb64), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        detected_faces = mtcnn.detect_faces(image)

        height, width, _ = image.shape

        try:
            face = max(detected_faces, key=lambda detected_face: detected_face['confidence'])
            x1, y1, x2, y2 = VGGFaceRecognizer.fix_coordinates(face['box'], width, height)
            cropped_face = image[y1:y2, x1:x2]

            cropped_face = Image.fromarray(cropped_face)

            db_faces = Person.query.all()

            list_of_faces = {face.cd_person: pickle.loads(face.face_attributes) for face in db_faces}

            face_name = face_recognizer.recognize(
                cropped_face, list_of_faces, thresh=0.35)

            return make_response(jsonify({"face_id": face_name}), 200) if face_name else Response(status=400)
        except Exception as e:
            LOGGER.error(e)
            return Response(status=400)


@app.route('/api/face/get_attributes', methods=['POST'])
def add_face():
    if request.method == "POST":
        imgb64 = request.data['image']
        image = np.fromstring(base64.b64decode(imgb64), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        detected_faces = mtcnn.detect_faces(image)

        height, width, _ = image.shape

        try:
            face = max(detected_faces, key=lambda detected_face: detected_face['confidence'])
            x1, y1, x2, y2 = VGGFaceRecognizer.fix_coordinates(face['box'], width, height)
            cropped_face = image[y1:y2, x1:x2]

            cropped_face = Image.fromarray(cropped_face)

            features = face_recognizer.feature_extractor(cropped_face)

            return make_response(jsonify({"face_attributes": features.squeeze().tolist()}), 200)

        except Exception as e:
            LOGGER.error(e)
            return Response(status=400)
