from app import app
from flask import request, make_response, Response
from flask import jsonify
from app.controllers.face_recognition import VGGFaceRecognizer
import cv2
import base64
import numpy as np
from mtcnn import MTCNN
import os
from PIL import Image
import logging

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


@app.route('/api/face/recognizer', methods=['POST', 'GET'])
def recognizer():
    if request.method == "POST":
        imgb64 = request.data['image']
        image = np.fromstring(base64.b64decode(imgb64), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        detected_faces = mtcnn.detect_faces(image)

        height, width, _ = image.shape

        if len(detected_faces) > 1:
            return Response(status=409)
        else:
            x1, y1, x2, y2 = VGGFaceRecognizer.fix_coordinates(detected_faces[0]['box'], width, height)
            cropped_face = image[y1:y2, x1:x2]

            cropped_face = Image.fromarray(cropped_face)

            face_name = face_recognizer.recognize(
                cropped_face, thresh=0.30)

            return make_response(jsonify({"name": face_name}), 200)

    return {"Responde": "Face Recognition API"}


@app.route('/api/face/add', methods=['POST', 'GET'])
def add_face():
    if request.method == "POST":
        imgb64 = request.data['image']
        person_name = request.data["name"]
        image = np.fromstring(base64.b64decode(imgb64), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        detected_faces = mtcnn.detect_faces(image)

        height, width, _ = image.shape

        face_list = list()
        label_list = list()

        if len(detected_faces) > 1:
            return Response(status=409)
        else:
            x1, y1, x2, y2 = VGGFaceRecognizer.fix_coordinates(detected_faces[0]['box'], width, height)
            cropped_face = image[y1:y2, x1:x2]

            cropped_face = Image.fromarray(cropped_face)

            face_list.append(cropped_face)
            label_list.append(person_name)

        face_recognizer.add_faces(face_list, label_list)

        return Response(status=201)

    return {"Responde": "Face Recognition API"}
