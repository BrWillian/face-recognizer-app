from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
from PIL import Image
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class VGGFaceRecognizer:
    def __init__(self, model='senet50'):
        self.model = model
        self.face_dict = dict()
        self.recognizer = VGGFace(include_top=False, model=model)

    @staticmethod
    def calculate_similarity(vector_1, vector_2):
        vector_1 = np.squeeze(vector_1)
        vector_2 = np.squeeze(vector_2)

        return cosine(vector_1, vector_2)

    def register_faces(self, face_list, name_list):
        self.face_dict = dict()
        for face, name in zip(face_list, name_list):
            self.face_dict[name] = self.feature_extractor(face)

        return self.face_dict

    def add_faces(self, face_list, name_list):
        for face, name in zip(face_list, name_list):
            self.face_dict[name] = self.feature_extractor(face)

        return self.face_dict

    def feature_extractor(self, face):
        face = face.resize((224, 224), Image.ANTIALIAS)
        face = np.asarray(face).astype(np.float64)
        face = np.expand_dims(face, axis=0)

        face = preprocess_input(face, version=2)

        return self.recognizer.predict(face)

    def recognize(self, face: np.ndarray, list_of_faces: dict, thresh: float = 0.25):
        query_features = self.feature_extractor(face)
        temp_sim_dict = dict()

        for key, value in list_of_faces.items():
            db_face_features = np.array(value)
            score = self.calculate_similarity(
                db_face_features.squeeze(), query_features
            )
            temp_sim_dict[key] = score
            print(score)

        try:
            if min(temp_sim_dict.values()) > thresh:
                return None
        except:
            return None

        most_similar_face = min(temp_sim_dict, key=temp_sim_dict.get)

        return most_similar_face

    @staticmethod
    def fix_coordinates(box: list, width: int, height: int):
        x1, y1, w, h = box
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(w, width) + x1
        y2 = min(h, height) + y1
        return x1, y1, x2, y2
