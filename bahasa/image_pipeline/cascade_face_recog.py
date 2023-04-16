from .base import ImagePipeline, np
import cv2

class CascadeFaceRecogPipeline(ImagePipeline):
    face_cascade_classifier = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

    def __init__(self):
        pass

    def run(self, frame: np.array) -> np.array:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = CascadeFaceRecogPipeline.face_cascade_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        curframe = frame
        for (x, y, w, h) in faces:
            curframe = cv2.rectangle(curframe, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return curframe
