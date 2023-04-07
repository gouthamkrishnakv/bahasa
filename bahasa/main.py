from os import error
from cv2 import VideoCapture
import numpy
import cv2

def main():
    vcap: VideoCapture = cv2.VideoCapture(0)

    if not vcap.isOpened():
        print("Error in finding camera")
        exit(1)
    
    while True:
        ret, frame = vcap.read()

        if not ret:
            print(f"Can't recieve from camera (stream end?). Exiting")
            exit(1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the grayscale
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) == ord('q'):
            break
    
    vcap.release()
    cv2.destroyAllWindows()
