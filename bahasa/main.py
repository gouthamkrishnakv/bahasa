from math import floor
import time
from typing import List, Protocol
import cv2
import numpy as np
import torch

assert torch.cuda.is_available(), "Cuda is not available"
print("Cuda is available")

class ImagePipeline(Protocol):
    def run(self, frame: np.array) -> np.array:
        ...

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


class NNFaceRecogPipeline(ImagePipeline):
    nn_model = torch.hub.load('./yolov5', 'custom', source='local', path='models/nn_exp/weights/best.pt')

    def __init__(self) -> None:
        pass

    def run(self, frame: np.array) -> np.array:
        result = NNFaceRecogPipeline.nn_model(frame)
        print(str(f" FACES: {result.xyxyn[0].shape[0]}"), end="\r")
        return result.render()[0]

class VCDevice:
    vcIndex: int
    vcDevice: cv2.VideoCapture
    pipelines: List[ImagePipeline]
    latencies: np.ndarray
    lastLatency: float
    lastTimeTaken: int = 0

    def __init__(self, vcIndex: int = 0):
        self.vcIndex = vcIndex
        self.vcDevice = cv2.VideoCapture(0)
        if not self.vcDevice.isOpened():
            raise OSError(1, "Error in finding camera")
        self.pipelines = []
        self.latencies = np.array([], dtype='float64')
        self.lastTimeTaken = 0
        self.lastLatency = 0.0
        # Add face recognition pipeline
        # self.pipelines.append(CascadeFaceRecogPipeline())
        self.pipelines.append(NNFaceRecogPipeline())

    def run_pipeline(self, frame: np.array):
        # Image processing pipeline goes here
        for pipeline in self.pipelines:
            frame = pipeline.run(frame)
        return frame
    
    def run(self):
        isErrored, errorVal = False, None
        while True:
            # Read a frame
            ret, frame = self.vcDevice.read()
            # Check if frame exists
            if not ret:
                isErrored, errorVal = True, "Can't recieve from camera!"
                break
            ## PERF
            start_time = time.perf_counter()
            # Process the frame from the pipeline
            final_frame = self.run_pipeline(frame)
            ## PERF
            end_time = time.perf_counter()
            # Calculate latencies for the pipeline to run
            self.latencies = np.append(self.latencies, end_time - start_time)
            print("\rLATENCY: {:06.3f}s".format(np.average(self.lastLatency)), end="")
            if (processed_time := floor(time.process_time())) > self.lastTimeTaken:
                self.lastLatency = np.average(self.latencies)
                self.lastTimeTaken = processed_time
                self.latencies = np.array([], dtype='float64')
            # Show the frame
            cv2.imshow('frame', final_frame)
            # Wait 1ms for a key clicked
            if cv2.waitKey(1) == ord('q'):
                break
        self.vcDevice.release()
        cv2.destroyAllWindows()

        if isErrored:
            raise OSError(1, errorVal)


def main():
    # Create the video capture device
    dev = VCDevice()
    # Run the device
    dev.run()