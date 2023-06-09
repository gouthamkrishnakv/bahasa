from math import floor
import time
from typing import List
import cv2
import numpy as np

from .image_pipeline.base import ImagePipeline
from .image_pipeline import nn_face_recog, cascade_face_recog


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
        self.latencies = np.array([], dtype="float64")
        self.lastTimeTaken = 0
        self.lastLatency = 0.0
        # Add face recognition pipeline
        # self.pipelines.append(cascade_face_recog.CascadeFaceRecogPipeline())
        # Add neural network pipeline
        self.pipelines.append(nn_face_recog.NNFaceRecogPipeline())

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
                self.latencies = np.array([], dtype="float64")
            # Show the frame
            cv2.imshow("frame", final_frame)
            # Wait 1ms for a key clicked
            if cv2.waitKey(1) == ord("q"):
                break
        self.vcDevice.release()
        cv2.destroyAllWindows()

        if isErrored:
            raise OSError(1, errorVal)
