from cv2 import VideoCapture
import cv2
import numpy as np

class VCDevice:
    vcIndex: int
    vcDevice: VideoCapture

    def __init__(self, vcIndex: int = 0):
        self.vcIndex = vcIndex
        self.vcDevice = cv2.VideoCapture(0)
        if not self.vcDevice.isOpened():
            raise OSError(1, "Error in finding camera")

    def run_pipeline(self, frame: np.array) -> np.array:
        # Image processing pipeline goes here
        return frame
    
    def run(self):
        isErrored, errorVal = False, None
        while True:
            # Read a frame
            ret, frame = self.vcDevice.read()
            # Check if frame exists
            if not ret:
                isErrored, errorVal = True, "Can't recieve from camera!"
            # Process the frame from the pipeline
            processed_frame = self.run_pipeline(frame)
            # Show the frame
            cv2.imshow('frame', processed_frame)
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