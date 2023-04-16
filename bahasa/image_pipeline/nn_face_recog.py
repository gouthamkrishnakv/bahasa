from .base import ImagePipeline, np
import torch

assert torch.cuda.is_available(), "Cuda is not available"
print("Cuda is available")


class NNFaceRecogPipeline(ImagePipeline):
    nn_model = torch.hub.load('./yolov5', 'custom', source='local', path='models/nn_exp/weights/best.pt')

    def __init__(self) -> None:
        pass

    def run(self, frame: np.array) -> np.array:
        result = NNFaceRecogPipeline.nn_model(frame)
        print(str(f" FACES: {result.xyxyn[0].shape[0]}"), end="\r")
        return result.render()[0]
