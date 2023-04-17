import numpy as np
from typing import Protocol


class ImagePipeline(Protocol):
    def run(self, frame: np.array) -> np.array:
        ...
