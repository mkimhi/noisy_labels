from typing import Literal


class ExpParams:
    def __init__(self, max_epochs: int, num_workers: int,
                 models_wrappers_package: Literal['mmdet', 'detectron2'],
                 noise_name: str):
        self.max_epochs = max_epochs
        self.num_workers = num_workers
        self.models_wrappers_package = models_wrappers_package
        self.noise_name = noise_name
