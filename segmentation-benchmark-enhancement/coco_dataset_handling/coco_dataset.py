from pathlib import Path

from pycocotools.coco import COCO
from torch.utils.data import Dataset


class COCOPairsDataset(Dataset):
    def __init__(self, coco_dataset_f_path: Path):
        data_raw_dictionary = COCO(coco_dataset_f_path)
