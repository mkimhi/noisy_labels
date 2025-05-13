from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from pycocotools import mask as maskUtils
import numpy as np
import cv2


class COCOAnnsPairsDataset(Dataset):
    def __init__(self, clean_coco_dataset_f_path: Path, noisy_coco_dataset_f_path: Path):
        self.clean_coco_dataset = COCO(clean_coco_dataset_f_path)
        self.noisy_coco_dataset = COCO(noisy_coco_dataset_f_path)
        self.anns_ids_list = np.array(list(map(int, self.clean_coco_dataset.anns.keys())))

    def __len__(self):
        return len(self.clean_coco_dataset.anns)

    def __getitem__(self, idx: int):
        ann_id = self.anns_ids_list[idx]
        clean_polygons = self.get_ann_from_coco_data_ann(self.clean_coco_dataset.anns[ann_id]['segmentation'])
        noisy_polygons = self.get_ann_from_coco_data_ann(self.noisy_coco_dataset.anns[ann_id]['segmentation'])
        return clean_polygons, noisy_polygons

    @classmethod
    def get_ann_from_coco_data_ann(cls, coco_data_ann) -> Polygon:
        if isinstance(coco_data_ann, list) and all((isinstance(sublist, list) for sublist in coco_data_ann)):
            polygons_list = list(map(cls.get_polygon_from_interchanging_xy_arr, coco_data_ann))
        elif isinstance(coco_data_ann, dict) and 'counts' in coco_data_ann.keys():
            try:
                binary_mask = maskUtils.decode(coco_data_ann).astype(np.uint8)
            except TypeError:
                rle_fixed = maskUtils.frPyObjects(coco_data_ann, coco_data_ann['size'][0], coco_data_ann['size'][1])
                binary_mask = maskUtils.decode(rle_fixed).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons_list = [Polygon(c[:, 0, :]) for c in contours if len(c) >= 3]
        else:
            raise NotImplementedError()
        return polygons_list


    @staticmethod
    def get_polygon_from_interchanging_xy_arr(interchanging_xy_arr) -> Polygon:
        interchanging_xy_arr = np.array(interchanging_xy_arr)
        num_of_points = len(interchanging_xy_arr)
        xy_arr = np.vstack((
            interchanging_xy_arr[range(0, num_of_points, 2)],
            interchanging_xy_arr[range(1, num_of_points, 2)]
        ))
        polygon = Polygon(xy_arr.T)
        return polygon
