import json
from pathlib import Path
from coco_dataset_handling.coco_pairs_dataset import COCOAnnsPairsDataset
from common.constants import RAW_DATA_DIR_PATH, PROCESSED_DATA_DIR_PATH
from shapely.ops import unary_union
from tqdm import tqdm
import matplotlib.pyplot as plt


def meas_segmentation_error(clean_coco_ann_f_path: Path, noisy_coco_ann_f_path: Path, images_dir_path: Path):
    coco_pair_dataset = COCOAnnsPairsDataset(clean_coco_ann_f_path, noisy_coco_ann_f_path)

    num_of_pairs = len(coco_pair_dataset)
    # iou_list, dice_list = np.empty(num_of_pairs, dtype=float), np.empty(num_of_pairs, dtype=float)
    iou_list, dice_list = [], []

    # for pair_idx, (clean_polygons, noisy_polygons) in tqdm(enumerate(coco_pair_dataset), total=num_of_pairs):
    for pair_idx in tqdm(range(num_of_pairs), total=num_of_pairs):
        clean_polygons, noisy_polygons = coco_pair_dataset[pair_idx]
        if clean_polygons is None:
            continue
        clean_polygons = list(map(lambda poly: poly.buffer(0), clean_polygons))
        noisy_polygons = list(map(lambda poly: poly.buffer(0), noisy_polygons))
        unary_clean, unary_noisy = unary_union(clean_polygons), unary_union(noisy_polygons)
        # todo - resolve validity in data creation, not in here
        # unary_clean = unary_clean.buffer(0)
        # unary_noisy = unary_noisy.buffer(0)
        intersection = unary_clean.intersection(unary_noisy)
        total_union = unary_clean.union(unary_noisy)

        iou_list.append(round(intersection.area / total_union.area, 6))
        dice_list.append(round((2 * intersection.area) / (unary_clean.area + unary_noisy.area), 6))

    with open(PROCESSED_DATA_DIR_PATH / 'coco' / 'segmentation_error_spatial.json', 'w') as f:
        json.dump({'iou': iou_list, 'dice': dice_list},
                  f, indent=4)
    plt.hist(iou_list, bins=100), plt.title('Histogram of IoU'), plt.show()
    plt.hist(dice_list, bins=100), plt.title('dice'), plt.show()
    return


if __name__ == '__main__':
    split = 'val'
    noise_name = 'spatial_approximation'
    clean_coco_annotation_f_path = RAW_DATA_DIR_PATH / 'coco' / split / 'annotations.json'
    noised_coco_annotation_f_path = RAW_DATA_DIR_PATH / 'coco-noised' / split / (noise_name + '.json')
    images_dir_path = RAW_DATA_DIR_PATH / 'coco' / 'split' / 'images'
    meas_segmentation_error(clean_coco_annotation_f_path, noised_coco_annotation_f_path, images_dir_path)
