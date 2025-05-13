import sys
from pathlib import Path

_CODE_PATH = Path(__file__).parent.parent.parent

sys.path.append(str(_CODE_PATH))
sys.path.append(str(_CODE_PATH / 'external_libraries' / 'detectron2'))
sys.path.append(str(_CODE_PATH / 'external_libraries' / 'mmdetection'))
import logging
from seg_utils.os_utils import is_pycharm_and_vscode_hosted
import json
from collections import defaultdict

import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from shapely.geometry import Polygon
from tqdm import tqdm
import random
import argparse

import config
from common.constants import RAW_DATA_DIR_PATH, PROCESSED_DATA_DIR_PATH, SEED
from constants import EASY, MEDIUM, HARD, APPROXIMATION, LOCALIZATION, SCALE, AMOUNT, PARAMS, DELETE_ANN, FLIP
from rnd.plot_utils import plot_benchmarks_params


def get_noises_params_dict() -> dict:
    flip_class_val = 0.05
    delete_ann_val = 0.05

    noise_intensity_params_vals = {
        APPROXIMATION: [5, 10, 15],
        LOCALIZATION: [2, 3, 4],
        SCALE: [3, 5, 7],
    }

    images_percents = [0.2, 0.25, 0.3]

    bool_plot_benchmarks_params = False
    if bool_plot_benchmarks_params:
        plot_benchmarks_params(noise_intensity_params_vals, images_percents, flip_class_val, delete_ann_val)

    noise_types_list = [APPROXIMATION, LOCALIZATION, SCALE]

    bms_dicts = {}
    for intensity_idx, intensity in enumerate([EASY, MEDIUM]):
        bms_dicts['-'.join((intensity, AMOUNT))] = {
            noise_type: (images_percents[intensity_idx], noise_intensity_params_vals[noise_type][-1])
            for noise_type in noise_types_list
        }
        bms_dicts['-'.join((intensity, PARAMS))] = {
            noise_type: (images_percents[-1], noise_intensity_params_vals[noise_type][intensity_idx])
            for noise_type in noise_types_list
        }
    hard_intensity_idx = 2
    bms_dicts[HARD] = {
        noise_type: (images_percents[hard_intensity_idx], noise_intensity_params_vals[noise_type][hard_intensity_idx])
        for noise_type in noise_types_list
    }

    for bm_dict in bms_dicts.values():
        bm_dict[FLIP] = flip_class_val
        bm_dict[DELETE_ANN] = delete_ann_val
    return bms_dicts


NOISES_PARAMS_DICT = get_noises_params_dict()


def pick_noise(probabilities):
    # Calculate the remaining probability for no noise.
    mask_noise_names_list = ['approximation', 'localization', 'scale', 'none']
    probabilities['none'] = [0.0, None]
    remaining_probability = 1 - sum(probabilities[key][0] for key in mask_noise_names_list)
    probabilities['none'][0] = remaining_probability

    probs = [probabilities[key][0] for key in mask_noise_names_list]
    chosen_key = random.choices(mask_noise_names_list, weights=probs, k=1)[0]
    return chosen_key, probabilities[chosen_key][1]


def new_boundaries_with_prob(d, coco, ann_id, file, annotation,
                             sorted_images_ids, im_id_locs_2_im_anns_amount,
                             categories_ids_2_categories_map):
    """
    Modifies the boundaries of an annotation with a given probability.
    """
    current_ann = coco.loadAnns(ann_id)[0]
    mask = coco.annToMask(current_ann)

    img_id_loc = sorted_images_ids.searchsorted(annotation['image_id'])
    img_anns_amount = im_id_locs_2_im_anns_amount[img_id_loc]
    bool_to_delete = False
    if img_anns_amount >= 2:
        prob_to_delete_ann = d[DELETE_ANN]
        if prob_to_delete_ann > 0.0 and np.random.choice(2, p=[1 - prob_to_delete_ann, prob_to_delete_ann]):
            bool_to_delete = True
            return None, None, None, bool_to_delete

    orig_category_id = annotation['category_id']
    category_dict = categories_ids_2_categories_map[orig_category_id]
    updated_category_id = np.random.choice(
        category_dict['flip_categories_distribution_support'],
        p=category_dict['flip_categories_distribution']
    )
    # the output of random choice is int64, which is not jsonified
    updated_category_id = int(updated_category_id)
    # bool() is to attain a bool type and and not np.bool, to be able serialize this item in json
    changed_class = bool(orig_category_id != updated_category_id)
    if changed_class:
        annotation['category_id'] = updated_category_id

    spatial_noise_type, k = pick_noise(d)
    if spatial_noise_type == 'scale':
        kernel = np.ones((k, k), np.uint8)
        new_mask = (
            cv2.dilate(mask, kernel, iterations=1)
            if np.random.rand() < 0.5
            else cv2.erode(mask, kernel, iterations=1)
        )
    elif spatial_noise_type == 'localization':
        new_mask = add_localization_noise(mask, k)
    elif spatial_noise_type == 'approximation':
        new_mask = add_approximation_noise(mask, k)
    elif spatial_noise_type == 'none':
        new_mask = mask
    else:
        raise ValueError(f'Unknown boundary version: {spatial_noise_type}')

    # Convert modified mask back to RLE
    rle_modified = maskUtils.encode(np.asfortranarray(new_mask))
    rle_modified['counts'] = rle_modified['counts'].decode('utf-8') if isinstance(rle_modified['counts'], bytes) else \
        rle_modified['counts']

    return rle_modified, spatial_noise_type, changed_class, bool_to_delete


def get_sorted_images_ids_and_im_id_locs_2_im_anns_amount(file: dict) -> [np.ndarray, np.ndarray]:
    '''
    counts annotations for each image id in the order given by images_ids_arr
    '''
    sorted_images_ids = np.sort([im['id'] for im in file['images']])
    annotation_id_2_im_id = np.array([annotation['image_id'] for annotation in file['annotations']])
    images_ids_to_images_ids_locs = {im_id: im_id_loc for im_id_loc, im_id in enumerate(sorted_images_ids)}
    annotation_id_2_im_id_loc = np.array([images_ids_to_images_ids_locs[im_id] for im_id in annotation_id_2_im_id])
    im_id_locs_2_im_anns_amount = np.bincount(annotation_id_2_im_id_loc)
    return sorted_images_ids, im_id_locs_2_im_anns_amount


def create_categories_labels_noise_distribution(categories_dict: dict, labels_flip_percents: float) -> None:
    """
    Creates labels noise distribution over the labels' superclasses.
    Returns in form of mapping of id's to
    :param categories_dict:
    :return:
    """
    super_categories_2_categories_list = defaultdict(lambda: [])
    for category in categories_dict:
        super_categories_2_categories_list[category['supercategory']].append(category['id'])
    supercategory_data_dict = {}
    for supercategory, categories_list in super_categories_2_categories_list.items():
        categories_ids_arr = np.array(categories_list)
        num_of_sub_categories = len(categories_ids_arr)
        if num_of_sub_categories == 1:
            percents_per_flip, non_flip_percents = 0.0, 1.0
        else:
            percents_per_flip, non_flip_percents = labels_flip_percents / (
                    num_of_sub_categories - 1), 1 - labels_flip_percents
        supercategory_data_dict[supercategory] = {
            'categories_ids_arr': categories_ids_arr,
            'percents_per_flip': percents_per_flip,
            'non_flip_percents': non_flip_percents,
        }
    for category in categories_dict:
        supercategory_data = supercategory_data_dict[category['supercategory']]
        flip_categories_distribution_support = supercategory_data['categories_ids_arr']
        category['flip_categories_distribution_support'] = flip_categories_distribution_support
        flip_distribution = np.ones_like(flip_categories_distribution_support, dtype=np.float32) * supercategory_data[
            'percents_per_flip']
        flip_distribution[flip_categories_distribution_support == category['id']] = supercategory_data[
            'non_flip_percents']
        category['flip_categories_distribution'] = flip_distribution
    return


def mask_to_polygon(mask):
    """
    Converts a mask to a polygon.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour[:, 0, :] for contour in contours] if contours else []


def polygon_to_mask(polygon, h, w):
    """
    Converts a polygon to a mask.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def add_gaussian_noise(vertices, mean=0, std_dev=1):
    """
    Adds Gaussian noise to each vertex in a polygon.
    """
    noise = np.random.normal(mean, std_dev, vertices.shape)
    return np.round(vertices + noise).astype(int)


def add_localization_noise(mask, std_dev=1):
    """
    Adds Gaussian noise to the vertices of the polygon.
    """
    if np.sum(mask) == 0:
        return mask

    final_mask_noisy = np.zeros(mask.shape, dtype=np.uint8)
    for polygon in mask_to_polygon(mask):
        vertices_noisy = add_gaussian_noise(polygon, std_dev=std_dev)
        final_mask_noisy = np.maximum(final_mask_noisy, polygon_to_mask(vertices_noisy, mask.shape[0], mask.shape[1]))

    return final_mask_noisy


def simplify_polygon(polygon, tolerance):
    """
    Simplifies the polygon by removing vertices.
    """
    if len(polygon) < 4:
        return None

    shapely_polygon = Polygon(polygon)
    return shapely_polygon.simplify(tolerance, preserve_topology=True)


def simplified_polygon_to_mask(simplified_polygon, h, w):
    """
    Converts a simplified polygon back to a mask.
    """
    new_mask = np.zeros((h, w), dtype=np.uint8)
    simplified_coords = np.array(simplified_polygon.exterior.coords).reshape((-1, 1, 2))
    cv2.fillPoly(new_mask, [simplified_coords.astype(np.int32)], color=(1))
    return new_mask


def add_approximation_noise(mask, tolerance):
    """
    Adds noise to the vertices of the polygon by simplifying it.
    """
    if np.sum(mask) == 0:
        return mask

    final_mask_noisy = np.zeros(mask.shape, dtype=np.uint8)
    for polygon in mask_to_polygon(mask):
        simplified_polygon = simplify_polygon(polygon, tolerance)
        if simplified_polygon is None:
            continue
        mask_noisy = simplified_polygon_to_mask(simplified_polygon, mask.shape[0], mask.shape[1])
        final_mask_noisy = np.maximum(final_mask_noisy, mask_noisy)

    return final_mask_noisy


def main_noise_annotations(src_annotations_f_path: Path, dst_annotations_f_path: Path,
                           noise_name: str, seed: int, rnd_mode: bool, experiment_desc: str = None):
    if experiment_desc is None:
        experiment_desc = noise_name

    with open(src_annotations_f_path) as f:
        coco_file = json.load(f)
    coco = COCO(src_annotations_f_path)

    np.random.seed(seed)

    noise_params_dict = NOISES_PARAMS_DICT[noise_name]
    sorted_images_ids, im_id_locs_2_im_anns_amount = get_sorted_images_ids_and_im_id_locs_2_im_anns_amount(coco_file)
    create_categories_labels_noise_distribution(categories_dict=coco_file['categories'],
                                                labels_flip_percents=noise_params_dict[FLIP])
    categories_ids_2_categories_map = {category['id']: category for category in coco_file['categories']}

    annotations_indices_to_delete = []
    file_annotations = coco_file['annotations']
    for ann_idx, annotation in tqdm(enumerate(file_annotations), total=len(file_annotations), desc=experiment_desc):
        if rnd_mode and ann_idx >= 100:
            break
        new_mask, spatial_noise_type, class_noise, bool_to_delete = new_boundaries_with_prob(
            noise_params_dict.copy(), coco, annotation['id'], coco_file, annotation,
            sorted_images_ids, im_id_locs_2_im_anns_amount,
            categories_ids_2_categories_map
        )
        if bool_to_delete:
            annotations_indices_to_delete.append(ann_idx)
            continue
        annotation['spatial_noise_type'] = spatial_noise_type
        annotation['segmentation'] = new_mask

        annotation['class_noise'] = class_noise
    annotations_indices_to_delete_descending_order = annotations_indices_to_delete[::-1]
    # the deletion must be in descending indices order because the deletion is from a list
    for ann_idx_to_delete in annotations_indices_to_delete_descending_order:
        del file_annotations[ann_idx_to_delete]
    for cat in coco_file['categories']:
        del cat['flip_categories_distribution_support']
        del cat['flip_categories_distribution']

    dst_annotations_f_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_annotations_f_path, 'w') as f:
        json.dump(coco_file, f, indent=4)
    return


def argument_parser(args_list) -> argparse.Namespace:
    # todo add custom noise like {scale: [0.2, 4]}
    parser = argparse.ArgumentParser()
    parser.add_argument('src_annotations_f_path', type=Path, metavar='SRC_ANNOTATIONS_PATH',
                        help='The path to the original annotations')
    parser.add_argument('dst_annotations_f_path', type=Path, metavar='DST_ANNOTATIONS_PATH',
                        help='The path to the updated annotations')
    parser.add_argument('--noise-name', type=str, required=True, choices=NOISES_PARAMS_DICT.keys(),
                        help='The noise name to be used for the corruption')
    parser.add_argument('--seed', type=int, default=SEED, required=False,
                        help='The seed to be used for the random noise')
    parser.add_argument('--rnd-mode', action='store_true',
                        help='Activating RnD mode')
    return parser.parse_args(args_list)


def validate_args(args: argparse.Namespace) -> None:
    # prevents from the user to accidentally override teh src
    # assert not args.dst_annotations_f_path.exists(), f'dst-annotations-path already exists: {args.dst_annotations_f_path}'
    # assert args.dst_annotations_f_path.suffix == '', f'dst-annotations-path must have a suffix'
    pass


if __name__ == '__main__':
    if is_pycharm_and_vscode_hosted():
        split = 'val'
        noise_name = 'hard'
        benchmark = 'coco'

        annotation_out_f_stem = noise_name
        if config.RND_MODE:
            annotation_out_f_stem += '_rnd-mode'

        args_list = [
            str(RAW_DATA_DIR_PATH / benchmark / split / 'annotations.json'),  # SRC_ANNOTATIONS_PATH
            str(PROCESSED_DATA_DIR_PATH / 'coco' / split / (annotation_out_f_stem + '.json')),  # DST_ANNOTATIONS_PATH
            '--noise-name', 'hard',
        ]
        if config.RND_MODE:
            args_list.append('--rnd-mode')
    else:
        args_list = sys.argv[1:]
    args = argument_parser(args_list)
    validate_args(args)

    logging.basicConfig(level=logging.INFO)
    logging.info(args)

    main_noise_annotations(**vars(args))
