import argparse
import json
import logging
import sys
from collections import defaultdict
from copy import deepcopy
from os import remove
from pathlib import Path
from PIL import Image

_CODE_PATH = Path(__file__).parent.parent.parent

sys.path.append(str(_CODE_PATH))
sys.path.append(str(_CODE_PATH / 'external_libraries' / 'detectron2'))
sys.path.append(str(_CODE_PATH / 'external_libraries' / 'mmdetection'))
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_utils
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely.ops import unary_union
from enums import Noises
from enum import Enum, auto
import config
from typing import List, Dict, Union
from common.constants import PROCESSED_DATA_DIR_PATH, RAW_DATA_DIR_PATH, SEED, DEFAULT_PLT_COLORS
from seg_utils.os_utils import is_pycharm_and_vscode_hosted
import numpy as np
import cv2
import random
from tqdm import tqdm


def polygon_to_mask(polygon, h, w):
    """
    Converts a polygon to a mask.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    modified_polygon = [polygon_part.astype(np.int32).T for polygon_part in polygon]
    cv2.fillPoly(mask, modified_polygon, 1)
    return mask


def mask_to_polygon(mask: np.ndarray) -> List[np.ndarray]:
    return [
        contour[:, 0].T
        for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    ]


def preproc_polygons(ann_data: dict, coco_obj: COCO) -> List[np.ndarray]:
    orig_segmentation = ann_data['segmentation']
    if isinstance(orig_segmentation, list):
        expand_polygon_dim = lambda polygon: [polygon[:-1:2], polygon[1::2]]
        preproc_polygons = list(map(np.array, map(expand_polygon_dim, orig_segmentation)))
    elif isinstance(orig_segmentation, dict):
        mask = coco_obj.annToMask(ann_data)
        preproc_polygons = [
            contour[:, 0].T
            for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        ]
    else:
        raise TypeError('orig_segmentation is not a standard COCO format')
    return preproc_polygons


def get_deletion_noise(noise_params: dict) -> bool:
    delete_prob = noise_params[Noises.DELETION]
    delete_noise = np.random.choice([True, False], p=[delete_prob, 1 - delete_prob])
    return delete_noise


def get_class_flip_noise(ann_cat_id: int, cat_id_to_group_other_cats_ids: dict, class_flip_prob) -> int:
    updated_cat_id = ann_cat_id
    class_flip_noise = np.random.choice([True, False], p=[class_flip_prob, 1 - class_flip_prob])
    if class_flip_noise:
        group_other_cats_ids = cat_id_to_group_other_cats_ids[ann_cat_id]
        if group_other_cats_ids:  # do not flip in case there are no additional items in category
            num_of_other_cats = len(group_other_cats_ids)
            updated_cat_id = np.random.choice(num_of_other_cats, p=np.full(num_of_other_cats, 1 / num_of_other_cats))
            updated_cat_id = int(updated_cat_id)  # to be able jsonify
    return updated_cat_id


def add_approximation_noise(polygon_arr: np.ndarray, approx_noise_params: tuple) -> np.ndarray:
    if polygon_arr.shape[1] <= 3:
        return np.array([])
    tolerance = max(np.random.normal(*approx_noise_params), 0)
    polygon_obj = Polygon(polygon_arr.T)
    polygon_obj = polygon_obj.simplify(tolerance)
    noised_polygon_arr = np.array(polygon_obj.exterior.xy)
    return noised_polygon_arr


def add_localization_noise(polygon_arr: np.ndarray, localization_noise_params: tuple) -> np.ndarray:
    if not polygon_arr.size:
        return polygon_arr
    gaussian_std = max(np.random.normal(*localization_noise_params), 0)
    distortions = np.random.normal(0, gaussian_std, polygon_arr.shape)
    noise_polygon_arr = polygon_arr + distortions
    noise_polygon_arr[:, -1] = noise_polygon_arr[:, 0]
    return noise_polygon_arr


def add_scale_noise(polygon_arr: List[np.ndarray], scale_noise_params: tuple,
                    scale_noise_type: str, im_height_width: [int, int]) -> List[np.ndarray]:
    orig_scale_param = np.random.normal(*scale_noise_params)
    if not polygon_arr:
        return polygon_arr
    polygon_arr = list(filter(lambda poly_part: poly_part.size >= 3, polygon_arr))
    if scale_noise_type == 'polygon':
        scale_param = orig_scale_param * np.random.choice([-1, 1])
        noise_polygon_arr = []
        for polygon_part in polygon_arr:
            l1_polygon_center = (polygon_part.max(axis=1) + polygon_part.min(axis=1)) / 2
            centered_polygon_arr = (polygon_part[:].T - l1_polygon_center).transpose()
            # convert num of pixels for the most distant vertex, to a scale ratio
            vertices_distance_from_center = np.linalg.norm(centered_polygon_arr, axis=0)
            max_distance_from_center = vertices_distance_from_center.max()
            scale_multiplier = (max_distance_from_center + scale_param) / max_distance_from_center
            #####
            noise_polygon_part = ((centered_polygon_arr * scale_multiplier).T + l1_polygon_center).transpose()
            noise_polygon_arr.append(noise_polygon_part)
    elif scale_noise_type == 'mask':
        scale_param = int(max(orig_scale_param, 0))
        if scale_param == 0:
            return polygon_arr
        mask = polygon_to_mask(polygon_arr, *im_height_width)
        k = int(scale_param)
        kernel = np.ones((k, k), np.uint8)
        updated_mask = (
            cv2.dilate(mask, kernel, iterations=1)
            if np.random.rand() < 0.5
            else cv2.erode(mask, kernel, iterations=1)
        )
        noise_polygon_arr = mask_to_polygon(updated_mask)
    else:
        raise TypeError('scale_noise_type must be "polygon" or "mask"')
    return noise_polygon_arr


def join_polygons(polygon_list: List[np.ndarray]) -> List[List[float]]:
    if not polygon_list:
        return polygon_list
    updated_polygon_list = list(map(Polygon, map(np.transpose, polygon_list)))
    # handle self-intersections
    updated_polygon_list = list(
        map(lambda polygon_part: polygon_part if polygon_part.is_valid else polygon_part.buffer(0),
            updated_polygon_list))
    updated_polygon_list = unary_union(updated_polygon_list)
    updated_polygon_list = list(updated_polygon_list.geoms) if isinstance(updated_polygon_list, MultiPolygon) else [
        updated_polygon_list]
    updated_polygon_list = list(filter(lambda polygon_part: not polygon_part.is_empty, updated_polygon_list))
    updated_polygon_list = list(map(lambda polygon_part: np.array(polygon_part.exterior.xy), updated_polygon_list))
    return updated_polygon_list


def noise_ann(polygons: List[np.ndarray], noise_params: dict,
              cat_id_to_group_other_cats_ids: Dict[int, List[int]],
              ann_cat_id: int, scale_noise_type: str, im_height_width: [int, int]) -> [List[np.ndarray], int]:
    '''
    Noise a list of polygons. If the noise is deletion, return None, otherwise returns
    the noise polygons list.
    '''
    ### deletion noise
    delete_noise = get_deletion_noise(noise_params)
    if delete_noise:
        return [], -1
    updated_cat_id = get_class_flip_noise(ann_cat_id, cat_id_to_group_other_cats_ids,
                                          class_flip_prob=noise_params[Noises.CLASS_FLIP])
    noised_polygons = [add_approximation_noise(polygon_part, noise_params[Noises.APPROXIMATION]) for polygon_part in
                       polygons]
    noised_polygons = [add_localization_noise(polygon_part, noise_params[Noises.LOCALIZATION]) for polygon_part in
                       noised_polygons]
    noised_polygons = add_scale_noise(noised_polygons, noise_params[Noises.SCALE], scale_noise_type, im_height_width)
    noised_polygons = list(filter(lambda polygon_part: polygon_part.shape[1] >= 3, noised_polygons))
    noised_polygons = join_polygons(noised_polygons)
    return noised_polygons, updated_cat_id


def calc_ann_area(polygon: List[np.array], im_height_width: [int, int]) -> int:
    polygon_arr_dtype = np.int64
    polygon_as_int = list(map(np.ndarray.transpose, polygon))
    polygon_as_int = list(map(lambda polygon_part: polygon_part.astype(polygon_arr_dtype), polygon_as_int))
    mask_dtype = np.uint8
    mask = np.zeros(im_height_width, dtype=mask_dtype)
    cv2.fillPoly(mask, polygon_as_int, color=1)
    area = mask.sum()
    area = int(area)  # to be able jsonify
    # equivalent to:
    # rle_mask = coco_mask_utils.area(coco_mask_utils.encode(np.asfortranarray(mask)))
    return area


def get_cat_id_to_group_other_cats_ids(coco_cats: List[dict]) -> dict:
    supercat_to_cats_ids = defaultdict(lambda: [])
    for cat in coco_cats:
        cat_id = cat['id']
        try:
            super_cat = cat['supercategory']
        except KeyError:
            super_cat = 'object'
        supercat_to_cats_ids[super_cat].append(cat_id)

    cat_id_to_group_other_cats_ids = {}
    for super_cat, cats_ids in supercat_to_cats_ids.items():
        for cat_id in cats_ids:
            copied_cats_ids = deepcopy(cats_ids)
            copied_cats_ids.remove(cat_id)
            cat_id_to_group_other_cats_ids[cat_id] = copied_cats_ids
    return cat_id_to_group_other_cats_ids


def calc_bbox(polygon: List[np.array]) -> [float, float, float, float]:
    if not polygon:
        return [0, 0, 0, 0]
    max_xy = np.array(list(map(lambda polygon_part: polygon_part.max(axis=1), polygon))).max(axis=0)
    min_xy = np.array(list(map(lambda polygon_part: polygon_part.min(axis=1), polygon))).min(axis=0)
    bbox = [*min_xy, *(max_xy - min_xy)]  # format [x, y, width, height]
    return bbox


def noise_annotations_v2(src_coco_anns_f_path, dst_annotations_f_path: Path,
                         noise_params: dict, scale_noise_type, seed,
                         rnd_mode, exp_num, plot_examples, save_output):
    orig_segmentation: List[List[float]]
    preprocessed_polygons: List[np.ndarray]
    noised_polygon: List[np.ndarray]
    postprocessed_polygon: List[List[float]]

    coco_obj: COCO = COCO(src_coco_anns_f_path)
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    src_annotations_list = coco_obj.dataset['annotations']
    dst_annotations_list = []
    cat_id_to_group_other_cats_ids = get_cat_id_to_group_other_cats_ids(coco_obj.dataset['categories'])
    random.shuffle(src_annotations_list)
    flatten_polygon_method = lambda arr: np.insert(arr[1], np.arange(len(arr[0])), arr[0]).tolist()

    for ann_idx, ann_data in enumerate(tqdm(src_annotations_list)):
        if rnd_mode and ann_idx >= 30:
            break
        preprocessed_polygons = preproc_polygons(ann_data, coco_obj)
        im_f_name = coco_obj.loadImgs(ann_data['image_id'])[0]['file_name']
        orig_cat_id = ann_data['category_id']
        image_info = coco_obj.loadImgs(ann_data['image_id'])[0]
        im_height_width = (image_info['height'], image_info['width'])
        noised_polygon, updated_cat_id = noise_ann(preprocessed_polygons, noise_params,
                                                   cat_id_to_group_other_cats_ids,
                                                   ann_cat_id=ann_data['category_id'],
                                                   scale_noise_type=scale_noise_type,
                                                   im_height_width=im_height_width)
        if not noised_polygon:
            continue
        bool_class_noise = orig_cat_id == updated_cat_id
        ann_data['bool_class_noise'] = bool_class_noise
        if bool_class_noise:
            ann_data['category_id'] = updated_cat_id

        ann_data['area'] = calc_ann_area(noised_polygon, im_height_width)
        ann_data['bbox'] = calc_bbox(noised_polygon)
        postprocessed_polygon = list(map(flatten_polygon_method, noised_polygon))
        ann_data['segmentation'] = postprocessed_polygon

        dst_annotations_list.append(ann_data)
        if plot_examples and ann_idx < 100:
            im_f_path = src_coco_anns_f_path.parent / 'images' / im_f_name
            image = np.array(Image.open(im_f_path))
            fig, axs = plt.subplots(1, 2)
            for ax_idx, ax in enumerate(axs):
                ax.imshow(image)
                for i, part_polygon in enumerate(preprocessed_polygons):
                    ax.plot(*part_polygon, c=DEFAULT_PLT_COLORS[0],
                            label='orig_ann' if i == ax_idx == 0 else None)
                if noised_polygon:
                    for i, part_polygon in enumerate(noised_polygon):
                        ax.plot(*part_polygon, c=DEFAULT_PLT_COLORS[1],
                                label='noised_ann' if i == ax_idx == 0 else None)
            margin = 5
            axs[1].set_xlim(ann_data['bbox'][0] - margin, ann_data['bbox'][0] + ann_data['bbox'][2] + margin)
            axs[1].set_ylim(ann_data['bbox'][1] - margin, ann_data['bbox'][1] + ann_data['bbox'][3] + margin)
            axs[1].invert_yaxis()
            axs[0].set_title(next(cat['name'] for cat in coco_obj.dataset['categories'] if cat['id'] == orig_cat_id))
            if updated_cat_id != orig_cat_id:
                axs[1].set_title(
                    next((cat['name'] for cat in coco_obj.dataset['categories'] if cat['id'] == updated_cat_id), None))
            elif not noised_polygon:
                axs[1].set_title('deleted')
            fig.legend()
            fig.suptitle(f'exp: {exp_num}')
            fig.show()
        elif not save_output:
            break
    if save_output:
        coco_dict = json.load(open(src_coco_anns_f_path))
        coco_dict['annotations'] = dst_annotations_list
        if dst_annotations_f_path.is_file():
            remove(dst_annotations_f_path)
        dst_annotations_f_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_annotations_f_path, 'w') as f:
            json.dump(coco_dict, f, indent=4)
    return


def noise_annotations_v2_wrapper(exp_num, split, dataset, noise_intensity, rnd_mode, dataset_format):
    scale_noise_type = 'mask'  # "polygon" or "mask"

    deletion_prob = 0.05
    plot_examples = False
    save_output = True
    noise_bases = {
        'low': np.array([5, 2, 3]),
        'mid': np.array([10, 3, 5]),
        'high': np.array([15, 4, 7]),
    }
    orig_diff = np.array([5, 1, 2])
    std_coeffidients = {
        'low': 0.5,
        'mid': 0.5,
        'high': 2.0,
    }

    chosen_mean = noise_bases[noise_intensity]
    chosen_std = orig_diff * std_coeffidients[noise_intensity]

    exp_id = 'v' + str(exp_num)
    print('exp_id: ', exp_id)
    print('scale_noise_type: ', scale_noise_type)
    print('deletion_prob: ', deletion_prob)
    print('split: ', split)
    print('save_output: ', save_output)
    print('chosen_mean: ', chosen_mean)
    print('chosen_std: ', chosen_std)
    annotation_out_f_stem = exp_id
    if rnd_mode:
        annotation_out_f_stem += '_rnd-mode'

    if dataset_format == 'coco':
        src_coco_anns_f_path = RAW_DATA_DIR_PATH / dataset / split / 'annotations.json'
        dst_annotations_f_path = PROCESSED_DATA_DIR_PATH / dataset / split / (annotation_out_f_stem + '.json')
    elif dataset_format == 'cityscapes':
        src_coco_anns_f_path = RAW_DATA_DIR_PATH / dataset / 'annotations' / f'instancesonly_filtered_gtFine_{split}.json'
        ver_id_dir_name = f'v{exp_num}'
        if rnd_mode:
            ver_id_dir_name += '_rnd_mode'
        dst_annotations_f_path = PROCESSED_DATA_DIR_PATH / dataset / ver_id_dir_name / f'instancesonly_filtered_gtFine_{split}.json'
    else:
        raise Exception('Dataset format not supported: {}'.format(dataset_format))

    noise_params = {
        Noises.DELETION: deletion_prob,  # prob for deletion
        Noises.CLASS_FLIP: 0.05,  # prob for class flip
        Noises.APPROXIMATION: (chosen_mean[0], chosen_std[0]),  # mean, std
        Noises.LOCALIZATION: (chosen_mean[1], chosen_std[1]),  # mean, std
        Noises.SCALE: (chosen_mean[2], chosen_std[2]),  # mean, std
    }

    logging.basicConfig(level=logging.INFO)
    logging.info(args)

    noise_annotations_v2(src_coco_anns_f_path, dst_annotations_f_path,
                         scale_noise_type=scale_noise_type,
                         noise_params=noise_params, exp_num=exp_num, plot_examples=plot_examples,
                         save_output=save_output, rnd_mode=rnd_mode, seed=SEED)
    print('DONE')


def parse_args(args_list: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='noise annotations v2')
    parser.add_argument('--split', type=str, choices=['train', 'val'])
    parser.add_argument('--dataset', type=str, choices=['coco', 'lvis', 'cityscapes'])
    parser.add_argument('--exp-num', type=int)
    parser.add_argument('--noise-intensity', type=str, choices=['low', 'mid', 'high'])
    parser.add_argument('--dataset-format', type=str, choices=['coco', 'cityscapes'])
    parser.add_argument('--rnd-mode', action='store_true')
    return parser.parse_args(args_list)


if __name__ == '__main__':
    if is_pycharm_and_vscode_hosted():
        split = 'val'
        dataset = 'cityscapes'
        dataset_format = 'cityscapes'
        exp_num = 26
        noise_intensity = 'low'
        rnd_mode = True

        print('split', split, 'exp_num', exp_num, 'noise_intensity', noise_intensity)
        argv = [
            '--split', split,
            '--dataset', dataset,
            '--exp-num', str(exp_num),
            '--noise-intensity', noise_intensity,
            '--dataset-format', dataset_format,
        ]
        if rnd_mode:
            argv.append('--rnd-mode')
        print('argv:')
        for i, arg in enumerate(argv):
            print(arg, end='\n' if i == (len(argv) - 1) else ' ')
    else:
        argv = sys.argv[1:]
    print('argv:', argv)
    args = parse_args(argv)
    print('args:', args)
    noise_annotations_v2_wrapper(**vars(args))
