import json
import sys
from copy import deepcopy
from itertools import count as iter_count
from pathlib import Path

from tqdm import tqdm

_CODE_PATH = Path(__file__).parent.parent

# sys.path.append(str(_CODE_PATH))
# sys.path.append(str(_CODE_PATH / 'external_libraries' / 'detectron2'))
# sys.path.append(str(_CODE_PATH / 'external_libraries' / 'mmdetection'))
import numpy as np
import argparse
from common.constants import RAW_DATA_DIR_PATH
from seg_utils.os_utils import is_pycharm_and_vscode_hosted

import logging
from pycocotools import mask as cocomask
import config

POLY_ANN_DIR_NAME = 'polygons-annotations'


def argument_parser(args_list) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('cs_dir_path', type=Path, metavar='CITYSCAPES_DIR_PATH',
                        help='Source directory with structure: <phase>/polygons-annotations/json-files for phase=train and val')
    parser.add_argument('--rnd-mode', action='store_true',
                        help='Activate RnD mode for experimental features or testing (e.g., evaluate fewer samples/epochs)')
    return parser.parse_args(args_list)


def validate_args(args: argparse.Namespace) -> None:
    # prevents from the user to accidentally override teh src
    for phase in ['val', 'train']:
        dst_f_path = args.cs_dir_path / phase / args.annotations_f_name
        if not args.rnd_mode:
            assert not dst_f_path.exists(), f'destination already exists: {dst_f_path}'


def convert_cityscapes_to_coco(cs_dir_path: Path, rnd_mode: bool, annotations_f_name: str) -> None:
    poly_ann_suffix = '_gtFine_polygons.json'
    im_addition_suffix = '_leftImg8bit'
    ann_phase_list = ['train', 'val']
    im_id_gen = iter_count(start=1)
    cat_id_gen = iter_count(start=1)
    ann_id_gen = iter_count(start=1)

    images_lists_dict = {}
    annotations_lists_dict = {}
    categories_dict = {}

    for phase in ann_phase_list:
        phase_images_list = []
        phase_annotations_list = []
        for f_idx, json_f_path in tqdm(
            enumerate(l := list((cs_dir_path / phase / POLY_ANN_DIR_NAME).glob(f'*{poly_ann_suffix}'))),
            total=len(l),
            desc=phase
        ):
            if f_idx >= 25 and rnd_mode:
                break
            with open(json_f_path, 'r') as f:
                image_anns_info = json.load(f)
            image_height, image_width = image_anns_info['imgHeight'], image_anns_info['imgWidth']
            image_dict = {
                'id': next(im_id_gen),
                'width': image_width,
                'height': image_height,
                'file_name': json_f_path.stem + im_addition_suffix + '.png',
                'license': None,
                'date_captured': None,
            }
            phase_images_list.append(image_dict)
            for ann_dict in image_anns_info['objects']:
                category = ann_dict['label']
                if category not in categories_dict:
                    categories_dict[category] = {
                        'id': next(cat_id_gen),
                        'name': category,
                        'supercategory': None,
                    }
                category_dict = categories_dict[category]
                inp_polygon = np.array(ann_dict['polygon'])

                ### calc segmentation
                polygon = inp_polygon.flatten().tolist()
                rle = cocomask.frPyObjects([polygon], image_height, image_width)[0]

                rle_to_save = deepcopy(rle)
                rle_to_save['counts'] = rle_to_save['counts'].decode('utf-8') if isinstance(rle_to_save['counts'], bytes) else rle_to_save['counts']
                annotation_dict = {
                    'id': next(ann_id_gen),
                    'image_id': image_dict['id'],
                    'category_id': category_dict['id'],
                    'segmentation': rle_to_save, # RLE or [polygon],
                    'area': float(cocomask.area(rle)),  # float  # the wrapper of float is to be able to save in json
                    'bbox': cocomask.toBbox(rle).astype(int).tolist(),
                    'iscrowd': 0
                }
                phase_annotations_list.append(annotation_dict)
        images_lists_dict[phase] = phase_images_list
        annotations_lists_dict[phase] = phase_annotations_list
    categories_list = list(categories_dict.values())

    for phase in ann_phase_list:
        phase_dst_f_path = args.cs_dir_path / phase / annotations_f_name
        coco_dict_to_save = {
            'images': images_lists_dict[phase],
            'annotations': annotations_lists_dict[phase],
            'categories': categories_list,
        }
        with open(phase_dst_f_path, 'w') as f:
            json.dump(coco_dict_to_save, f, indent=4)
    return


if __name__ == "__main__":
    if is_pycharm_and_vscode_hosted():
        args_list = [
            str(RAW_DATA_DIR_PATH / 'cityscapes'),
        ]
        if config.RND_MODE:
            args_list.append('--rnd-mode')
    else:
        args_list = sys.argv[1:]

    args = argument_parser(args_list)
    args.annotations_f_name = 'annotations-rnd_mode.json' if args.rnd_mode else 'annotations.json'
    validate_args(args)

    logging.basicConfig(level=logging.INFO)
    logging.info(args)

    convert_cityscapes_to_coco(**vars(args))
