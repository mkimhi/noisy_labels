import sys
from pathlib import Path
from shutil import rmtree

from segment_anything import SamAutomaticMaskGenerator

_CODE_PATH = Path(__file__).parent.parent

sys.path.append(str(_CODE_PATH))
sys.path.append(str(_CODE_PATH / 'external_libraries' / 'detectron2'))
sys.path.append(str(_CODE_PATH / 'external_libraries' / 'mmdetection'))

from typing import Literal, List
from argparse import Namespace, ArgumentParser

from pycocotools.coco import COCO
from pycocotools import mask as mask_util
import json
import numpy as np
import segment_anything
from PIL import Image
from tqdm import tqdm
from os import remove

import config
from common.constants import RAW_DATA_DIR_PATH, PROCESSED_DATA_DIR_PATH, CHECKPOINTS_DIR_PATH


def obtain_vfm_annotations(model_name: str, model_size: Literal['base', 'large', 'huge '],
                           imaegs_dir_path: Path,
                           src_annotation_f_path: Path,
                           out_annotation_f_path: Path) -> None:
    coco = COCO(str(src_annotation_f_path))
    orig_image_ids = coco.getImgIds()
    total_num_images = len(orig_image_ids)

    progress_backup_dir = out_annotation_f_path.parent / (out_annotation_f_path.stem + '-progress_backups')
    progress_backup_dir.mkdir(parents=True, exist_ok=True)
    backup_f_name_prefix = f'image_idx_'
    backup_f_paths = progress_backup_dir.glob(backup_f_name_prefix + '*')
    backup_indices = map(lambda bu_f_path: bu_f_path.stem[len(backup_f_name_prefix):], backup_f_paths)
    backup_indices = filter(str.isdecimal, backup_indices)
    backup_indices = map(int, backup_indices)
    try:
        prev_run_bu_idx = max(backup_indices)
    except ValueError:  # case no prev run
        prev_run_bu_idx = -1
        anns_out = {'anns_out_list': []}
    else:  # case there is a prev run
        prev_backup_f_path = progress_backup_dir / f'{backup_f_name_prefix}{prev_run_bu_idx}.json'
        with open(prev_backup_f_path, 'r') as f:
            anns_out = json.load(f)
    anns_out_list = anns_out['anns_out_list']

    img_idx_to_start_with = prev_run_bu_idx + 1

    out_annotation_f_path.parent.mkdir(parents=True, exist_ok=True)

    if model_name == 'sam':
        backup_every = 1 if config.RND_MODE else 75

        model_subname = 'vit_' + (model_size[0])
        checkpoint_path = str(CHECKPOINTS_DIR_PATH / 'sam' / f'{model_subname}.pth')
        sam = segment_anything.sam_model_registry[model_subname](checkpoint=checkpoint_path)
        sam.to(config.DEVICE)
        mask_generator = SamAutomaticMaskGenerator(sam)
        # predictor = segment_anything.SamPredictor(sam)

        annotation_id = 1
        for image_idx, img_id in tqdm(enumerate(orig_image_ids[img_idx_to_start_with:], start=img_idx_to_start_with),
                                      initial=img_idx_to_start_with, total=total_num_images):
            # if image_idx >= 2 and config.RND_MODE:
            #     break
            img_info = coco.loadImgs(img_id)[0]
            img_path = imaegs_dir_path / img_info['file_name']
            image = np.array(Image.open(img_path))
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            masks = mask_generator.generate(image)
            for mask in masks:
                segmentation = mask['segmentation']
                rle = mask_util.encode(np.asfortranarray(segmentation))
                rle['counts'] = rle['counts'].decode('utf-8')  # Ensure counts are serialized correctly
                anns_out_list.append({
                    'id': annotation_id,
                    'image_id': img_id,
                    'category_id': None,
                    'segmentation': rle,
                    'area': float(segmentation.sum()),
                    'bbox': list(mask_util.toBbox(rle)),
                    'iscrowd': None,
                })
                annotation_id += 1
            if ((image_idx + 1) % backup_every == 0) and (image_idx != total_num_images - 1):
                updated_backup_f_path = progress_backup_dir / f'{backup_f_name_prefix}{image_idx}.json'
                with open(updated_backup_f_path,
                          'w') as f:  # todo correctly save it in a coco format and not in as plain dict
                    json.dump(anns_out, f, indent=4)
                prev_backup_f_path = progress_backup_dir / f'{backup_f_name_prefix}{prev_run_bu_idx}.json'
                try:
                    remove(prev_backup_f_path)
                except Exception:
                    pass
                prev_run_bu_idx = image_idx
        with open(out_annotation_f_path, 'w') as f:
            json.dump(anns_out, f, indent=4)
        try:
            rmtree(progress_backup_dir)
        except FileNotFoundError:
            pass
    else:
        raise NotImplementedError()
    return


def parse_args(default_args: List[str]) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--rnd-mode', action='store_true')
    parser.add_argument('--phase', type=str, choices=['train', 'test', 'val'], required=True)
    parser.add_argument('--model-size', type=str, choices=['base', 'large', 'huge'], required=True)
    parser.add_argument('--noise-name', type=str, choices=['spatial-medium'], required=False, default='spatial-medium')
    parser.add_argument('--model-name', type=str, choices=['sam'], required=False, default='sam')
    parser.add_argument('--ver', type=int, required=False, default=None)
    argv = sys.argv[1:] if len(sys.argv) > 1 else default_args
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    default_args = [
        '--phase', 'train',
        '--noise-name', 'spatial-medium',
        '--model-name', 'sam',
        '--model-size', 'large',
        '--ver', '6',
    ]
    if config.RND_MODE:
        default_args.append('--rnd-mode')

    args = parse_args(default_args)
    config.RND_MODE = args.rnd_mode
    print('args:', args)

    dst_out_name = '-'.join((args.model_name, args.model_size, f'ver_{args.ver}'))
    if config.RND_MODE:
        dst_out_name += '-rnd_mode'
    obtain_vfm_annotations(
        args.model_name, args.model_size,
        imaegs_dir_path=RAW_DATA_DIR_PATH / 'coco' / args.phase / 'images',
        src_annotation_f_path=PROCESSED_DATA_DIR_PATH / 'coco' / args.phase / (args.noise_name + '.json'),
        out_annotation_f_path=PROCESSED_DATA_DIR_PATH / 'coco' / args.phase / (dst_out_name + '.json')
    )
