import contextlib
import datetime
import io
import logging
import sys
from pathlib import Path
from typing import List
from shutil import rmtree
import numpy as np
import time

_CODE_PATH = Path(__file__).parent.parent
if sys.path[-1].endswith('external-code-utils/detectron2'):
    sys.path = sys.path[:-1]
sys.path.append(str(_CODE_PATH))
sys.path.append(str(_CODE_PATH / 'external_libraries' / 'detectron2'))
sys.path.append(str(_CODE_PATH / 'external_libraries' / 'mmdetection'))

from seg_utils.os_utils import is_pycharm_and_vscode_hosted
import argparse
import json
import pandas as pd
from pycocotools.coco import COCO
import tempfile
from common.constants import RESOURCES_PATH, RAW_DATA_DIR_PATH, RESULTS_DIR_PATH, SEED
from common.retrieve_exp_info import retrieve_exp_info, get_iters_from_exp_info
import matplotlib.pyplot as plt
import cv2
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


def get_inference_dir(exp_id: str) -> Path:
    model_ann_inference_f_path = RESULTS_DIR_PATH / f'exp-{exp_id}' / 'inference' / 'coco_instances_results.json'
    if model_ann_inference_f_path.exists():
        return model_ann_inference_f_path.parent
    pattern_suffix = 'inference/coco_instances_results.json'
    files_paths_nominees = list((RESULTS_DIR_PATH / f'exp-{exp_id}').rglob(f'*-iters/{pattern_suffix}'))
    iters_names_list = list(map(lambda f_path: f_path.parts[-3][:-len('-iters')], files_paths_nominees))
    iters_mapped_names_list = list(map(
        lambda name: (
            int(name) if name.isdigit()
            else 1000 * int(name[:-1]) if (name[-1] == 'K' and name[:-1].isnumeric())
            else 1000000 * int(name[:-1]) if (name[-1] == 'M' and name[:-1].isnumeric())
            else name
        ),
        iters_names_list
    ))
    if any(map(lambda name: isinstance(name, int), iters_mapped_names_list)):
        iters_mapped_names_list = list(
            map(lambda name: name if isinstance(name, int) else -np.inf, iters_mapped_names_list))
    max_idx = np.argmax(iters_mapped_names_list)
    model_ann_inference_f_path = files_paths_nominees[max_idx]
    return model_ann_inference_f_path.parent


def anns_analysis(exp_id: str, num_of_images_to_output: int,
                  _remove_dir_if_existed = True,
                  _calc_metrics: bool = True,
                  _skip_if_exists: bool = True):
    logging.basicConfig(level=logging.INFO)
    exp_info = retrieve_exp_info(exp_id)
    dataset_name = exp_info.dataset

    val_orig_ann_f_path = RAW_DATA_DIR_PATH / dataset_name / 'val' / 'annotations.json'
    val_noisy_ann_f_path = RESOURCES_PATH / 'processed-resources' / dataset_name / 'val' / (
            exp_info.noise_name + '.json')
    inference_dir = get_inference_dir(exp_id)
    model_ann_inference_f_path = inference_dir / 'coco_instances_results.json'

    anns_images_out_dir = inference_dir / 'annotations-images'
    if _remove_dir_if_existed:
        try:
            rmtree(anns_images_out_dir)
        except FileNotFoundError:
            pass
    anns_images_out_dir.mkdir(parents=True, exist_ok=True)

    gt_coco = COCO(val_orig_ann_f_path)
    # noisy_coco = COCO(val_noisy_ann_f_path)

    dt_coco_dict = json.load(open(val_noisy_ann_f_path))
    dt_anns_list = json.load(open(model_ann_inference_f_path))
    for ann_id, ann in enumerate(dt_anns_list):
        ann['id'] = ann_id
        ann['iscrowd'] = 0
        ann['area'] = ann['bbox'][2] * ann['bbox'][3]
    dt_coco_dict['annotations'] = dt_anns_list
    logging.info(' creating coco detections anns...')
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as temp_json:
        json.dump(dt_coco_dict, temp_json)
        temp_json.flush()  # Ensure data is written
        dt_coco = COCO(temp_json.name)

    logging.info(' calculating evaluation metrics')
    if _calc_metrics:
        metrics_f_path = inference_dir / 'eval_res.txt'
        if _skip_if_exists and not metrics_f_path.exists():
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                coco_eval = COCOeval(gt_coco, dt_coco, iouType='segm')
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()

            with open(metrics_f_path, 'w') as f:
                f.write(output.getvalue())
        else:
            logging.info(' calc metrics already exists (skipping)')
    logging.info(' plotting annotations on images')
    images_ids = sorted(gt_coco.getImgIds())
    for img_id in tqdm(images_ids[:num_of_images_to_output], desc='plotting anns images'):
        np.random.seed(SEED)  # to have segmentation colors in the same colors
        img_info = gt_coco.loadImgs(img_id)[0]
        f_name = img_info['file_name']
        dst_f_path = anns_images_out_dir / f_name
        if _skip_if_exists and dst_f_path.exists():
            continue
        src_f_path = RAW_DATA_DIR_PATH / dataset_name / 'val' / 'images' / f_name

        image = cv2.imread(str(src_f_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis('off')

        ann_ids = dt_coco.getAnnIds(imgIds=img_id)
        anns = dt_coco.loadAnns(ann_ids)
        dt_coco.showAnns(anns, draw_bbox=False)
        # fig.show()
        fig.savefig(dst_f_path)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-id', type=str)
    parser.add_argument('--num-imgs', type=int, default=15,
                        help='number of annotations images to save',
                        dest='num_of_images_to_output')
    return parser.parse_args(argv)


if __name__ == '__main__':
    optional_kwargs = {}
    if is_pycharm_and_vscode_hosted():
        argv = [
            '--exp-id', '49',
            '--num-imgs', '60',
        ]
        optional_kwargs = {
            '_remove_dir_if_existed': False,
            '_calc_metrics': True,
        }
    else:
        argv = sys.argv[1:]
    args = parse_args(argv)
    print(args)
    start_time = time.time()
    anns_analysis(**vars(args), **optional_kwargs)
    time.sleep(5)
    print('job time: ', str(datetime.timedelta(seconds=time.time() - start_time)))
