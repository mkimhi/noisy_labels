import contextlib
import io

from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from os import remove
from common.constants import PROCESSED_DATA_DIR_PATH, RAW_DATA_DIR_PATH, RESULTS_DIR_PATH
from pathlib import Path

from pycocotools.coco import COCO


def annotations_anylisys(anns_f_path_gt: Path, anns_f_path_dt: Path, output_f_path: Path) -> None:
    coco_gt = COCO(anns_f_path_gt)
    coco_dt = COCO(anns_f_path_dt)

    for cur_coco in [coco_gt, coco_dt]:
        for ann in cur_coco.anns.values():
            ann['score'] = 1.0

    output_f_path.parent.mkdir(parents=True, exist_ok=True)
    output = io.StringIO()
    if output_f_path.is_file():
        remove(output_f_path)
    with contextlib.redirect_stdout(output):
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        with open(output_f_path, 'w') as f:
            f.write(output.getvalue())
        return


def annotations_anylisys_wrapper(exp_id, dataset_name):
    exp_id = 'v' + str(exp_id)

    print(f'Running {exp_id}')
    anns_f_path_gt = RAW_DATA_DIR_PATH / dataset_name / 'val' / 'annotations.json'
    anns_f_path_dt = PROCESSED_DATA_DIR_PATH / dataset_name / 'val' / f'{exp_id}.json'
    output_f_path = RESULTS_DIR_PATH / 'analysis' / f'coco-val-noise-{exp_id}-vs-gt.txt'
    annotations_anylisys(anns_f_path_gt, anns_f_path_dt, output_f_path=output_f_path)
    print(open(output_f_path).read())


if __name__ == '__main__':
    exp_id = 20
    dataset_name = 'lvis'
    annotations_anylisys_wrapper(exp_id, dataset_name)