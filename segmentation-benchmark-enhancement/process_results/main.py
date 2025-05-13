from pathlib import Path
import json
from typing import Tuple, Dict

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from typing import Optional
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common.constants import RAW_DATA_DIR_PATH, RESULTS_DIR_PATH, PROCESSED_DATA_DIR_PATH, CODE_RESOURCES_PATH


def _write_results_df(results_df: pd.DataFrame, results_csv_f_path: Path) -> None:
    '''
    load the df from the csv file, add the new results, and write the combined df to the file
    '''
    try:
        prev_df = pd.read_csv(results_csv_f_path)
    except Exception:
        combined_df = results_df
    else:
        combined_df = pd.concat([results_df, prev_df], ignore_index=True)
    combined_df.to_csv(results_csv_f_path, index=False)
    return


def write_results_df(results_df: pd.DataFrame, results_csv_f_dir_path: Path, results_csv_f_stem: str) -> None:
    f_name = results_csv_f_stem + '.csv'
    results_csv_f_path = results_csv_f_dir_path / f_name
    try:
        _write_results_df(results_df, results_csv_f_path)
    except Exception:
        files_parts_list = list(results_csv_f_dir_path.rglob(results_csv_f_stem + '*'))
        base_part_name = results_csv_f_stem + '_part_'
        base_len = len(base_part_name)
        parts_ints_list = map(lambda f_path: int(f_path.stem[base_len:]), files_parts_list)
        try:
            last_part_int = max(parts_ints_list)
        except ValueError:
            new_part_idx = 0
        else:
            new_part_idx = last_part_int + 1
        f_name = results_csv_f_stem + f'_part_{new_part_idx}.csv'
        results_csv_f_path = results_csv_f_dir_path / f_name
        _write_results_df(results_df, results_csv_f_path)
    return


def produce_segmentation_results_images(images_dir: Path, res_annotation_f_path: Path, gt_annotation_f_path: Path,
                                        images_dst_dir_path: Path, num_samples: int,
                                        boundary_type: Optional[str] = None) -> None:
    res_annotations_list = json.load(open(res_annotation_f_path))
    gt_coco = COCO(gt_annotation_f_path)

    if boundary_type:
        target_dir = images_dst_dir_path / boundary_type
    else:
        target_dir = images_dst_dir_path / 'all'

    target_dir.mkdir(parents=True, exist_ok=True)
    shown_images_list = []
    counter = 0

    for image_info in gt_coco.dataset['images']:
        image_id = image_info['id']
        gt_anns = gt_coco.imgToAnns.get(image_id, [])

        if boundary_type:
            if not any(a['boundary_type'] == boundary_type for a in gt_anns):
                continue

        if image_id in shown_images_list:
            continue

        res_anns = [a for a in res_annotations_list if a['image_id'] == image_id]
        if not res_anns:
            continue

        shown_images_list.append(image_id)
        image_path = images_dir / image_info['file_name']
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        for a in res_anns:
            if 'bbox' in a:
                x, y, width, height = a['bbox']
                draw.rectangle([(x, y), (x + width, y + height)], outline="red", width=2)
            category_name = gt_coco.loadCats(a['category_id'])[0]['name']
            text_position = (x, y - 10) if y - 10 > 0 else (x, y + 5)
            draw.text(text_position, category_name, fill="red")

        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.axis('off')
        output_path = target_dir / f"{image_info['file_name'].split('.')[0]}_annotated.png"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        counter += 1
        if counter >= num_samples:
            break


def produce_segmentation_results_images_combined(images_dir: Path, res_annotation_f_path: Path,
                                                 gt_annotation_f_path: Path,
                                                 images_dst_dir_path: Path, num_samples: int) -> None:
    produce_segmentation_results_images(
        images_dir, res_annotation_f_path, gt_annotation_f_path,
        images_dst_dir_path, num_samples
    )

    gt_coco = COCO(gt_annotation_f_path)
    boundary_types = set(ann['boundary_type'] for ann in gt_coco.anns.values())

    for boundary_type in boundary_types:
        produce_segmentation_results_images(
            images_dir, res_annotation_f_path, gt_annotation_f_path,
            images_dst_dir_path, num_samples, boundary_type
        )

def calc_metrics(res_annotation_f_path: Path, gt_annotation_f_path: Path) -> Tuple[float, float]:
    gt_coco = COCO(gt_annotation_f_path)
    res_coco = gt_coco.loadRes(str(res_annotation_f_path))

    coco_eval = COCOeval(gt_coco, res_coco, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    box_ap = coco_eval.stats[0]

    coco_eval = COCOeval(gt_coco, res_coco, iouType='segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mask_ap = coco_eval.stats[0]

    return box_ap, mask_ap

def calc_ap_with_boundary_types(res_annotation_f_path: Path, gt_annotation_f_path: Path) -> Dict[
    str, Dict[str, float]]:
    gt_coco = COCO(gt_annotation_f_path)
    res_coco = gt_coco.loadRes(str(res_annotation_f_path))

    ap_results = {}

    ### Overall AP (All Annotations)
    # BBox Evaluation
    coco_eval_bbox = COCOeval(gt_coco, res_coco, iouType='bbox')
    coco_eval_bbox.evaluate()
    coco_eval_bbox.accumulate()
    coco_eval_bbox.summarize()
    overall_bbox_ap = coco_eval_bbox.stats[0]

    # Mask Evaluation
    coco_eval_mask = COCOeval(gt_coco, res_coco, iouType='segm')
    coco_eval_mask.evaluate()
    coco_eval_mask.accumulate()
    coco_eval_mask.summarize()
    overall_mask_ap = coco_eval_mask.stats[0]

    ap_results['overall'] = {
        'bbox_ap': overall_bbox_ap,
        'mask_ap': overall_mask_ap
    }

    ### Per Boundary Type AP
    boundary_types = set(ann['boundary_type'] for ann in gt_coco.anns.values())

    for boundary_type in boundary_types:
        # Filter annotations by boundary_type
        filtered_ann_ids = [
            ann_id for ann_id, ann in gt_coco.anns.items()
            if ann['boundary_type'] == boundary_type
        ]

        # Create subset COCO for ground truth
        gt_subset = gt_coco.loadAnns(filtered_ann_ids)
        gt_coco_subset = COCO()
        gt_coco_subset.dataset = {
            'images': [img for img in gt_coco.dataset['images'] if img['id'] in {ann['image_id'] for ann in gt_subset}],
            'annotations': gt_subset,
            'categories': gt_coco.dataset['categories']
        }
        gt_coco_subset.createIndex()

        # Evaluate BBox
        coco_eval_bbox = COCOeval(gt_coco_subset, res_coco, iouType='bbox')
        coco_eval_bbox.evaluate()
        coco_eval_bbox.accumulate()
        coco_eval_bbox.summarize()
        bbox_ap = coco_eval_bbox.stats[0]

        # Evaluate Mask
        coco_eval_mask = COCOeval(gt_coco_subset, res_coco, iouType='segm')
        coco_eval_mask.evaluate()
        coco_eval_mask.accumulate()
        coco_eval_mask.summarize()
        mask_ap = coco_eval_mask.stats[0]

        ap_results[boundary_type] = {
            'bbox_ap': bbox_ap,
            'mask_ap': mask_ap
        }

    return ap_results


def evaluate_results_wrapper():
    noise_name = 'spatial-medium'
    exps_to_evaluate_list = [0, 1]

    bool_plot_images = True
    bool_calc_metrics = False

    images_dir = RAW_DATA_DIR_PATH / 'coco' / 'val' / 'images'
    num_samples = 50

    results_csv_f_dir_path = CODE_RESOURCES_PATH / 'exps_summary'
    results_csv_f_stem = 'results_data'

    results_data = []
    for exp_num in exps_to_evaluate_list:
        gt_annotation_f_path = PROCESSED_DATA_DIR_PATH / 'coco' / 'val' / (noise_name + '.json')
        res_annotation_f_path = RESULTS_DIR_PATH / f'exp-{exp_num}' / 'inference' / 'coco_instances_results.json'
        images_dst_dir_path = RESULTS_DIR_PATH / f'exp-{exp_num}' / 'images' / 'val'
        if bool_plot_images:
            produce_segmentation_results_images_combined(images_dir, res_annotation_f_path, gt_annotation_f_path,
                                                images_dst_dir_path, num_samples)
        if bool_calc_metrics:
            # box_ap, mask_ap = calc_metrics(res_annotation_f_path, gt_annotation_f_path)
            # results_data.append({'exp_num': exp_num, 'box_ap': box_ap, 'mask_ap': mask_ap})
            d = calc_ap_with_boundary_types(res_annotation_f_path, gt_annotation_f_path)
            for boundary_type, vals in d.items():
                results_data.append({'exp_num': exp_num, 'boundary_type': boundary_type, 'box_ap': vals['bbox_ap'], 'mask_ap': vals['mask_ap']})
    if bool_calc_metrics:
        results_df = pd.DataFrame(results_data)
        write_results_df(results_df, results_csv_f_dir_path, results_csv_f_stem)
    return


if __name__ == "__main__":
    evaluate_results_wrapper()
