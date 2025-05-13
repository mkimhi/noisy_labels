from common.constants import RAW_DATA_DIR_PATH, PROCESSED_DATA_DIR_PATH, DEVICE, CHECKPOINTS_DIR_PATH, \
    MMDETECTION
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

def segmentation_one_im():
    splits = ['val']
    sample_image_f_name = '000000000885.jpg'
    for split in splits:
        images_path = RAW_DATA_DIR_PATH / 'coco' / split / 'images'
        noised_annotations_path = PROCESSED_DATA_DIR_PATH / 'coco' / f'annotations_{split}_spatial_noise.json'
        config_file = CHECKPOINTS_DIR_PATH / MMDETECTION / 'mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py'
        checkpoint_file = CHECKPOINTS_DIR_PATH / MMDETECTION / 'mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
        # register all modules in mmdet into the registries
        register_all_modules()
        model = init_detector(str(config_file), str(checkpoint_file), device=DEVICE)

        image = mmcv.imread(images_path / sample_image_f_name, channel_order='rgb')
        result = inference_detector(model, image)
        print(result)

if __name__ == '__main__':
    segmentation_one_im()
