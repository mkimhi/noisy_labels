import random

from pycocotools.coco import COCO
from common.constants import PROCESSED_DATA_DIR_PATH, CODE_RESOURCES_PATH, SEED, RAW_DATA_DIR_PATH, \
    CHECKPOINTS_DIR_PATH, MMDETECTION
from mmengine import Config
from mmengine.runner import set_random_seed
from pathlib import Path

from exp_params import ExpParams

def create_config_file_mmdetection(config_f_path: Path,
                                   exp_params: ExpParams, coco_dataset: COCO) -> None:
    categories = coco_dataset.loadCats(coco_dataset.getCatIds())
    categories_list = list(map(lambda cat: cat['name'], sorted(categories, key=lambda cat: cat['id'])))

    cfg = Config.fromfile(CHECKPOINTS_DIR_PATH / MMDETECTION / 'my-mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py')

    random.seed(SEED)
    cfg.metainfo = {
        'classes': tuple(categories_list),
        'palette': [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(len(categories_list))]
    }
    ###############
    cfg.train_dataloader.dataset.ann_file = str(PROCESSED_DATA_DIR_PATH / 'coco' / 'train' / f'{exp_params.noise_name}.json')
    cfg.train_dataloader.dataset.data_root = str(RAW_DATA_DIR_PATH / 'coco' / 'train' / 'images')
    cfg.train_dataloader.dataset.metainfo = cfg.metainfo

    cfg.val_dataloader.dataset.ann_file = str(PROCESSED_DATA_DIR_PATH / 'coco' / 'val' / f'{exp_params.noise_name}.json')
    cfg.val_dataloader.dataset.data_root = str(RAW_DATA_DIR_PATH / 'val' / 'coco' / 'images')
    cfg.val_dataloader.dataset.metainfo = cfg.metainfo

    cfg.test_dataloader = cfg.val_dataloader

    cfg.val_evaluator.ann_file = str(PROCESSED_DATA_DIR_PATH / 'coco' / 'val' / f'{exp_params.noise_name}.json')
    cfg.test_evaluator = cfg.val_evaluator

    # Modify num classes of the model in box head and mask head
    cfg.model.roi_head.bbox_head.num_classes = len(categories_list)
    cfg.model.roi_head.mask_head.num_classes = len(categories_list)

    # We can still the pre-trained Mask RCNN model to obtain a higher performance
    # cfg.load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
    cfg.load_from = str(CHECKPOINTS_DIR_PATH / MMDETECTION / 'mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth')

    # Set up working dir to save files and logs.
    cfg.work_dir = str(CODE_RESOURCES_PATH / 'logs')

    # We can set the evaluation interval to reduce the evaluation times
    cfg.train_cfg.val_interval = 3
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.default_hooks.checkpoint.interval = 3

    cfg.optim_wrapper.optimizer.lr = 2.5e-3
    cfg.default_hooks.logger.interval = 10

    # Set seed thus the results are more reproducible
    # cfg.seed = 0
    set_random_seed(SEED, deterministic=False)

    # We can also use tensorboard to log the training process
    cfg.visualizer.vis_backends.append({"type": 'TensorboardVisBackend'})

    cfg._cfg_dict.train_cfg.max_epochs = exp_params.max_epochs
    cfg._cfg_dict.train_dataloader.num_workers = exp_params.num_workers
    cfg._cfg_dict.test_dataloader.num_workers = exp_params.num_workers

    with open(config_f_path, 'w') as f:
        f.write(cfg.pretty_text)
    return