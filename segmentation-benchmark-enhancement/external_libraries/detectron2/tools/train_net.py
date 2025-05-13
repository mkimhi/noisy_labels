#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import sys
from pathlib import Path
from shutil import rmtree

_CODE_PATH = Path(__file__).parent.parent.parent.parent

# sys.path.append(str(_CODE_PATH))
# sys.path.append(str(_CODE_PATH / 'external_libraries' / 'detectron2'))
# sys.path.append(str(_CODE_PATH / 'external_libraries' / 'mmdetection'))
import logging
import os
from typing import Union
from collections import OrderedDict
import importlib
import detectron2.utils.comm as comm
from common.constants import PROCESSED_DATA_DIR_PATH, RAW_DATA_DIR_PATH, RESULTS_DIR_PATH, CPU, CODE_PATH
import config
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    suffix = Path(args.config_file).suffix
    if suffix == ".yaml":
        cfg.merge_from_file(args.config_file)
    elif suffix == '.py':
        config_module = importlib.import_module(args.config_file)
        # todo to fix
    else:
        raise NotImplementedError()

    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def apply_kwargs_to_cfg(cfg, kwargs):
    try:
        val = kwargs['train_dataset_name']
    except ValueError:
        pass
    else:
        if val is not None:
            cfg.DATASETS.TRAIN = (val,)
    try:
        val = kwargs['val_dataset_name']
    except ValueError:
        pass
    else:
        if val is not None:
            cfg.DATASETS.TEST = (val,)
    try:
        val = kwargs['input_mask_format']
    except ValueError:
        pass
    else:
        if val is not None:
            cfg.INPUT.MASK_FORMAT = val
    try:
        val = kwargs['output_dir']
    except ValueError:
        pass
    else:
        if val is not None:
            cfg.OUTPUT_DIR = val
    try:
        val = kwargs['max_iter']
    except ValueError:
        pass
    else:
        if val is not None:
            cfg.SOLVER.MAX_ITER = val
    try:
        val = kwargs['images_per_batch']
    except ValueError:
        pass
    else:
        if val is not None:
            cfg.SOLVER.IMS_PER_BATCH = val
    try:
        val = kwargs['checkpoint_period']
    except ValueError:
        pass
    else:
        if val is not None:
            cfg.SOLVER.CHECKPOINT_PERIOD = val
    try:
        val = kwargs['lr']
    except ValueError:
        pass
    else:
        if val is not None:
            cfg.SOLVER.BASE_LR = val
    try:
        val = kwargs['max_cp_to_keep']
    except ValueError:
        pass
    else:
        if val is not None:
            cfg.SOLVER.MAX_CP_TO_KEEP = val


def main(args, rnd_mode=False, inference_folder=None, **kwargs):
    cfg = setup(args)
    cfg.defrost()

    apply_kwargs_to_cfg(cfg, kwargs)

    cfg.freeze()

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model, rnd_mode=rnd_mode, inference_folder=inference_folder)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json


def register_custom_dataset(dataset_name, annotations_f_path, images_root_path):
    # dataset_name = "my_custom_coco_train"
    # json_file = "/path/to/your/annotations/instances_train2017.json"
    # image_root = "/path/to/your/images/"

    DatasetCatalog.register(
        dataset_name,
        lambda: load_coco_json(annotations_f_path, images_root_path, dataset_name)
    )
    MetadataCatalog.get(dataset_name).set(
        json_file=annotations_f_path,
        image_root=images_root_path,
        evaluator_type="coco"
    )
    # print(f"Registered dataset '{dataset_name}' with json: {json_file} and images: {image_root}")
    return


def invoke_main(exp_id, benchmark_name, noise_name, rnd_mode, new_train, bool_eval_only,
                max_iter=None, checkpoint_period=1000, lr=None, num_gpus=None,
                inference_folder: Union[str, Path]=None) -> None:
    if isinstance(inference_folder, Path):
        inference_folder = str(inference_folder)

    model_name = 'mask_rcnn_R_50_FPN_3x'
    # model_name = 'mask_rcnn_R_50_FPN_50ep_LSJ'
    images_per_batch = 4

    exp_str = f'exp-{exp_id}'

    kwargs = dict()
    kwargs['max_cp_to_keep'] = 2
    kwargs['lr'] = lr
    kwargs['input_mask_format'] = 'bitmask'
    kwargs['output_dir'] = str(RESULTS_DIR_PATH / exp_str)
    if new_train:
        try:
            rmtree(kwargs['output_dir'])
        except FileNotFoundError:
            pass
    kwargs['max_iter'] = max_iter
    kwargs['images_per_batch'] = images_per_batch
    kwargs['checkpoint_period'] = checkpoint_period
    dataset_name = benchmark_name + '-' + noise_name
    for phase in ['val', 'train']:
        dataset_phase_name = '+'.join([dataset_name, phase])
        register_custom_dataset(
            dataset_name=dataset_phase_name,
            annotations_f_path=str(PROCESSED_DATA_DIR_PATH / benchmark_name / phase / (noise_name + '.json')),
            images_root_path=str(RAW_DATA_DIR_PATH / benchmark_name / phase / 'images')
        )
        kwargs[phase + '_dataset_name'] = dataset_phase_name
    parser = default_argument_parser()
    # if argsname is None:
    #     args = parser.parse_args()  # get sys args
    # else:  # case we already used the sys args
    #     args = parser.parse_args(args=[])
    args = parser.parse_args(args=[])

    config_file_prefix = CODE_PATH / 'external_libraries' / 'detectron2'
    if model_name in [
        'mask_rcnn_R_50_FPN_3x',
    ]:
        args.config_file = str(config_file_prefix / f'configs/COCO-InstanceSegmentation/{model_name}.yaml')
    elif model_name in [
        'mask_rcnn_R_50_FPN_50ep_LSJ',
    ]:
        args.config_file = str(config_file_prefix / f'configs/new_baselines/{model_name}.py')
    else:
        raise NotImplementedError()

    if num_gpus is None:
        num_gpus = args.num_gpus
    if config.DEVICE.type == CPU:
        args.num_gpus = 0
        args.opts.extend(["MODEL.DEVICE", CPU])
    else:
        args.num_gpus = num_gpus
    args.eval_only = bool_eval_only
    args.resume = not new_train

    # print("Command Line Args:", args)
    launch(
        lambda args: main(args, rnd_mode=rnd_mode, inference_folder=inference_folder, **kwargs),
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
