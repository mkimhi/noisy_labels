import argparse
import sys
from pathlib import Path


_CODE_PATH = Path(__file__).parent.parent
if sys.path[-1].endswith('external-code-utils/detectron2'):
    sys.path = sys.path[:-1]
sys.path.append(str(_CODE_PATH))
sys.path.append(str(_CODE_PATH / 'external_libraries' / 'detectron2'))
sys.path.append(str(_CODE_PATH / 'external_libraries' / 'mmdetection'))

import config
from seg_utils.os_utils import is_pycharm_and_vscode_hosted
from common.constants import CHECKPOINTS_DIR_PATH, MMDETECTION, DETECTRON2, RAW_DATA_DIR_PATH
from tools.train_net import invoke_main as detectron2_main
from pycocotools.coco import COCO

from exp_params import ExpParams


def segmentation_model_wrapper(models_wrappers_package, args_namespace, optional_kwargs) -> None:
    if models_wrappers_package == MMDETECTION:
        from external_libraries.mmdetection.tools.train import mmdetection_train_main
        from external_libraries.mmdetection.custom_code_mmdetection.create_config_file import \
            create_config_file_mmdetection
        exp_params = ExpParams(
            max_epochs=1,
            num_workers=1,
            models_wrappers_package=DETECTRON2,
            noise_name='spatial-medium'
        )
        noised_annotations_path = RAW_DATA_DIR_PATH / 'coco' / 'val' / 'annotations.json'
        coco_dataset = COCO(noised_annotations_path)
        config_f_path = CHECKPOINTS_DIR_PATH / exp_params.models_wrappers_package / f'local_segmentation_config_{exp_params.noise_name}.py'
        override_config_file = True
        if override_config_file or not config_f_path.is_file():
            create_config_file_mmdetection(config_f_path, exp_params, coco_dataset)
        mmdetection_train_main(config_f_path)
    elif models_wrappers_package == DETECTRON2:
        num_gpus = 1
        optional_kwargs['num_gpus'] = optional_kwargs.get('num_gpus', num_gpus)
        default_lr = 0.02 * (num_gpus * 2) / 16 if num_gpus > 0 else 0.02 * (1 * 2) / 16 # equation from https://github.com/facebookresearch/detectron2/issues/1128
        optional_kwargs['lr'] = optional_kwargs.pop('lr', args_namespace.lr or default_lr)
        if args_namespace.rnd_mode:
            optional_kwargs['checkpoint_period'] = 1
        args = vars(args_namespace)
        del args['lr']
        detectron2_main(bool_eval_only=False, **args, **optional_kwargs)
    else:
        raise ValueError(f'Unknown model {models_wrappers_package}')
    return


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-name', type=str, required=True,
                        choices=['easy-amount', 'easy-params', 'medium-amount', 'medium-params', 'hard'])
    parser.add_argument('--exp-id', type=int, required=True)
    parser.add_argument('--benchmark-name', type=str, required=True, choices=['coco', 'cityscapes'])
    parser.add_argument('--rnd-mode', action='store_true')
    parser.add_argument('--new-train', action='store_true')
    parser.add_argument('--lr', type=float, default=None)
    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    optional_kwargs = {}
    if is_pycharm_and_vscode_hosted():
        argv = [
            '--noise-name', 'hard',
            '--exp-id', '999',
            '--benchmark-name', 'coco',
            '--new-train',
        ]
        if config.RND_MODE:
            argv.append('--rnd-mode')
            optional_kwargs['max_iter'] = 2
    else:
        # argv = (CODE_RESOURCES_PATH / 'input_params' / 'user_argv.txt').read_text().splitlines()
        argv = sys.argv[1:]
    args = parse_args(argv)
    print(args)
    print(optional_kwargs)
    segmentation_model_wrapper(models_wrappers_package=DETECTRON2, args_namespace=args, optional_kwargs=optional_kwargs)
