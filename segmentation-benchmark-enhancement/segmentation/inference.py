import argparse
import sys
from pathlib import Path


_CODE_PATH = Path(__file__).parent.parent
if sys.path[-1].endswith('external-code-utils/detectron2'):
    sys.path = sys.path[:-1]
sys.path.append(str(_CODE_PATH))
sys.path.append(str(_CODE_PATH / 'external_libraries' / 'detectron2'))
sys.path.append(str(_CODE_PATH / 'external_libraries' / 'mmdetection'))

from common.constants import RESULTS_DIR_PATH
from common.retrieve_exp_info import retrieve_exp_info, get_iters_from_exp_info
import config
from seg_utils.os_utils import is_pycharm_and_vscode_hosted
from tools.train_net import invoke_main as detectron2_main


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-number', dest='exp_id', type=str, required=True)
    parser.add_argument('--rnd-mode', action='store_true')
    args = parser.parse_args(argv)
    return args


def inference_model(exp_id: str, rnd_mode: bool, get_mock_exp_info: bool=False):
    exp_info = retrieve_exp_info(exp_id, get_mock_exp_info=get_mock_exp_info)
    iters = get_iters_from_exp_info(exp_info)
    inference_folder = RESULTS_DIR_PATH / f'exp-{exp_id}' / f'{iters}-iters' / 'inference'
    if exp_info.package == 'detectron2':
        detectron2_main(
            exp_id=exp_id,
            benchmark_name=exp_info.dataset,
            noise_name=exp_info.noise_name,
            rnd_mode=rnd_mode,
            new_train=False,
            bool_eval_only=True,
            inference_folder=inference_folder
        )
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    get_mock_exp_info = False
    optional_kwargs = {}
    if is_pycharm_and_vscode_hosted():
        argv = [
            '--exp-number', '1001',
        ]
        if config.RND_MODE:
            argv.append('--rnd-mode')
            get_mock_exp_info = True
    else:
        argv = sys.argv[1:]
    args = parse_args(argv)
    print(args)
    inference_model(**vars(args), get_mock_exp_info=get_mock_exp_info)
