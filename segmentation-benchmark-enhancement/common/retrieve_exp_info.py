import logging
from os import listdir
import warnings
from typing import Union

import pandas as pd

from common.constants import CODE_RESOURCES_PATH, RESULTS_DIR_PATH


def retrieve_exp_info(exp_id, allow_reading_while_file_open: bool = False, get_mock_exp_info: bool = False) -> pd.Series:
    if get_mock_exp_info:
        logging.warn('retrieving mock exp info!!')
        return pd.Series({
            'exp_id': exp_id,
            'package': 'detectron2',
            'dataset': 'coco',
            'noise_name': 'hard',
        })

    exps_summary_dir = CODE_RESOURCES_PATH / 'exps_summary'
    if allow_reading_while_file_open:
        warnings.warn('when _allow_reading_while_file_open not sure most updated'
                      'history version is read, and not sure algo can detect'
                      'if file is open.')
    # elif f'.~lock.exps-summary.xlsx#' in listdir(exps_summary_dir):
    #     raise OSError(f'exps-summary.xlsx is open. To ignore: set _allow_reading_while_file_open to True')
    exps_info_f_path = exps_summary_dir / 'exps-summary.xlsx'
    all_exps_info_df = pd.read_excel(exps_info_f_path)
    relevant_exps_info_df = all_exps_info_df[all_exps_info_df['exp_id'].astype(str) == str(exp_id)]
    if len(relevant_exps_info_df) == 0:
        raise ValueError(f'Exp {exp_id} has no experiment registered in the exps-summary.')
    if len(relevant_exps_info_df) > 1:
        raise ValueError(f'Exp {exp_id} has more than one experiment.')
    exp_info = relevant_exps_info_df.iloc[0]
    return exp_info

def get_iters_from_exp_info(exp_info: pd.Series) -> Union[str, int]:
    last_cp_str_f_path = RESULTS_DIR_PATH / f'exp-{exp_info.exp_id}' / 'last_checkpoint'
    try:
        last_cp_f_name = last_cp_str_f_path.read_text()  # form of model_0193999.pth or model_final.pth
    except FileNotFoundError:
        unknown_num_iters = True
    else:
        if last_cp_f_name == 'model_final.pth':
            unknown_num_iters = True
        else:
            unknown_num_iters = False
            last_cp_iter = int(last_cp_f_name[len('model_'):-len('.pth')])
            iters = last_cp_iter
    if unknown_num_iters:
        iters = exp_info.cur_iters
    if isinstance(iters, int):
        if (iters >= 99) and (iters % 10 == 9):
            iters += 1
        if iters % 1000 == 0:
            iters = f'{int(iters / 1000)}K'
    return iters