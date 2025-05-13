import json

from tqdm import tqdm

from common.constants import RESOURCES_PATH


def adjust_cityscapes_files_names_in_json():
    dir_path = RESOURCES_PATH / 'data' / 'cityscapes' / 'train'
    orig_anns_f_path = dir_path / 'annotations.json'
    dst_anns_f_path = dir_path / 'updated_annotations.json'
    coco_dict = json.load(open(orig_anns_f_path))
    for f_name in tqdm(coco_dict['images']):
        registered_f_name = f_name['file_name']
        parts_list = registered_f_name.split('_')
        f_name['file_name'] = '_'.join(parts_list[:-3] + parts_list[-1:])
    with open(dst_anns_f_path, 'w') as f:
        json.dump(coco_dict, f, indent=4)

if __name__ == '__main__':
    adjust_cityscapes_files_names_in_json()
