from common.constants import PROCESSED_DATA_DIR_PATH
from pathlib import Path
import json
from tqdm import tqdm


def remove_coco_images_names_suffix(coco_dir_path: Path, src_coco_f_name, dst_coco_f_name):
    data = json.load(open(coco_dir_path / src_coco_f_name))
    suffix_to_remove = '_leftImg8bit'
    for im_dict in tqdm(data["images"]):
        orig_f_name_path = Path(im_dict['file_name'])
        orig_stem_name = orig_f_name_path.stem
        if orig_stem_name.endswith(suffix_to_remove):
            updated_f_name = orig_stem_name[:-len(suffix_to_remove)] + orig_f_name_path.suffix
        else:
            updated_f_name = orig_stem_name
        im_dict['file_name'] = updated_f_name
    with open(coco_dir_path / ('updated_' + dst_coco_f_name), 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    coco_dir_path = PROCESSED_DATA_DIR_PATH / 'cityscapes' / 'train'
    src_coco_f_name = 'v19.json'
    dst_coco_f_name = 'v19_updated.json'
    remove_coco_images_names_suffix(coco_dir_path, src_coco_f_name, dst_coco_f_name)
