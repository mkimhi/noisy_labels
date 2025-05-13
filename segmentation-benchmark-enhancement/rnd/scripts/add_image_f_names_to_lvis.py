import json
from os import remove

from tqdm import tqdm
from common.constants import RESOURCES_PATH
from pycocotools.coco import COCO
from shutil import copyfile

def add_image_f_names_to_lvis():
    lvis_val_dir_path = RESOURCES_PATH / 'data' / 'lvis' / 'val'
    lvis_train_dir_path = RESOURCES_PATH / 'data' / 'lvis' / 'train'
    lvis_val_images_dir_path = lvis_val_dir_path / 'images'
    lvis_train_images_dir_path = lvis_train_dir_path / 'images'

    lvis_val_dict = json.load(open(lvis_val_dir_path / 'annotations.json'))
    lvis_train_coco = COCO(lvis_train_dir_path / 'annotations.json')
    for lvis_im_dict in tqdm(lvis_val_dict['images']):
        im_id = lvis_im_dict['id']
        f_name = f'{im_id:012}.jpg'
        lvis_im_dict['file_name'] = f_name
        if not (lvis_val_images_dir_path / f_name).is_file():
            try:
                _ = lvis_train_coco.loadImgs(im_id)
            except KeyError:
                # meaning the image id is not in the train - as expected
                pass
            else:
                raise KeyError('image id is in val and train')
            print(im_id)
            copyfile(lvis_train_images_dir_path / f_name, lvis_val_images_dir_path / f_name)
            remove(lvis_train_images_dir_path / f_name)

    # with open(lvis_dir_path / updated_lvis_f_name, 'w') as f:
    #     json.dump(lvis_dict, f, indent=4)


if __name__ == '__main__':
    add_image_f_names_to_lvis()
