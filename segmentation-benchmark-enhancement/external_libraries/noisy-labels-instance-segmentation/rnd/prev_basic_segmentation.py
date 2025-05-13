from common.constants import CODE_PATH
from mmdet.apis import DetInferencer
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS


def get_path_from_mm(str_path):
    mmdetection_path = CODE_PATH / 'external_libraries' / 'mmdetection'
    return str(mmdetection_path / str_path)


def basic_detection():
    # Choose to use a config
    model_name = 'rtmdet_tiny_8xb32-300e_coco'
    # Setup a checkpoint file to load
    checkpoint = get_path_from_mm('checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth')
    device = 'cpu'

    # Initialize the DetInferencer
    inferencer = DetInferencer(model_name, checkpoint, device)
    x = 1
    print(x)
    # Use the detector to do inference
    img = get_path_from_mm('./demo/demo.jpg')
    result = inferencer(img, out_dir=get_path_from_mm('./output'))

    x = 1
    return


def basic_segmentation():
    mmdet_path = CODE_PATH / 'external_libraries' / 'mmdetection'

    # Choose to use a config and initialize the detector
    config_file = str(mmdet_path / 'configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py')
    # Setup a checkpoint file to load
    checkpoint_file = str(mmdet_path / 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth')

    # register all modules in mmdet into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

    # Use the detector to do inference
    image = mmcv.imread(str(mmdet_path / 'demo/demo.jpg'), channel_order='rgb')
    result = inference_detector(model, image)
    print(result)

    # init visualizer(run the block only once in jupyter notebook)
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    # show the results
    visualizer.add_datasample(
        'result',
        image,
        data_sample=result,
        draw_gt=None,
        wait_time=0,
    )
    visualizer.show()


if __name__ == '__main__':
    basic_segmentation()
