# Noisy-Labels-Instance-Segmentation
## This is the official repo for the paper A Benchmark for Learning with Noisy Labels in Instance Segmentation

![paper meme](https://github.com/mkimhi/noisy_labels/blob/main/meme.jpg)

### ReadMe:
Important! The original annotations should be in coco format.

To run the benchmark, run the following:
```
python noise_annotations.py /path/to/annotations --benchmark {easy, medium, hard} (choose the benchmark level) --seed 1
```

For example:
```
python noise_annotations.py /path/to/annotations --benchmark easy --seed 1
```


To run a custom noise method, run the following:
```
python noise_annotations.py /path/to/annotations --method_name method_name --corruption_values [{'rand': [scale_proportion, kernel_size(should be odd number)],'localization': [scale_proportion, std_dev], 'approximation': [scale_proportion, tolerance], 'flip_class': percent_class_noise}]}]
```

For example:
```
 python noise_annotations.py /path/to/annotations --method_name my_noise_method --corruption_values [{'rand': [0.2, 3], 'localization': [0.2, 2], 'approximation': [0.2, 5], 'flip_class': 0.2}]
```

To run the coco-wan, do the following:

download the weights for sam vit-h:
```
mkdir -p weights
cd weights
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
Create a conda env with the required libraries:
```
 conda env create -f wan_environment.yml

 conda activate cuda_env
```

For the easy noise, run create_gt_plus_sam_noise.py:
```
python create_gt_plus_sam_noise.py --annotations_path=a_path --data_path=d_path --sam_path=s_path
```

For example:
```
python create_gt_plus_sam_noise.py --annotations_path=data/coco_ann2017/annotations --data_path=data --sam_path=weights/sam_vit_h_4b8939.pth
```

For the medium noise, do the exact same thing but with the file create_gt_plus_sam_point_noise.py


For the hard noise, apply class noise on top of the medium weak annotation noise by running noise_annotations.py with the proper arguments.




## Benchmark
![image](https://github.com/mkimhi/noisy_labels/blob/main/chicken.pdf)



**Table 1: Evaluation Results of Instance Segmentation Models under Different Benchmarks (mAP)**

*Note: CS-N stands for Cityscapes benchmark.*

| Dataset | Model  | Backbone | Clean | Easy | Mid  | Hard |
|---------|--------|----------|-------|------|------|------|
| **COCO-N** | M-RCNN | R-50     | 34.6  | 27.9 | 24.8 | 22.3 |
|         | YOLACT | R-50     | 28.5  | 26.4 | 23.3 | 20.8 |
|         | SOLO   | R-50     | 35.9  | 25.2 | 17.1 | 12.4 |
|         | HTC    | R-50     | 34.1  | -    | 28.4 | 25.5 |
|         | M2F    | R-50     | 42.9  | 33.5 | 30.1 | 26.7 |
|         | M-RCNN | R-101    | 36.2  | 28.8 | 31.8 | 23.7 |
|         | M2F    | Swin-S   | 46.1  | 39.6 | 37.9 | 33.6 |
|---------|--------|----------|-------|------|------|------|
| **CS-N** | M-RCNN | R-50     | 36.1  | 26.4 | 22.0 | 16.3 |
|         | YOLACT | R-50     | 19.3  | 19.1 | 17.1 | 16.3 |
|         | M-RCNN | R-101    | 37.0  | 33.7 | 30.7 | 27.0 |

# Synthetic data
![image](https://github.com/mkimhi/noisy_labels/blob/main/viper.pdf)

<object data="[http://yoursite.com/the.pdf](https://github.com/mkimhi/noisy_labels/blob/main/viper.pdf)" type="application/pdf" width="700px" height="700px">
    <embed src="[http://yoursite.com/the.pdf](https://github.com/mkimhi/noisy_labels/blob/main/viper.pdf)">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="http://yoursite.com/the.pdf">Download PDF</a>.</p>
    </embed>
</object>


Upon acceptance we will publish the work with Synthetic data and noising it


## Citation


If you use this benchmark in your research, please cite this project.


```
Soon
```


## License

This project is released under the Apache 2.0 license.


Please make sure you use it with proper licenced Datasets.

We use [MS-COCO/LVIS](https://cocodataset.org/#termsofuse) and [Cityscapes](https://www.cityscapes-dataset.com/license/)


