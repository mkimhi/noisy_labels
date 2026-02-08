# Hugging Face datasets

This repo provides code to generate / evaluate **noisy instance segmentation annotations** (COCO format).

We also publish curated datasets and noisy-label variants on the Hugging Face Hub.

## Datasets

Hugging Face collection:
- https://huggingface.co/collections/Kimhi/noisy-labels-for-instance-segmentation-coco-format

Datasets:
- **COCO-N**: (see collection)
- **VIPER-N**: (see collection)
- **VIPER curated as COCO**: (see collection)

## Quick usage

### Option A: Hugging Face `datasets`

```python
from datasets import load_dataset

# Example (replace with your dataset id)
# ds = load_dataset("<HF_DATASET_ID_COCO_N>")
```

### Option B: `huggingface_hub` snapshot download

```bash
pip install -U huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='<HF_DATASET_ID_COCO_N>', repo_type='dataset', local_dir='data/COCO-N')"
```

## Expected layout

The benchmark expects **COCO-format** annotations. If a dataset provides images, keep paths consistent with the COCO JSON `file_name` entries.

## Integrity notes

If you are uploading large image collections to the Hub with Git LFS, prefer:
- verifying that all files exist before `git lfs push`
- uploading in smaller batches
- using `huggingface_hub` for large-scale uploads when possible
