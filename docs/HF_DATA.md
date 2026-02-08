# Hugging Face datasets

This repo provides code to generate / evaluate **noisy instance segmentation annotations** (COCO format).

We also publish curated datasets and noisy-label variants on the Hugging Face Hub.

## Datasets

> TODO: fill in the exact dataset repo IDs below (e.g. `username/dataset_name`).

- **COCO-N**: `<HF_DATASET_ID_COCO_N>`
- **VIPER-N**: `<HF_DATASET_ID_VIPER_N>`
- (Optional) **VIPER curated as COCO**: `<HF_DATASET_ID_VIPER_AS_COCO>`

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
