# LaneDet: Real-time Lane Detection Benchmarking

LaneDet is a computer vision project for real-time lane detection using deep learning based semantic segmentation.

The project benchmarks multiple lightweight lane segmentation models on the TuSimple dataset and provides a simple Gradio demo for visual inference.

## Project Overview

The goal of this project is to compare different lane detection model architectures in terms of accuracy and real-time performance.

Models implemented:

- U-Net baseline
- CLRNet-style context refinement model
- LaneATT-style attention model
- SCNN-style spatial context model

The models are trained on 3,626 annotated TuSimple lane detection images.

## Results

Final benchmark on 3,626 annotated TuSimple images:

| Model | IoU | F1-score | Precision | Recall | FPS |
|---|---:|---:|---:|---:|---:|
| U-Net | 0.3889 | 0.5590 | - | - | 91.5 |
| CLRNet-style | 0.3835 | 0.5536 | - | - | - |
| LaneATT-style | 0.3455 | 0.5128 | - | - | 154.8 |
| SCNN-style | 0.1482 | 0.2572 | - | - | 117.2 |

Best model: **U-Net**, achieving **0.3889 IoU**, **0.5590 F1-score**, and **91.5 FPS**.

## Demo

The project includes a local Gradio demo where users can upload a road image, select a model, and view the predicted lane overlay.

Add your screenshot here:

```markdown
![LaneDet Demo](assets/demo_screenshot.png)
```

## Project Structure

```text
LaneDet/
├── app/
│   └── gradio_app.py
├── assets/
│   └── demo_screenshot.png
├── checkpoints/
│   ├── unet_best.pt
│   ├── clrnet_best.pt
│   └── laneatt_best.pt
│   └── scnn_best.pt
├── outputs/
│   ├── clrnet_preds/
│   ├── laneatt_preds/
│   ├── scnn_preds/
│   ├── unet_preds/
│   └── mask_vis/
│   ├── clrnet_train_log.json
│   ├── laneatt_eval.json
│   └── laneatt_train_log.json
│   └── scnn_eval.json
│   └── scnn_train_log.json
│   └── unet_eval.json
│   └── unet_train_log.json
├── runlogs/
├── scripts/
│   ├── verify_dataset.py
│   ├── visualize_masks.py
│   ├── train_unet.py
│   ├── eval_unet.py
│   ├── train_clrnet.py
│   ├── eval_clrnet.py
│   ├── train_laneatt.py
│   ├── eval_laneatt.py
│   ├── train_scnn.py
│   └── eval_scnn.py
├── src/
│   ├── paths.py
│   ├── lane_mask.py
│   ├── tusimple_dataset.py
│   ├── metrics.py
│   ├── unet.py
│   ├── clrnet_seg.py
│   ├── laneatt_seg.py
│   └── scnn.py
├── requirements.txt
├── app.sub
├── evaluate.sub
├── evaluate_clrnet.sub
├── evaluate_laneatt.sub
├── evaluate_scnn.sub
├── run_app.sh
├── run_train.sh
├── run_train_clrnet.sh
├── run_train_laneatt.sh
├── run_train_scnn.sh
```

## Dataset

This project uses the TuSimple lane detection dataset.

The training labels are created from:

```text
label_data_0313.json
label_data_0531.json
label_data_0601.json
```

Each annotated sample contains:

- lane x-coordinates
- h-sample y-coordinates
- image path

The lane annotations are converted into binary segmentation masks where lane pixels are marked as foreground.

Expected dataset format:

```text
train_subset_3626/
├── label_data_subset_3626.json
└── clips/
    ├── 0313-1/
    ├── 0313-2/
    ├── 0531/
    └── 0601/
```

Update dataset paths in:

```python
src/paths.py
```

Example:

```python
DATA_ROOT = PROJECT_ROOT / "train_subset_3626"
LABEL_JSON = DATA_ROOT / "label_data_subset_3626.json"
```

## Installation

Create and activate a virtual environment.

### Windows

```bash
python -m venv lanedet_app
lanedet_app\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv lanedet_app
source lanedet_app/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Verify Dataset

```bash
python scripts/verify_dataset.py
```

Expected output:

```text
Annotations: 3626
Missing images: 0
```

## Visualize Masks

```bash
python scripts/visualize_masks.py --index 0 --n 4
```

Generated images are saved in:

```text
outputs/mask_vis/
```

## Training

Train U-Net:

```bash
python scripts/train_unet.py --epochs 30 --batch-size 8
```

Train CLRNet-style model:

```bash
python scripts/train_clrnet.py --epochs 30 --batch-size 8
```

Train LaneATT-style model:

```bash
python scripts/train_laneatt.py --epochs 30 --batch-size 8
```

Train SCNN-style model:

```bash
python scripts/train_scnn.py --epochs 30 --batch-size 8
```

## Evaluation

```bash
python scripts/eval_unet.py
python scripts/eval_clrnet.py
python scripts/eval_laneatt.py
python scripts/eval_scnn.py
```

Evaluation reports are saved in:

```text
outputs/
```

Prediction visualizations are saved in:

```text
outputs/unet_preds/
outputs/clrnet_preds/
outputs/laneatt_preds/
outputs/scnn_preds/
```

## Run Gradio Demo

```bash
python app/gradio_app.py
```

The app opens locally in the browser.

The demo works best with:

- front-facing road images
- clear lane markings
- reasonable image resolution
- dashcam-style images similar to TuSimple

Very small images or unusual side-angle images may produce weak predictions.

## Notes

The LaneATT-style, CLRNet-style, and SCNN-style models in this project are lightweight segmentation-inspired versions, not exact official paper implementations. They are implemented for practical benchmarking within a unified binary segmentation pipeline.

The dataset is not included in this repository. Download TuSimple separately and place it in the expected folder format.

If checkpoint files are too large for GitHub, upload them to Google Drive, Hugging Face, or GitHub Releases and link them here.

## Resume Bullet

Built a real-time lane detection benchmarking system on 3,626 TuSimple images, comparing U-Net, CLRNet-style, LaneATT-style, and SCNN-style segmentation models; achieved 0.3889 IoU, 0.5590 F1-score, and 91.5 FPS inference with a reproducible Gradio demo.
