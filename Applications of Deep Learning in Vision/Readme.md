# Deep Learning in Vision: Classification, Detection, Segmentation and Vision Transformers

## Overview
This project compares the main tasks of computer vision using **PyTorch**, **torchvision**, **Ultralytics YOLO** and **segmentation-models-pytorch**. Pretrained models are used for image classification (ResNet-50 vs. a Vision Transformer), object detection (YOLO11n) and semantic segmentation (a U-Net with a ResNet-34 encoder). Each experiment checks the underlying metric (IoU, Dice, top-1 accuracy, mAP) by hand and then measures what transfer learning does in practice.

---

## Features
- **IoU and Dice** implemented from scratch, with the Dice/IoU relationship checked numerically
- **Classification** with pretrained ResNet-50 and ViT-B/16 on Imagenette, including a softmax sanity check
- **Fine-tuning** both models with a frozen backbone and with all weights trainable
- **Object detection** with YOLO11n on COCO128, with hand-computed box matching at IoU 0.5
- Effect of the **NMS threshold** and **confidence threshold** on the number of predicted boxes
- **mAP evaluation** and fine-tuning of YOLO with frozen layers
- **Segmentation** with a U-Net (pretrained ResNet-34 encoder) on Oxford-IIIT Pet
- Why **pixel accuracy is misleading** compared with Dice and IoU
- Seven lab exercises on rotation robustness, small-data fine-tuning, loss functions and freezing

---

## Project Structure

```
deep-learning-in-vision/
│── MCO23378_Lab12.ipynb
│── data/                  <- Imagenette, CIFAR-10, Oxford-IIIT Pet (downloaded automatically)
│── README.md
```

---

## Requirements

- Python 3.x
- PyTorch
- Torchvision
- Ultralytics
- segmentation-models-pytorch
- Matplotlib
- NumPy
- OpenCV (installed with Ultralytics)

Install the required packages:

```bash
pip install torch torchvision ultralytics segmentation-models-pytorch matplotlib numpy
```

A GPU is strongly recommended. The notebook falls back to CPU, but fine-tuning the full ViT-B/16 took over an hour on CPU.

---

## How to Run

### Step 1: Open the Notebook

```bash
jupyter notebook MCO23378_Lab12.ipynb
```

### Step 2: Run All Cells in Order

Helper functions (`iou`, `dice`, `iou_from_counts`, `evaluate`, `finetune`, `load_truth`, `PetMasks`) are defined once and reused later, including in the exercises. The first run downloads Imagenette, COCO128, Oxford-IIIT Pet (about 800 MB), the pretrained weights and, for Exercise 2, CIFAR-10.

### Step 3: Review Results

Each experiment prints its metrics. Experiments 3 and 4 also draw prediction-vs-truth figures.

---

## Models Used

| Task | Model | Parameters | Notes |
|---|---|---|---|
| Classification | ResNet-50 (ImageNet V2 weights) | 25,557,032 | 224 × 224 input |
| Classification | ViT-B/16 (ImageNet V1 weights) | 86,567,656 | 224 × 224 input |
| Detection | YOLO11n | 2.6 million | 80 COCO classes |
| Segmentation | U-Net, ResNet-34 encoder | 21,284,672 (encoder) + 3,151,697 (decoder and head) | Binary pet / background mask |

---

## Experiments

| # | Topic | What it demonstrates |
|---|---|---|
| 1 | Metrics | IoU of two boxes = 0.3636; Dice of two masks = 0.7778, which equals 2·IoU / (1 + IoU) for IoU = 0.6364 |
| 2 | Classification | Pretrained ResNet-50 and ViT-B/16 on Imagenette, then fine-tuned with a new 10-way head |
| 3 | Detection | YOLO11n on COCO128: box matching, NMS, mAP and fine-tuning |
| 4 | Segmentation | U-Net on 600 training and 200 test pet photos at 256 × 256, trained with BCE + Dice loss for 3 epochs |

---

## Key Results

### Experiment 2: Classification

Zero-shot, both models score 100% top-1 on 500 Imagenette validation images (ResNet-50 at 287.5 ms/image, ViT-B/16 at 819.0 ms/image on CPU).

| Model | Training | Trainable parameters | Val accuracy | Time (s) |
|---|---|---|---|---|
| ResNet-50 | Frozen | 20,490 | 97.8% | 431 |
| ResNet-50 | Full | 23,528,522 | 96.4% | 1,473 |
| ViT-B/16 | Frozen | 7,690 | 99.6% | 1,498 |
| ViT-B/16 | Full | 85,806,346 | 90.8% | 4,615 |

Training only the new head was both faster and more accurate than updating all weights in this short, small-data setting.

### Experiment 3: Detection (COCO128, YOLO11n)

| Measure | Value |
|---|---|
| Images / ground-truth boxes | 20 / 89 |
| Predicted boxes (confidence ≥ 0.25) | 75 |
| Matched at IoU 0.5 | 61 (81% of predictions) |
| Mean best IoU | 0.706 |
| mAP@0.5 / mAP@0.5:0.95 (full COCO128) | 0.671 / 0.503 |
| mAP@0.5:0.95 after 5-epoch fine-tuning (`freeze=10`) | 0.491 |

NMS threshold sweep: 0.1 gives 64 boxes, 0.5 gives 69, 0.7 gives 75 and 0.95 gives 152, so a very permissive threshold lets duplicate boxes through. Fine-tuning for 5 epochs did not improve mAP because the model was already trained on COCO and COCO128 is a subset of it.

### Experiment 4: Segmentation (U-Net)

| Measure | Value |
|---|---|
| Pet share of all test pixels | 42.3% |
| Pixel accuracy | 0.9346 |
| Pixel accuracy of guessing "all background" | 0.5772 (Dice 0.0000) |
| Dice (pet) | 0.9237 |
| IoU (pet) / IoU (background) | 0.8583 / 0.8916 |
| Mean IoU | 0.8750 |

---

## Lab Exercises

| # | Exercise | Result |
|---|---|---|
| 1 | Rotate images by 90° before classification | ResNet-50 falls from 100.0% to 94.0%; ViT-B/16 falls from 100.0% to 83.4% |
| 2 | Frozen ResNet-50, 1 epoch on CIFAR-10 (2,000 train / 1,000 test images) | 65.1% accuracy |
| 3 | Sweep YOLO confidence from 0.05 to 0.9 | Plot of number of predictions vs. match fraction |
| 4 | U-Net with BCE, Dice and combined loss | Dice / IoU: BCE 0.8577 / 0.7509, Dice 0.8684 / 0.7675, Combined 0.8835 / 0.7913 |
| 5 | Shrink fine-tuning data (Imagenette) | 100 images: frozen 64.4%, full 66.8%. 20 images: frozen 36.8%, full 14.4% |
| 6 | YOLO fine-tune with `freeze=0` and `lr0=1e-2` | Code provided; no output saved in the notebook |
| 7 | U-Net with a frozen encoder | Dice 0.9226, IoU 0.8564 |

With very little data (20 images), training all weights collapses to 14.4%, while training only the head still reaches 36.8%.

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- Ultralytics (YOLO11)
- segmentation-models-pytorch
- Matplotlib

---

## Key Takeaway

Pretrained models transfer remarkably well, and updating only a small new head is often the better choice when data and compute are limited: it was faster, matched or beat full fine-tuning for both classifiers, and kept YOLO and the U-Net strong. Metrics must also match the task. Pixel accuracy hides the failure of a trivial "all background" guess (0.5772 accuracy but Dice 0), while Dice, IoU and mAP expose it. Finally, the ViT proved more sensitive than the CNN to a 90° rotation, a reminder that benchmark accuracy does not guarantee robustness.
