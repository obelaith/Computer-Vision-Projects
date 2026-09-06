# YawDD Yawning Detection

YawDD is a computer vision project for detecting yawning behavior from video frames.

The project builds a complete pipeline starting from raw videos, extracting frames, detecting facial landmarks, generating mouth crops, and training a deep learning classifier to distinguish between yawning and non-yawning frames.

The goal is to focus the model on the visual features related to yawning while maintaining a reproducible preprocessing and evaluation pipeline.

---

# Overview

Yawning detection can be useful in applications such as driver monitoring systems and human behavior analysis.

Instead of directly training on full video frames, this project extracts the mouth region using facial landmarks. This reduces irrelevant visual information and allows the model to focus on the area most related to yawning behavior.

The pipeline consists of:

- Video frame extraction
- Facial landmark detection using MediaPipe FaceMesh
- Mouth Aspect Ratio (MAR) calculation
- Automatic frame labeling
- Mouth crop generation
- CNN-based classification using ResNet18

---

# Pipeline

```text
Raw Videos
     |
     v
Frame Extraction
     |
     v
MediaPipe FaceMesh
     |
     v
Mouth Landmark Detection
     |
     v
MAR Calculation
(Mouth Aspect Ratio)
     |
     v
Frame Labeling
(Yawn / Non-Yawn)
     |
     v
Mouth Crop Generation
     |
     v
ResNet18 Classification
     |
     v
Yawning Prediction
```

---

# Features

- Video-to-frame preprocessing pipeline
- Automatic face landmark extraction
- Mouth-focused image crops
- MAR-based yawning detection
- Subject-aware dataset splitting
- Transfer learning with ResNet18
- Evaluation using multiple classification metrics

---

# Data Processing

## Frame Extraction

Input videos are processed into individual frames.

Frames are sampled at:

```text
10 FPS
```

Metadata is stored to maintain the relationship between:

- original video
- extracted frame
- subject identity
- labels

---

## Facial Landmark Detection

MediaPipe FaceMesh is used to locate facial landmarks.

The mouth region is defined using:

```text
Left lip:   61
Right lip:  291
Top lip:    13
Bottom lip: 14
```

---

## Mouth Aspect Ratio (MAR)

The Mouth Aspect Ratio is calculated as:

```text
MAR = mouth height / mouth width
```

Sustained high MAR values are used to identify potential yawning events.

Temporal smoothing is applied to reduce noise from individual frames.

---

## Crop Generation

Instead of using complete frames, the pipeline extracts mouth regions and uses these crops as the model input.

This helps the classifier focus on the visual features related to yawning.

---

# Model

The final classifier uses:

```text
ResNet18
```

Training setup:

| Parameter | Value |
|---|---|
| Image size | 224 × 224 |
| Model | ResNet18 |
| Framework | PyTorch |
| Optimizer | Adam |
| Task | Binary classification |

---

# Dataset Split

The dataset is split while keeping video/subject separation to reduce data leakage.

Generated metadata includes:

- video information
- extracted frames
- labels
- train/validation/test splits

---

# Results

Evaluation was performed on the held-out test set.

| Metric | Score |
|---|---:|
| Accuracy | 99.02% |
| Macro Precision | 95.18% |
| Macro Recall | 97.20% |
| Macro F1 | 96.16% |

Validation:

```text
Best validation Macro F1:
0.9172
```

The model achieves strong performance while being evaluated using macro metrics to account for class imbalance.

---

# Project Structure

```text
YawDD/
│
├── YawDD_mainn.ipynb
│
├── crops/
│   └── generated mouth region images
│
├── splits/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── training/
│   └── metrics_resnet18.json
│
├── videos_info.csv
├── frames_info.csv
├── frames_labels.csv
├── crops_labeled.csv
│
├── requirements.txt
└── README.md
```

---

# Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Main libraries:

```text
PyTorch
Torchvision
MediaPipe
OpenCV
Pandas
NumPy
Pillow
```

---

# Running the Project

The complete pipeline is implemented in:

```text
YawDD_mainn.ipynb
```

The notebook contains:

1. Dataset indexing
2. Frame extraction
3. Face landmark processing
4. MAR calculation
5. Label generation
6. Mouth crop extraction
7. Model training
8. Evaluation

---

# Limitations

- The current approach operates on extracted frames rather than full temporal video sequences.
- MAR-based labeling provides an automatic labeling method but may introduce some noise.
- Performance depends on face visibility and landmark detection quality.

---

# Future Improvements

- Add temporal models such as LSTM or video transformers
- Compare additional CNN architectures
- Improve augmentation strategies
- Add real-time webcam inference
- Evaluate on additional datasets

---
