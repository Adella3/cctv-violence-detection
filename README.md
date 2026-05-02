# CCTV Violence & Weapon Detection

> **"How can surveillance systems identify threats in real time — before harm occurs?"**

A deep learning pipeline for real-time violence and weapon detection in CCTV footage, using a CNN-LSTM hybrid architecture. Built for the UTS 31256 Image Processing and Pattern Recognition course (Autumn 2025).

---

## Results

| Metric | Score |
|---|---|
| **Accuracy** | **97.48%** |
| F1 Score (weighted) | 0.975 |
| Precision — Violence | 1.000 |
| Recall — Normal | 1.000 |
| False Positives | **0** |

The model achieved zero false positives on the test set — every video flagged as violent was genuinely violent.

---

## The Problem

Public safety monitoring systems generate enormous volumes of CCTV footage that human operators cannot review in real time. The question this project addresses: can a system reliably distinguish threatening from normal behavior — quickly enough and accurately enough to act on?

---

## Dataset

**Smart-City CCTV Violence Detection (SCVD)**

| Split | Normal | Violence | Weaponized | Total |
|---|---|---|---|---|
| Train | 872 | 970 | 832 | **2,674** |
| Test | 169 | 118 | 190 | **477** |

- 3 original classes: Normal, Violence, Weaponized
- Merged into binary (Normal vs Violence) to reflect real-world alerting needs
- Verified: **zero data leakage** between train and test sets

---

## Architecture

```
Video Input
    ↓
Frame Extraction (10 frames, uniform sampling)
    ↓
TimeDistributed MobileNetV2 (pretrained, ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
LSTM (128 units) → LSTM (64 units)
    ↓
Dense (256) → Dense (128) → Dense (64)
    ↓
Binary Output (sigmoid)
```

**Key design decisions:**
- **MobileNetV2** as CNN backbone — pretrained spatial feature extractor, frozen during training
- **LSTM layers** — capture temporal patterns across frames (violence unfolds over time)
- **Class weighting** — handles the imbalance between Normal and Violence samples
- **EarlyStopping + ReduceLROnPlateau** — prevents overfitting, adapts learning rate dynamically

---

## Training

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr = 3e-4) |
| Batch size | 16 |
| Epochs | 10 |
| Loss | Binary Crossentropy |
| Hardware | Apple M2 Max (Metal GPU) |

---

## Engineering Highlights

- **Data leakage check**: explicitly verified no filename overlap between train/test splits
- **Reproducible sampling**: sorted file loading + seed control ensures consistent results
- **K-Fold cross-validation**: robust evaluation across data subsets
- **Error analysis**: identified 12 false negatives (ambiguous violence clips with low model confidence) — surfacing where the model's limits are matters as much as where it succeeds

---

## Tech Stack

`Python` · `TensorFlow 2.16` · `Keras` · `OpenCV` · `MobileNetV2` · `scikit-learn` · `NumPy` · `Matplotlib`

---

## Context

This project was completed as part of UTS Bachelor of IT — Data Analytics (2025). It was a team project of 6, with primary responsibility for model architecture design and evaluation pipeline.
