# Pneumonia Detection from Chest X-Rays using DenseNet121

This repository contains a deep learning pipeline for the binary classification of Chest X-Ray images to detect **Pneumonia** vs. **Normal** cases. The project leverages transfer learning with a pre-trained DenseNet121 architecture, achieving high sensitivity for medical diagnosis.

## 📌 Project Overview
The goal is to provide an automated tool for assisting in the identification of pneumonia from medical imaging. The workflow includes:
* **Data Exploration:** Analyzing class distribution and visualizing samples.
* **Data Preparation:** Implementing heavy augmentation and ImageNet-standard normalization.
* **Model Architecture:** Fine-tuning a pre-trained **DenseNet121**.
* **Optimization:** Using BCEWithLogitsLoss, Adam optimizer, and Early Stopping.
* **Evaluation:** Comprehensive testing using Confusion Matrices and Classification Reports.

## 📊 Dataset Details
The model uses the **Chest X-Ray and OCT Medical Image Dataset**.



[Image of chest x-ray pneumonia vs normal]


### Class Distribution (After Split)
| Set | Normal | Pneumonia | Total |
| :--- | :--- | :--- | :--- |
| **Train** | 1,079 | 3,106 | 4,185 |
| **Validation** | 270 | 777 | 1,047 |
| **Test** | 234 | 390 | 624 |

## 🛠️ Technical Stack
* **Framework:** PyTorch
* **Model:** DenseNet121 (Pre-trained on ImageNet)
* **Optimization:** Adam (Learning Rate: 1e-4)
* **Scheduler:** ReduceLROnPlateau
* **Hardware:** CUDA-enabled GPU support

## 🚀 Model Architecture & Training
We employ **Transfer Learning** by freezing the initial layers and fine-tuning the last two dense blocks (`denseblock3` and `denseblock4`). The classifier was replaced with:
1.  **Dropout (0.3)** to prevent overfitting.
2.  **Linear Layer** outputting a single logit for binary classification.

### Training Strategy
* **Augmentation:** Random Resized Crops, Horizontal Flips, Rotations (20°), and Color Jitter.
* **Early Stopping:** Set with a patience of 5 epochs based on Validation Loss.
* **Validation:** The model reached its peak performance at **Epoch 7**, with training halting at Epoch 12 due to early stopping.

## 📈 Performance Results
The model demonstrates excellent performance, particularly in **Recall for Pneumonia**, which is critical in a medical context to ensure no cases are missed.

### Test Metrics
* **Accuracy:** 92.63%
* **Pneumonia Recall:** 1.00 (100% of Pneumonia cases detected)
* **Normal Precision:** 0.99

### Confusion Matrix
```text
[[189  45]  <- Normal
 [  1 389]]  <- Pneumonia
