# Lung Cancer Histopathological Classification using Deep Learning

This repository contains implementation of deep learning models for histopathological image classification of lung cancer tissues.

## Project Overview

This project focuses on the binary classification of lung cancer histopathological images, distinguishing between adenocarcinoma and benign tissue. The project compares two different approaches:

1. Convolutional Neural Networks (CNN)
2. Scattering Networks (ScatNet)

The goal is to investigate whether traditional CNN architectures or wavelet-based ScatNet approaches are more effective for identifying subtle tissue patterns and cellular structures in histopathological images.

## Dataset

The dataset consists of lung cancer histopathological images across three classes:

- Adenocarcinoma
- Squamous cell carcinoma (not used in the binary classification task)
- Benign tissue

Images are processed with their original size (768×768 pixels).

## Key Findings

- CNN significantly outperforms ScatNet in both accuracy and speed (98.9% vs 87.9% mean accuracy)
- Class average color alone is highly predictive for classification
- Learned features (CNN) outperform fixed mathematical representations (ScatNet)
- Models can achieve high accuracy even with grayscale images, focusing on structural features

## Model Architectures

### CNN Model

- Efficient architecture with only ~14K parameters
- Two convolutional blocks with batch normalization
- Simple classifier head (16-dimensional feature space)
- Mean Accuracy: 98.9%
- Mean F1 Score: 98.9%

### ScatNet Model

- Wavelet-based feature extraction
- J=3 scale parameter, L=8 orientations, M=2 scattering order
- Translation, rotation, and scaling invariant
- Complex classifier (217 → 64 → 2 neurons)
- Mean Accuracy: 87.9%
- Mean F1 Score: 86.7%

## Explainable AI Methods

The project implements several attribution methods to visualize model decisions:

- Vanilla Backpropagation
- Guided Backpropagation
- Integration with Captum library for additional attribution techniques

## System Requirements

- Python 3.11
- 8GB VRAM
- Required packages listed in `requirements.txt`

## Repository Structure

- `src/`: Source code for models and utilities
- `dataset/`: Processed dataset for binary classification
- `dataset_orig/`: Original dataset with all three classes
- `outputs/`: Trained model checkpoints and evaluation results
- `doc/`: Documentation and presentation materials

## Usage

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the training script:

```
python main.py
```

3. For interactive exploration, use Jupyter notebook:

```
jupyter notebook main.ipynb
```

## Results

The experiments were conducted using 10-fold cross-validation:

- CNN achieved 98.9% mean accuracy with a best fold accuracy of 99.6%
- ScatNet achieved 87.9% mean accuracy with a best fold accuracy of 92.6%

For detailed results, refer to the metrics files in the `outputs/` directory.
