# Lung Cancer Histopathological Classification using Deep Learning

This repository contains a comparative study between Convolutional Neural Networks (CNN) and Scattering Networks (ScatNet) for histopathological image classification of lung cancer tissues.

## Project Overview

The project aims to classify lung cancer histopathological images using two different deep learning approaches:

- **CNN**: A custom architecture with minimal parameters (~14K) achieving 99.26% ± 0.72% accuracy
- **ScatNet**: A wavelet-based approach with fixed mathematical representations achieving 92.99% ± 1.59% accuracy

Key features:

- Binary classification between adenocarcinoma and benign tissue samples
- K-fold cross-validation with 10 folds
- Comprehensive model interpretability analysis
- Advanced visualization techniques for model decisions

## System Requirements

- Python 3.11
- 8GB VRAM
- Required packages listed in `requirements.txt`

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the training script:

```bash
python main.py
```

3. For interactive exploration, use Jupyter notebook:

```bash
jupyter notebook main.ipynb
```

For detailed methodology and results, refer to the presentation in `doc/presentation.pdf`.
