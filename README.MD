# AWS Pneumonia Detection

Deep learning project for binary classification of chest X-ray images using **TensorFlow, Keras and EfficientNetB0**.

The project provides configuration-driven training and command-line inference, with support for local CPU/GPU environments and AWS-oriented configuration.

> **Important:** This project is an educational machine-learning project and is **not a medical diagnostic system**.

---

## Overview

The model classifies chest X-ray images into two classes:

* `NORMAL`
* `PNEUMONIA`

The current workflow is:

```text
Chest X-Ray Dataset
        │
        ▼
Dataset Loader
        │
        ▼
TensorFlow / Keras
        │
        ▼
EfficientNetB0
Transfer Learning
        │
        ▼
Binary Classification
        │
        ▼
pneumonia_model.keras
        │
        ▼
CLI Inference
```

---

## Technology Stack

| Component      | Technology                         |
| -------------- | ---------------------------------- |
| Language       | Python                             |
| Deep Learning  | TensorFlow / Keras                 |
| Model          | EfficientNetB0                     |
| Training       | Transfer Learning                  |
| Classification | Binary Classification              |
| Inference      | TensorFlow / Keras                 |
| Configuration  | JSON + `CONFIG_PATH`               |
| GPU            | NVIDIA CUDA-compatible environment |
| Cloud          | AWS-oriented configuration / S3    |

---

## Repository Structure

```text
aws-pneumonia-detection/
│
├── config/
│   ├── config_local.json
│   ├── config_gpu.json
│   └── config_aws.json
│
├── inference/
│   └── predict.py
│
├── screenshots/
│
├── training/
│   ├── start_training.py
│   ├── evaluate.py
│   └── upload_model.py
│
├── utils/
│   ├── config_loader.py
│   └── dataset_loader.py
│
├── gpu_test.py
├── requirements.txt
├── requirments.txt
└── README.MD
```

The repository also contains a `touch` file which appears to be an unused repository artifact.

---

## Configuration

The project uses the `CONFIG_PATH` environment variable to select the configuration file.

Example:

```bash
export CONFIG_PATH=config/config_local.json
```

Available configurations are:

```text
config/config_local.json
config/config_gpu.json
config/config_aws.json
```

The configuration controls items such as:

* dataset paths
* image size
* training parameters
* model path
* hardware settings
* AWS settings
* inference threshold

---

## Example Configuration

The local configuration follows this structure:

```json
{
  "environment": "local",
  "dataset": {
    "path": "data/raw/chest_xray",
    "train": "train",
    "validation": "val",
    "test": "test"
  },
  "model": {
    "path": "models/pneumonia_model.keras",
    "image_size": [
      224,
      224
    ]
  },
  "training": {
    "epochs": 10,
    "batch_size": 2,
    "learning_rate": 0.0001,
    "transfer_learning": true
  },
  "hardware": {
    "device": "gpu",
    "mixed_precision": false
  },
  "aws": {
    "enabled": false,
    "region": "eu-central-1"
  },
  "inference": {
    "confidence_threshold": 0.5
  }
}
```

The exact configuration files in the repository should be treated as the source of truth for the active settings.

---

## Dataset

The project is designed for the **Kaggle Chest X-Ray Pneumonia Dataset**.

The expected dataset layout is:

```text
data/
└── raw/
    └── chest_xray/
        ├── train/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        │
        ├── val/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        │
        └── test/
            ├── NORMAL/
            └── PNEUMONIA/
```

The dataset is not stored in the Git repository.

---

## Model

The training script creates an **EfficientNetB0** base model with ImageNet weights:

```python
tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet"
)
```

The base model is frozen and extended with:

```text
EfficientNetB0
     │
     ▼
GlobalAveragePooling2D
     │
     ▼
Dense(1, sigmoid)
```

This produces a binary classification model for `NORMAL` and `PNEUMONIA`.

---

## Training

Training is implemented in:

```text
training/start_training.py
```

The script reads its configuration through `CONFIG_PATH`, loads the datasets, creates the EfficientNetB0 model and saves the resulting Keras model to the configured model path.

### Training parameters

The repository's local configuration documents:

```text
Image size:       224 × 224
Epochs:           10
Batch size:       2
Learning ra
```

