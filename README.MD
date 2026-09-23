# AWS Pneumonia Detection System

Deep learning project for **binary chest X-ray classification** using TensorFlow/Keras and EfficientNetB0.

The project currently focuses on:

* Configuration-driven model training
* EfficientNetB0 transfer learning
* Local GPU support
* Command-line inference
* AWS S3 model upload
* A structure that can be extended toward cloud-based training and deployment

> **Medical disclaimer:** This project is for educational and technical purposes only. It is not a medical device and must not be used for clinical diagnosis or treatment decisions.

---

## Overview

The model classifies chest X-ray images into two categories:

* `NORMAL`
* `PNEUMONIA`

The current implementation uses **EfficientNetB0 with ImageNet weights** as the feature extractor and adds a binary classification head.

The training and inference configuration is loaded from JSON files instead of being hardcoded directly into the Python scripts.

---

## Tech Stack

| Component         | Technology                    |
| ----------------- | ----------------------------- |
| Language          | Python                        |
| Deep Learning     | TensorFlow / Keras            |
| Model             | EfficientNetB0                |
| Learning Approach | Transfer Learning             |
| Dataset           | Chest X-Ray Pneumonia Dataset |
| Configuration     | JSON + `CONFIG_PATH`          |
| Cloud Storage     | AWS S3                        |
| Inference         | Python CLI                    |
| Hardware          | CPU / GPU                     |

---

## Project Structure

```text
aws-pneumonia-detection/
│
├── config/
│   └── config_local.json
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
├── README.MD
│
├── requirments.txt
└── touch
```

`data/` and `models/` are runtime directories and are not part of the current repository tree.

---

## Dataset

The training configuration expects the following directory structure:

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

The dataset itself is not included in this repository.

---

## Configuration

The active configuration is selected through the `CONFIG_PATH` environment variable.

Example:

```bash
export CONFIG_PATH=config/config_local.json
```

The current local configuration contains:

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
    "path": "models/pneumonia_model.keras"
  },

  "training": {
    "epochs": 10,
    "batch_size": 2,
    "image_size": [160, 160],
    "learning_rate": 0.0001,
    "transfer_learning": true,
    "fine_tuning": false
  },

  "hardware": {
    "device": "gpu",
    "mixed_precision": false
  },

  "aws": {
    "enabled": false,
    "region": "eu-central-1",
    "bucket": "",
    "s3_key": ""
  },

  "inference": {
    "confidence_threshold": 0.5
  }
}
```

The important training parameters are currently:

* Image size: `160 × 160`
* Epochs: `10`
* Batch size: `2`
* Learning rate: `0.0001`
* Transfer learning: enabled
* Fine-tuning: disabled
* Inference threshold: `0.5`

---

## Model Architecture

The training script uses EfficientNetB0 with ImageNet weights.

The base model is frozen during the current training configuration.

The classification pipeline is approximately:

```text
Chest X-Ray
     │
     ▼
Image Preprocessing
     │
     ▼
EfficientNetB0
     │
     ▼
GlobalAveragePooling2D
     │
     ▼
Dense(1, sigmoid)
     │
     ▼
NORMAL / PNEUMONIA
```

The model is trained using:

* Adam optimizer
* Binary cross-entropy loss
* Accuracy metric

The resulting model is saved as:

```text
models/pneumonia_model.keras
```

---

## Installation

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

With the local configuration:

```bash
CONFIG_PATH=config/config_local.json \
python training/start_training.py
```

The training script:

1. Loads the configuration
2. Loads the train/validation/test dataset configuration
3. Creates an EfficientNetB0-based model
4. Freezes the pretrained base model
5. Trains the classification head
6. Saves the trained model to the configured model path

The current training script uses the training and validation datasets during `fit()`.

---

## Inference

Inference is performed through the command line.

Example:

```bash
CONFIG_PATH=config/config_local.json \
python inference/predict.py path/to/xray.jpg
```

The script:

1. Loads the configured model
2. Loads the supplied image
3. Resizes it according to the configuration
4. Runs model inference
5. Applies the configured confidence threshold
6. Prints the predicted class and confidence

Example output:

```text
Prediction: PNEUMONIA
Confidence: 95.4%
```

The threshold is currently configured as:

```text
0.5
```

---

## AWS S3 Integration

The repository includes a dedicated S3 upload script:

```text
training/upload_model.py
```

It uses `boto3` to upload the configured model file to an S3 bucket.

The relevant configuration is:

```json
"aws": {
  "enabled": false,
  "region": "eu-central-1",
  "bucket": "",
  "s3_key": ""
}
```

After configuring the bucket and S3 object key, the upload can be executed with:

```bash
CONFIG_PATH=config/config_local.json \
python training/upload_model.py
```

The script uploads:

```text
models/pneumonia_model.keras
```

to:

```text
s3://<bucket>/<s3_key>
```

AWS credentials should be provided through the normal AWS credential mechanisms or an IAM role rather than committed to the repository.

---

## GPU Support

The project contains:

```text
gpu_test.py
```

for checking the available TensorFlow GPU environment.

The configuration also contains:

```json
"hardware": {
  "device": "gpu",
  "mixed_precision": false
}
```

GPU availability depends on the local TensorFlow/CUDA environment.

---

## Evaluation

An evaluation script is present:

```text
training/evaluate.py
```

It is intended to calculate classification metrics such as:

* Precision
* Recall
* F1 score
* Confusion matrix

However, the current implementation is **not complete**. It references `test_dataset` and `test_labels` without defining/loading them first.

Therefore, evaluation should currently be considered **work in progress** rather than a completed pipeline.

---

## Security

No AWS credentials, access keys or other secrets should be committed to the repository.

For AWS deployments, prefer:

* IAM roles
* EC2 instance profiles
* Environment-based credentials
* AWS Secrets Manager where appropriate

Never hardcode AWS access keys into configuration files or Python source code.

---

## Current Status

### Implemented

* TensorFlow/Keras training
* EfficientNetB0 transfer learning
* Configuration-based training
* Local GPU configuration
* CLI inference
* `.keras` model output
* AWS S3 model upload
* GPU test utility

### Work in Progress

* Automated model evaluation
* Fine-tuning
* More extensive metrics
* Cloud training workflow
* Production deployment

The repository currently does **not** contain the FastAPI application described by the older README, so the documentation intentionally does not claim a web API or browser-based dashboard.

---

## Repository Cleanup

A few files in the current repository could be cleaned up:

* `requirments.txt` — duplicate/misspelled dependency file
* `touch` — appears to be an unused artifact
* `README.MD` — could optionally be renamed to the conventional `README.md`

These do not affect the core training/inference workflow.

---

## Future Improvements

Possible next steps:

* Complete the evaluation pipeline
* Add precision/recall/F1 reporting
* Add confusion-matrix visualization
* Implement fine-tuning of EfficientNetB0
* Add automated dataset validation
* Add experiment tracking
* Add model versioning
* Add Docker support
* Add AWS training/deployment infrastructure
* Add CI/CD
* Add automated testing

---

## Author

**Markus**

Machine Learning / Cloud Engineering Project

