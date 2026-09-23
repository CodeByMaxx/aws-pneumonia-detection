# AWS Pneumonia Detection

Deep learning project for binary chest X-ray classification using **TensorFlow/Keras** and **EfficientNetB0**.

The project is designed as a configuration-driven machine learning pipeline with local CPU/GPU training, command-line inference, and optional AWS S3 model storage.

> **Medical disclaimer:** This project is for educational and technical purposes only. It is not a medical device and must not be used for clinical diagnosis or treatment decisions.

## Overview

The model classifies chest X-ray images into two classes:

* `NORMAL`
* `PNEUMONIA`

The current implementation uses **EfficientNetB0 with ImageNet weights** as a frozen feature extractor and adds a binary classification head.

The project currently provides:

* configuration-driven training
* EfficientNetB0 transfer learning
* local CPU/GPU support
* command-line inference
* AWS S3 model upload
* a separate evaluation script under development

There is currently **no FastAPI application or web dashboard** in this repository.

---

## Tech Stack

| Component         | Technology                    |
| ----------------- | ----------------------------- |
| Language          | Python                        |
| Deep Learning     | TensorFlow / Keras            |
| Model             | EfficientNetB0                |
| Learning Approach | Transfer Learning             |
| Dataset           | Chest X-Ray Pneumonia Dataset |
| Configuration     | JSON                          |
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
└── README.MD
```

The dataset and trained model are runtime artifacts and are not included in the repository.

Expected runtime directories:

```text
data/
└── raw/
    └── chest_xray/

models/
└── pneumonia_model.keras
```

---

## Dataset

The training pipeline expects a chest X-ray dataset with the following structure:

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

The dataset itself is **not included** in this repository.

---

## Configuration

The project uses JSON configuration instead of hardcoding training parameters directly into the Python scripts.

The local configuration is stored in:

```text
config/config_local.json
```

The configuration contains settings for:

* dataset paths
* model output path
* training parameters
* hardware
* AWS
* inference threshold

Example:

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

### Current training configuration

| Parameter            |       Value |
| -------------------- | ----------: |
| Image size           | `160 × 160` |
| Epochs               |        `10` |
| Batch size           |         `2` |
| Learning rate        |    `0.0001` |
| Transfer learning    |     Enabled |
| Fine-tuning          |    Disabled |
| Confidence threshold |       `0.5` |

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

Start training with the local configuration:

```bash
CONFIG_PATH=config/config_local.json \
python training/start_training.py
```

The training pipeline:

1. loads the configuration
2. loads the configured dataset
3. creates an EfficientNetB0-based model
4. uses ImageNet pretrained weights
5. freezes the pretrained base model
6. trains the classification head
7. saves the resulting model

The configured output path is:

```text
models/pneumonia_model.keras
```

---

## Model Architecture

The current model consists of an EfficientNetB0 backbone followed by a binary classification head:

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

The current training configuration uses:

* EfficientNetB0
* ImageNet pretrained weights
* frozen base model
* GlobalAveragePooling2D
* one sigmoid output neuron
* Adam optimizer
* binary cross-entropy loss
* accuracy metric

---

## Inference

Inference is available through the command line.

Example:

```bash
CONFIG_PATH=config/config_local.json \
python inference/predict.py path/to/xray.jpg
```

The inference script:

1. loads the configured model
2. loads the supplied image
3. resizes the image
4. runs the model
5. applies the configured confidence threshold
6. prints the predicted class and confidence

Example output:

```text
Prediction: PNEUMONIA
Confidence: 95.4%
```

The current threshold is:

```text
0.5
```

---

## AWS S3

The repository contains an S3 upload script:

```text
training/upload_model.py
```

The script uses `boto3` to upload the configured model to an S3 bucket.

Configure the AWS section in `config/config_local.json`:

```json
"aws": {
  "enabled": false,
  "region": "eu-central-1",
  "bucket": "",
  "s3_key": ""
}
```

Then run:

```bash
CONFIG_PATH=config/config_local.json \
python training/upload_model.py
```

The configured model:

```text
models/pneumonia_model.keras
```

is uploaded to:

```text
s3://<bucket>/<s3_key>
```

AWS credentials should **never** be committed to the repository.

Use standard AWS credential mechanisms such as:

* AWS CLI credentials
* environment variables
* IAM roles
* EC2 instance profiles

---

## GPU Support

The repository includes:

```text
gpu_test.py
```

which can be used to check the TensorFlow GPU environment.

Run:

```bash
python gpu_test.py
```

GPU availability depends on the installed TensorFlow version and the local CUDA/GPU environment.

The configuration also contains:

```json
"hardware": {
  "device": "gpu",
  "mixed_precision": false
}
```

---

## Evaluation

An evaluation script is available at:

```text
training/evaluate.py
```

The intended purpose is to evaluate the trained model against the test dataset and provide classification metrics.

The evaluation pipeline is currently **work in progress** and should not be considered a completed benchmark.

Planned metrics include:

* accuracy
* precision
* recall
* F1 score
* confusion matrix

No production performance claims are made by this repository.

---

## Screenshots

Example project screenshots are available in:

```text
screenshots/
```

---

## Security

Do not commit:

* AWS access keys
* AWS secret keys
* passwords
* tokens
* private credentials
* locally generated datasets
* trained model artifacts unless intentionally versioned

For AWS workloads, prefer IAM roles and other standard AWS credential mechanisms.

---

## Current Status

### Implemented

* TensorFlow/Keras training
* EfficientNetB0 transfer learning
* JSON-based configuration
* local CPU/GPU workflow
* command-line inference
* `.keras` model output
* AWS S3 model upload
* GPU environment test

### Work in Progress

* complete automated evaluation
* additional model metrics
* fine-tuning
* dataset validation
* experiment tracking
* model versioning
* automated testing
* cloud training infrastructure
* production deployment

The repository intentionally documents only functionality that is currently present in the codebase.

---

## Future Improvements

Potential extensions include:

* complete the evaluation pipeline
* add confusion-matrix visualization
* add precision/recall/F1 reporting
* implement EfficientNet fine-tuning
* add automated dataset validation
* add experiment tracking
* add model versioning
* add Docker support
* add AWS training infrastructure
* add CI/CD
* add automated tests

---

## Author

**Markus**

Machine Learning / Cloud Engineering Project

## Disclaimer

This project is intended for educational and engineering purposes. Chest X-ray classification results produced by this software must not be interpreted as medical diagnosis or used as a substitute for professional medical evaluation.

