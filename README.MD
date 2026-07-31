# AWS Pneumonia Detection Model

End-to-end Deep Learning project for detecting pneumonia from chest X-Ray images using **TensorFlow**, **EfficientNetB0**, and **AWS infrastructure**.

The project demonstrates a complete machine learning workflow:

- Data preparation
- GPU accelerated model training
- Model evaluation
- Local inference
- Configuration-driven training
- AWS S3 model storage
- Optional AWS EC2 GPU training
- Planned FastAPI inference service
- Planned Docker deployment

---

# Architecture Overview

```text
                Chest X-Ray Dataset

                        |
                        ↓

              Data Preprocessing

                        |
                        ↓

          TensorFlow EfficientNetB0

                        |
                        ↓

              Model Evaluation

                        |
                        ↓

              pneumonia_model.keras

                        |
                        ↓

                 Amazon S3

                        |
                        ↓

              Inference Service
```

---

# Project Structure

```text
aws-pneumonia-detection/

├── config/
│   ├── config_local.json
│   └── config_aws.json
│
├── data/
│   └── raw/
│       └── chest_xray/
│           ├── train/
│           ├── test/
│           └── val/
│
├── models/
│   └── pneumonia_model.keras
│
├── utils/
│   └── config_loader.py
│
├── training/
│   ├── preprocess.py
│   ├── start_training.py
│   ├── evaluate.py
│   └── upload_model.py
│
├── inference/
│   ├── predict.py
│   └── download_model.py
│
├── requirements.txt
└── README.md
```

---

# Technology Stack

## Machine Learning

- Python 3.12
- TensorFlow 2.21
- EfficientNetB0
- Transfer Learning
- Binary Image Classification

## Infrastructure

- AWS S3
- AWS EC2 GPU Instances
- AWS CLI
- boto3

## Development

- Virtual Environment
- JSON Configuration
- Git
- Docker (planned)
- FastAPI (planned)

---

# Local Setup

## 1. Clone Repository

```bash
git clone https://github.com/CodeByMaxx/aws-pneumonia-detection.git

cd aws-pneumonia-detection
```

---

# 2. Create Python Environment

Create virtual environment:

```bash
python3 -m venv .venv
```

Activate:

Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

---

# 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:

- TensorFlow
- NumPy
- Pandas
- Scikit-Learn
- OpenCV
- boto3

---

# Configuration

Training parameters are stored in JSON files.

Example:

```json
{
    "image_size": [224,224],
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 0.0001,
    "model_path": "models/pneumonia_model.keras"
}
```

This allows different environments:

```text
config/

├── config_local.json

└── config_aws.json
```

without changing Python code.

---

# Dataset Preparation

This project uses the Kaggle Chest X-Ray Pneumonia dataset.

Download:

```bash
kaggle datasets download \
-d paultimothymooney/chest-xray-pneumonia
```

Extract:

```bash
unzip chest-xray-pneumonia.zip -d data/raw
```

Expected structure:

```text
data/

└── raw/

    └── chest_xray/

        ├── train/

        │   ├── NORMAL/

        │   └── PNEUMONIA/

        ├── test/

        └── val/
```

---

# GPU Environment

The project supports NVIDIA GPU acceleration.

Test TensorFlow GPU detection:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Example:

```text
[
 PhysicalDevice(
 name='/physical_device:GPU:0',
 device_type='GPU'
 )
]
```

Tested environment:

```text
GPU:
NVIDIA GeForce GTX 1660 Ti

TensorFlow:
2.21.0

CUDA:
12.5

cuDNN:
9
```

---

# Training Pipeline

## Preprocessing

Run:

```bash
python -m training.preprocess
```

Example:

```text
Found 5216 files belonging to 2 classes.
```

---

## Start Training

Training uses the JSON configuration:

```bash
python -m training.start_training \
--config config/config_local.json
```

Pipeline:

```text
Dataset

 ↓

Preprocessing

 ↓

EfficientNetB0 Transfer Learning

 ↓

Training

 ↓

Model Export

 ↓

pneumonia_model.keras
```

Output:

```text
models/

└── pneumonia_model.keras
```

---

# Model Evaluation

Run:

```bash
python -m training.evaluate \
--config config/config_local.json
```

Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

For medical classification, recall is especially important because false negatives should be minimized.

---

# Local Inference

Run prediction:

```bash
python -m inference.predict \
--image data/raw/chest_xray/test/PNEUMONIA/example.jpeg \
--config config/config_local.json
```

Example:

```text
Prediction:

PNEUMONIA

Confidence:

95.40%
```

---

# AWS Model Storage

The project uses Amazon S3 as model storage.

Workflow:

```text
Training

    ↓

pneumonia_model.keras

    ↓

Amazon S3

    ↓

Download Model

    ↓

Inference
```

---

# AWS Setup

## Create IAM User

Do not use the AWS root account.

Create:

```text
pneumonia-detection-user
```

Required permissions:

```json
{
    "Effect": "Allow",
    "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
    ],
    "Resource": "*"
}
```

For production environments restrict access to the required bucket.

---

# AWS CLI Configuration

Install:

```bash
aws --version
```

Configure:

```bash
aws configure
```

Example:

```text
Region:
eu-central-1

Output:
json
```

Verify:

```bash
aws sts get-caller-identity
```

---

# S3 Bucket Structure

Example:

```text
s3://codebymaxx-pneumonia-models/

└── models/

    └── pneumonia_model.keras
```

---

# Upload Model

After training:

```bash
python -m training.upload_model \
--config config/config_aws.json
```

Uploads:

```text
models/pneumonia_model.keras
```

to:

```text
Amazon S3
```

---

# Download Model

For inference:

```bash
python -m inference.download_model \
--config config/config_aws.json
```

---

# Security

Never commit:

```text
.aws/
.env
credentials
```

Never upload:

- AWS Access Keys
- AWS Secret Keys
- Private credentials

Production recommendations:

- IAM Roles
- EC2 Instance Profiles
- AWS Secrets Manager

---

# Optional AWS EC2 GPU Training

For larger workloads the project can run on AWS GPU instances.

Recommended:

```text
g4dn.xlarge

g5.xlarge
```

Workflow:

```text
AWS EC2 GPU Instance

        ↓

CUDA + TensorFlow Setup

        ↓

Dataset Download

        ↓

Training

        ↓

Upload Model to S3
```

---

# Future Deployment Roadmap

## FastAPI Prediction Service

Planned architecture:

```text
Client

 ↓

FastAPI

 ↓

TensorFlow Model

 ↓

Prediction JSON
```

Example response:

```json
{
    "prediction": "PNEUMONIA",
    "confidence": 0.954
}
```

---

## Docker Deployment

Planned:

```text
Docker Container

        ↓

FastAPI

        ↓

TensorFlow Model

        ↓

Prediction API
```

---

# Complete ML Workflow

```text
Developer Machine

        ↓

Dataset Preparation

        ↓

Configuration Driven Training

        ↓

Model Evaluation

        ↓

Model Version

        ↓

Upload to AWS S3

        ↓

Inference Service

        ↓

Prediction
```

---

# Project Goals

This project demonstrates:

- Deep Learning model development
- GPU accelerated training
- Cloud based model storage
- ML pipeline design
- Production oriented architecture
- Deployment preparation
