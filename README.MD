# AWS Pneumonia Detection System

Deep learning application for detecting pneumonia from chest X-ray images using **TensorFlow, Keras, EfficientNetB0, FastAPI and AWS S3**.

The project demonstrates an end-to-end machine learning workflow covering data preparation, transfer learning, model evaluation, local/GPU training, inference and cloud-based model storage.

> **Note:** This project is intended as a machine-learning engineering demonstration and is **not a medical diagnostic system**.

---

## Overview

The system performs binary image classification of chest X-ray images:

* `NORMAL`
* `PNEUMONIA`

The main components are:

* TensorFlow / Keras
* EfficientNetB0 transfer learning
* FastAPI
* Configuration-driven execution
* Local and GPU training
* Model evaluation
* Image inference
* AWS S3 model storage
* Optional AWS GPU training

---

## Architecture

```text
Chest X-Ray Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
TensorFlow / Keras
        │
        ▼
EfficientNetB0
Transfer Learning
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
pneumonia_model.keras
        │
        ├───────────────┐
        ▼               ▼
   FastAPI API      AWS S3 Storage
        │
        ▼
    Inference
```

The application is configuration-driven so that the same codebase can be used for local development, GPU training and AWS-based execution.

---

## Technology Stack

| Area          | Technology                           |
| ------------- | ------------------------------------ |
| Language      | Python                               |
| Deep Learning | TensorFlow / Keras                   |
| Model         | EfficientNetB0                       |
| ML approach   | Transfer Learning                    |
| API           | FastAPI                              |
| Server        | Uvicorn                              |
| Cloud Storage | AWS S3                               |
| Configuration | JSON + `CONFIG_PATH`                 |
| Hardware      | CPU / NVIDIA GPU                     |
| Dataset       | Kaggle Chest X-Ray Pneumonia Dataset |

---

## Project Structure

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
│   └── config_loader.py
│
├── gpu_test.py
├── requirements.txt
└── README.md
```

The training and inference workflows use configuration files to define paths, model settings, hardware and AWS behaviour.

---

## Configuration

The application uses the `CONFIG_PATH` environment variable to select the active configuration.

Example:

```bash
CONFIG_PATH=config/config_local.json
```

### Local configuration

```bash
CONFIG_PATH=config/config_local.json
```

### GPU configuration

```bash
CONFIG_PATH=config/config_gpu.json
```

### AWS configuration

```bash
CONFIG_PATH=config/config_aws.json
```

This approach avoids hard-coding environment-specific settings into the application code.

---

## Dataset

The project is designed around the **Kaggle Chest X-Ray Pneumonia Dataset**.

The expected dataset structure is:

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

The dataset itself is not included in the repository.

---

## Machine Learning Pipeline

### 1. Image preprocessing

The preprocessing pipeline prepares the X-ray images for model training by:

* loading the images
* resizing them
* normalizing image data
* creating TensorFlow datasets

### 2. Transfer learning

The model uses **EfficientNetB0** as the foundation for image classification.

Transfer learning allows the project to reuse features learned from a large image dataset and adapt them to the pneumonia classification task.

### 3. Training

The documented default training configuration uses:

```text
Image size:       224 × 224
Epochs:           10
Optimizer:        Adam
Learning rate:    0.0001
Transfer learning: enabled
```

The resulting model is stored as:

```text
pneumonia_model.keras
```

---

## Model Evaluation

The project considers multiple classification metrics instead of relying only on accuracy.

### Accuracy

Percentage of predictions classified correctly.

### Precision

Measures how many predicted positive cases are actually positive.

### Recall

Measures how many actual pneumonia cases are detected.

For medical classification tasks, recall is particularly important because false negatives can be significant.

### F1 Score

Combines precision and recall into a single metric.

The evaluation workflow is implemented in:

```text
training/evaluate.py
```

---

## FastAPI Application

The project provides a FastAPI-based web application for interacting with the trained model.

Start the application with:

```bash
CONFIG_PATH=config/config_local.json \
uvicorn api.main:app --reload
```

The application is then available at:

```text
http://127.0.0.1:8000
```

### Main application areas

The current project documentation describes the following routes:

| Route         | Purpose                              |
| ------------- | ------------------------------------ |
| `/`           | Application dashboard                |
| `/predict`    | Upload an X-ray and run inference    |
| `/train`      | Start model training                 |
| `/models`     | Model information                    |
| `/evaluation` | Evaluation results                   |
| `/settings`   | Active configuration and environment |

---

## Prediction

The prediction workflow accepts a chest X-ray image and passes it through the trained EfficientNetB0 model.

```text
X-Ray Image
     │
     ▼
Preprocessing
     │
     ▼
EfficientNetB0
     │
     ▼
Classification
     │
     ├── NORMAL
     └── PNEUMONIA
```

The prediction interface can expose both the predicted class and the model confidence.

---

## Training Interface

Training can be started through the FastAPI application as documented by the project.

The training workflow supports:

* starting training
* background execution
* training status
* training logs

The underlying training entry point is:

```text
training/start_training.py
```

---

## AWS Integration

The project supports storing trained models in **Amazon S3**.

The intended workflow is:

```text
Training
    │
    ▼
pneumonia_model.keras
    │
    ▼
Amazon S3
    │
    ▼
Model Download
    │
    ▼
Inference
```

AWS-specific configuration is selected through:

```bash
CONFIG_PATH=config/config_aws.json
```

The project is therefore structured so that model training and model storage can be separated from local development.

---

## Security

AWS credentials and secrets should **never be committed to the repository**.

Do not store:

```text
AWS access keys
AWS secret keys
.env files
passwords
API keys
other credentials
```

For production environments, use AWS-native identity and secret-management mechanisms such as:

* IAM roles
* EC2 instance profiles
* AWS Secrets Manager

---

## Installation

Create a Python virtual environment:

```bash
python3 -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

Start the FastAPI application locally:

```bash
CONFIG_PATH=config/config_local.json \
uvicorn api.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

---

## Run Training

### Local

```bash
CONFIG_PATH=config/config_local.json \
python training/start_training.py
```

### GPU

```bash
CONFIG_PATH=config/config_gpu.json \
python training/start_training.py
```

### AWS

```bash
CONFIG_PATH=config/config_aws.json \
python training/start_training.py
```

The GPU configuration can be used when a compatible NVIDIA CUDA environment is available.

---

## GPU Check

The repository also contains:

```text
gpu_test.py
```

This can be used to verify the local GPU environment before starting a GPU-based training run.

---

## Model Upload

The repository contains a dedicated training utility:

```text
training/upload_model.py
```

This is used as part of the AWS model-storage workflow.

---

## Screenshots

The repository contains application screenshots under:

```text
screenshots/
```

These can be used to document the FastAPI dashboard, prediction interface and other application views.

---

## Medical Disclaimer

This project is an educational and engineering demonstration.

It must **not** be used as a substitute for:

* medical diagnosis
* clinical decision-making
* professional radiological assessment
* patient treatment decisions

Model predictions can be incorrect and should not be interpreted as medical advice.

---

## Possible Future Improvements

Potential extensions include:

* Docker deployment
* AWS ECS deployment
* CI/CD automation
* model versioning
* authentication
* automated retraining
* AWS CloudWatch monitoring
* model registry integration
* improved experiment tracking
* automated model evaluation

---

## Project Goal

The main goal of this project is to demonstrate how a deep-learning image-classification workflow can be combined with a modern Python API and AWS infrastructure.

It brings together:

```text
Machine Learning
        +
Deep Learning
        +
Python API
        +
GPU Computing
        +
AWS Cloud Infrastructure
```

---

## Author

**Markus**

Machine Learning / Cloud Engineering Project

