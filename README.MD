# AWS Pneumonia Detection

An end-to-end deep learning application for classifying chest X-ray images as **NORMAL** or **PNEUMONIA**.

The project combines **TensorFlow**, **Keras**, **EfficientNetB0**, **FastAPI**, local/GPU training, and **AWS S3** model storage in a configuration-driven machine learning workflow.

> **Note:** This project is intended as a machine-learning engineering demonstration and is **not a medical diagnostic system**.

## Overview

The project covers the complete ML workflow:

```text
Chest X-Ray Dataset
        │
        ▼
Data Preprocessing
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
        ├──────────────► Local Inference
        │
        ├──────────────► FastAPI
        │
        └──────────────► AWS S3
```

The same codebase can be used for:

* Local development
* Local GPU training
* AWS-based training
* Model inference
* Browser-based predictions
* Model evaluation

## Technology Stack

| Area                 | Technology                    |
| -------------------- | ----------------------------- |
| Programming Language | Python                        |
| Deep Learning        | TensorFlow / Keras            |
| Model                | EfficientNetB0                |
| Training             | Transfer Learning             |
| API                  | FastAPI                       |
| Server               | Uvicorn                       |
| Cloud Storage        | AWS S3                        |
| Configuration        | JSON + Environment Variables  |
| Dataset              | Chest X-Ray Pneumonia Dataset |

## Project Structure

```text
aws-pneumonia-detection/
├── config/
│   ├── config_local.json
│   ├── config_gpu.json
│   └── config_aws.json
├── inference/
│   └── predict.py
├── screenshots/
├── training/
│   ├── start_training.py
│   ├── evaluate.py
│   └── upload_model.py
├── utils/
│   └── config_loader.py
├── requirements.txt
├── gpu_test.py
└── README.MD
```

The application also contains the FastAPI application and its routes for prediction, training, model management, evaluation, and settings.

## Dataset

The project uses the **Kaggle Chest X-Ray Pneumonia Dataset**.

The expected structure is:

```text
data/
└── raw/
    └── chest_xray/
        ├── train/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        ├── val/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        └── test/
            ├── NORMAL/
            └── PNEUMONIA/
```

The classification task is binary:

```text
NORMAL
PNEUMONIA
```

The dataset is used for development and experimentation. It should not be interpreted as sufficient evidence for clinical deployment.

## Image Preprocessing

The preprocessing pipeline:

1. Loads X-ray images
2. Resizes images to `224 × 224`
3. Normalizes image data
4. Creates TensorFlow datasets
5. Feeds the processed images into the training pipeline

## Model

The project uses **EfficientNetB0** with transfer learning.

### Training configuration

| Parameter         | Value                 |
| ----------------- | --------------------- |
| Input size        | `224 × 224`           |
| Model             | EfficientNetB0        |
| Task              | Binary classification |
| Optimizer         | Adam                  |
| Learning rate     | `0.0001`              |
| Epochs            | 10                    |
| Transfer learning | Enabled               |

The trained model is stored as:

```text
models/
└── pneumonia_model.keras
```

The configuration is intentionally kept outside the training code so that different environments can use different settings.

## Configuration-Driven Architecture

Runtime configuration is selected through the `CONFIG_PATH` environment variable.

```text
CONFIG_PATH
     │
     ▼
config_loader.py
     │
     ▼
config_*.json
     │
     ├──────────────► Training
     ├──────────────► Inference
     └──────────────► FastAPI
```

Available configurations:

```text
config/config_local.json
config/config_gpu.json
config/config_aws.json
```

This makes it possible to use the same application code across different environments.

### Example

```bash
CONFIG_PATH=config/config_local.json \
python training/start_training.py
```

For GPU training:

```bash
CONFIG_PATH=config/config_gpu.json \
python training/start_training.py
```

For AWS:

```bash
CONFIG_PATH=config/config_aws.json \
python training/start_training.py
```

## Model Evaluation

The project evaluates the classifier using multiple metrics rather than relying only on accuracy.

### Accuracy

Overall percentage of correct predictions.

### Precision

How many samples predicted as pneumonia were actually pneumonia cases.

### Recall

How many pneumonia cases were detected by the model.

Recall is particularly important for this type of classification problem because false negatives can be significant.

### F1 Score

The harmonic mean of precision and recall.

```text
F1 = 2 × (precision × recall)
     --------------------------
       precision + recall
```

The FastAPI application exposes an evaluation page where these metrics can be displayed.

## FastAPI Application

The project includes a browser-accessible FastAPI application.

Start the application with:

```bash
CONFIG_PATH=config/config_local.json \
uvicorn api.main:app --reload
```

The application is available at:

```text
http://127.0.0.1:8000
```

## API Features

### Dashboard

```text
/
```

Provides an overview of the application and environment.

### Prediction

```text
/predict
```

Allows an X-ray image to be uploaded for inference.

The prediction workflow is:

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
Prediction
     │
     ▼
Confidence Score
```

The result contains the predicted class and confidence information.

### Training

```text
/train
```

Allows training to be started from the web interface.

The application uses FastAPI background tasks for training execution.

### Model Management

```text
/models
```

Provides information about the currently available model, including:

* Model path
* File size
* Modification time
* Model availability

### Evaluation

```text
/evaluation
```

Displays model evaluation information including:

* Accuracy
* Precision
* Recall
* F1 score

### Settings

```text
/settings
```

Displays the currently loaded:

* Environment
* Configuration
* Hardware settings
* AWS settings

## AWS Integration

AWS S3 is used for model storage.

The workflow is:

```text
Training
    │
    ▼
pneumonia_model.keras
    │
    ▼
AWS S3
    │
    ▼
Model Download
    │
    ▼
Inference
```

This separates model storage from the application runtime.

The configuration also supports AWS-specific execution through:

```text
config/config_aws.json
```

## Security

AWS credentials and secrets should **never** be committed to the repository.

Sensitive configuration should be provided through environment variables, IAM roles, or a dedicated secret-management solution.

For production AWS deployments, suitable approaches include:

* IAM roles
* EC2 instance profiles
* AWS Secrets Manager
* Restricted S3 bucket permissions

Do not place:

```text
AWS access keys
AWS secret keys
.env files
passwords
tokens
```

inside the repository.

## Local Setup

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the API

Start the local FastAPI application:

```bash
CONFIG_PATH=config/config_local.json \
uvicorn api.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## Train the Model

### Local

```bash
CONFIG_PATH=config/config_local.json \
python training/start_training.py
```

### Local GPU

```bash
CONFIG_PATH=config/config_gpu.json \
python training/start_training.py
```

### AWS

```bash
CONFIG_PATH=config/config_aws.json \
python training/start_training.py
```

## Run Inference

Inference can be performed through the FastAPI interface or directly through the inference module.

The general workflow is:

```text
Input Image
     │
     ▼
Image Preprocessing
     │
     ▼
Loaded Keras Model
     │
     ▼
Prediction
     │
     ▼
Class + Confidence
```

## Model Lifecycle

The repository separates the main stages of the model lifecycle:

```text
Dataset
   │
   ▼
Training
   │
   ▼
Evaluation
   │
   ▼
Model Artifact
   │
   ├──────────────► Local Inference
   │
   └──────────────► AWS S3
                         │
                         ▼
                     Deployment
```

This structure makes it easier to extend the project with model versioning, automated training, or a model registry.

## What This Project Demonstrates

This project combines several areas of software and machine-learning engineering:

### Machine Learning

* Image classification
* Transfer learning
* EfficientNet
* TensorFlow / Keras
* Model evaluation

### Software Engineering

* Modular project structure
* Configuration-driven execution
* Separation of training and inference
* REST API
* Background tasks

### Cloud Engineering

* AWS S3
* Cloud-based model storage
* Environment-specific configuration
* Preparation for AWS GPU workloads

### Deployment

* Local execution
* GPU execution
* AWS execution
* Browser-based inference

## Limitations

This repository is a portfolio and engineering project rather than a clinically validated system.

Important limitations include:

* Dataset-specific training
* Limited training configuration
* No clinical validation
* No external clinical test set
* No production monitoring
* No model governance
* No automated model versioning
* No authentication for the demonstration API

Predictions from the application should therefore **not be used for medical diagnosis or treatment decisions**.

## Future Improvements

Potential extensions include:

* Docker deployment
* AWS ECS deployment
* Automated CI/CD
* Model versioning
* Model registry
* Automated retraining
* AWS CloudWatch monitoring
* API authentication
* Prediction logging
* Dataset versioning
* Experiment tracking
* Hyperparameter optimization
* Model explainability
* Production-grade model serving

## Project Summary

This project implements an end-to-end deep learning workflow for chest X-ray image classification.

The main components are:

```text
TensorFlow / Keras
        │
        ▼
EfficientNetB0
        │
        ▼
Model Evaluation
        │
        ▼
FastAPI
        │
        ├──────────► Prediction
        ├──────────► Training
        ├──────────► Evaluation
        └──────────► Model Management
        │
        ▼
AWS S3
```

The project demonstrates how a trained computer-vision model can be integrated into a configurable Python application and connected to cloud-based model storage.

## Author

**Markus**

Machine Learning / Cloud Engineering Project

