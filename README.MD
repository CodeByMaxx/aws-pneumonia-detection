# AWS Pneumonia Detection System

Deep Learning application for detecting pneumonia from chest X-Ray images using **TensorFlow, EfficientNetB0, FastAPI and AWS infrastructure**.

The goal of this project is to build an end-to-end machine learning system:

* Train a medical image classification model
* Evaluate model performance
* Store models in AWS S3
* Provide a web interface for prediction and training
* Deploy a production-style ML workflow

---

# Project Overview

## Problem

Pneumonia detection from chest X-Ray images is a binary image classification problem.

The model predicts:

```
NORMAL
```

or

```
PNEUMONIA
```

based on an input X-Ray image.

---

## Solution Architecture

```
Chest X-Ray Dataset

        |
        v

Data Preprocessing

        |
        v

TensorFlow Training

        |
        v

EfficientNetB0 Transfer Learning

        |
        v

Model Evaluation

        |
        v

pneumonia_model.keras

        |
        v

FastAPI Application

        |
        +----------------+
        |                |
        v                v

Prediction        Training Dashboard

        |
        v

AWS S3 Storage
```

---

# Project Demo Screenshots

## 1. Dashboard

*Add screenshot here*

Example:

```
screenshots/dashboard.png
```

The dashboard provides:

* System information
* GPU configuration
* Links to all application features

---

## 2. Prediction Interface

*Add screenshot here*

Example:

```
screenshots/predict.png
```

Features:

* Upload X-Ray image
* Run inference
* Display prediction result
* Display confidence score

Example:

```
Input:

Chest X-Ray

↓

Model

↓

Prediction:

PNEUMONIA

Confidence:

95.4%
```

---

## 3. Training Interface

*Add screenshot here*

Example:

```
screenshots/training.png
```

Features:

* Start training from browser
* Background execution
* Training status monitoring
* Log output

---

## 4. Model Management

*Add screenshot here*

Example:

```
screenshots/models.png
```

Displays:

* Model name
* Model location
* File size
* Last modification date
* Availability status

---

## 5. Evaluation Dashboard

*Add screenshot here*

Example:

```
screenshots/evaluation.png
```

Shows:

* Accuracy
* Precision
* Recall
* F1 Score
* Model performance

---

# Technology Stack

## Machine Learning

| Technology     | Purpose                 |
| -------------- | ----------------------- |
| TensorFlow     | Deep Learning Framework |
| EfficientNetB0 | Transfer Learning Model |
| Keras          | Model API               |
| OpenCV         | Image Processing        |
| Scikit-Learn   | Evaluation Metrics      |

---

## Backend

| Technology       | Purpose       |
| ---------------- | ------------- |
| FastAPI          | REST API      |
| Uvicorn          | ASGI Server   |
| Jinja2           | Web Templates |
| Python Multipart | File Uploads  |

---

## Cloud Infrastructure

| Technology | Purpose           |
| ---------- | ----------------- |
| AWS S3     | Model Storage     |
| AWS EC2    | GPU Training      |
| IAM        | Access Management |

---

# Machine Learning Pipeline

## Dataset

Dataset:

Kaggle Chest X-Ray Pneumonia Dataset

Structure:

```
data/

└── raw/

    └── chest_xray/

        ├── train/

        │   ├── NORMAL

        │   └── PNEUMONIA


        ├── validation/

        └── test/
```

---

# Model Training

The model uses:

* Transfer Learning
* EfficientNetB0
* Binary Classification
* GPU acceleration

Training configuration:

```
Image Size:

224 x 224


Epochs:

10


Optimizer:

Adam


Learning Rate:

0.0001
```

Output:

```
models/

└── pneumonia_model.keras
```

---

# Model Evaluation

Medical classification requires focusing on recall.

Important metrics:

## Accuracy

Overall correctness of predictions.

## Precision

How many positive predictions were correct.

## Recall

How many pneumonia cases were detected.

## F1 Score

Balance between precision and recall.

---

# FastAPI Web Application

The application provides:

## Dashboard

```
/
```

Overview of the system.

---

## Prediction

```
/predict
```

Upload an X-Ray image and receive:

```
Prediction:

PNEUMONIA


Confidence:

95%
```

---

## Training

```
/train
```

Starts model training from the browser.

Uses:

```
FastAPI BackgroundTasks
```

to run training without blocking the server.

---

## Model Information

```
/models
```

Shows:

* Current model
* Path
* Size
* Status

---

## Evaluation

```
/evaluation
```

Displays model metrics.

---

## Configuration

```
/settings
```

Shows current application configuration.

---

# Configuration

Example:

```
config/config_local.json
```

Contains:

```json
{
    "environment": "local",

    "model": {
        "path": "models/pneumonia_model.keras",
        "image_size": [
            224,
            224
        ]
    },

    "training": {
        "epochs": 10,
        "batch_size": 2
    },

    "hardware": {
        "device": "gpu"
    }
}
```

---

# AWS Workflow

```
Local Training

        |

        v

pneumonia_model.keras

        |

        v

AWS S3 Bucket

        |

        v

Download Model

        |

        v

Inference
```

---

# Security

Sensitive information is never stored in the repository.

Protected:

```
.env

AWS credentials

Access keys

Private configuration
```

Production approach:

* IAM Roles
* EC2 Instance Profiles
* AWS Secrets Manager

---

# Local Installation

Clone repository:

```bash
git clone https://github.com/CodeByMaxx/aws-pneumonia-detection.git

cd aws-pneumonia-detection
```

Create environment:

```bash
python3 -m venv .venv
```

Activate:

Linux:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Start application:

```bash
uvicorn api.main:app --reload
```

Open:

```
http://127.0.0.1:8000
```

---

# Future Improvements

Possible extensions:

* Docker deployment
* AWS ECS deployment
* CI/CD pipeline
* Automated retraining
* Training with different parameters

---

# Interview Explanation

Short explanation:

> "I built an end-to-end machine learning application for pneumonia detection. The project includes data preprocessing, transfer learning with EfficientNetB0, model evaluation, AWS S3 model storage and a FastAPI web interface where users can run predictions and trigger training."

---

# Author

Markus

Machine Learning / Cloud Project

