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
├── screen
```

