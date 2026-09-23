# AWS Pneumonia Detection

A deep learning project for detecting pneumonia from chest X-ray images using TensorFlow and Keras.

The project demonstrates an image-classification workflow from model training and evaluation to prediction and an AWS-oriented deployment setup.

## Features

* Pneumonia detection from chest X-ray images
* TensorFlow / Keras
* EfficientNetB0-based image classification
* Training and evaluation workflow
* Image prediction
* GPU support
* AWS-oriented workflow
* S3 integration
* FastAPI-based inference components
* Visual results and screenshots

## Machine Learning Workflow

```text
Chest X-ray Images
        │
        ▼
Data Preparation
        │
        ▼
EfficientNetB0
        │
        ▼
Model Training
        │
        ▼
Evaluation
        │
        ▼
Prediction
```

## Technology Stack

* **Python**
* **TensorFlow**
* **Keras**
* **EfficientNetB0**
* **NumPy**
* **Pillow**
* **FastAPI**
* **AWS S3**

## Project Structure

```text
aws-pneumonia-detection/
├── config/
├── inference/
├── screenshots/
│   ├── models.png
│   ├── training.png
│   ├── dashboard.png
│   ├── evaluation.png
│   └── predict.png
├── training/
├── utils/
├── .gitignore
├── gpu_test.py
├── requirements.txt
└── README.md
```

### `config/`

Contains configuration used by the project.

### `training/`

Contains the training-related implementation for the pneumonia detection model.

### `inference/`

Contains components used for model inference and prediction.

### `utils/`

Contains supporting utilities used by the project.

### `screenshots/`

Contains visual results from the project, including model information, training, evaluation, dashboard, and prediction output.

## Configuration

Project configuration is kept separately under:

```text
config/
```

This keeps configuration values separate from the main training and inference implementation.

## Training

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The training workflow uses TensorFlow / Keras and an EfficientNetB0-based model for image classification.

Depending on the environment, TensorFlow can use an available GPU for accelerated training.

## Model

The classifier is based on **EfficientNetB0**.

The model is used to classify chest X-ray images according to the pneumonia detection task.

The general model workflow is:

```text
Input Image
    │
    ▼
Image Preprocessing
    │
    ▼
EfficientNetB0
    │
    ▼
Classification Layer
    │
    ▼
Prediction
```

## Evaluation

The project includes an evaluation workflow for measuring the trained model on the available evaluation data.

Evaluation results are visualized in the project screenshots.

## Results & Screenshots

The repository contains visual results of the training and inference workflow.

### Model

![Model](screenshots/models.png)

### Training

![Training](screenshots/training.png)

### Dashboard

![Dashboard](screenshots/dashboard.png)

### Evaluation

![Evaluation](screenshots/evaluation.png)

### Prediction

![Prediction](screenshots/predict.png)

## Inference

The `inference/` directory contains the components used for running predictions with the trained model.

A typical inference workflow is:

```text
X-ray Image
    │
    ▼
Preprocessing
    │
    ▼
Trained Model
    │
    ▼
Prediction
```

## AWS Workflow

The project is designed with an AWS-oriented workflow in mind.

AWS S3 can be used for storing and accessing machine-learning data and model-related artifacts.

A simplified workflow is:

```text
Dataset
   │
   ▼
AWS S3
   │
   ▼
Training
   │
   ▼
Trained Model
   │
   ▼
Inference
```

## Local Development

Create a Python environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

## GPU Test

The repository includes:

```text
gpu_test.py
```

This can be used to check the available GPU / TensorFlow environment.

Run:

```bash
python gpu_test.py
```

## Security & Data Handling

Chest X-ray data can contain sensitive medical information.

For real-world deployment, datasets and model artifacts should therefore be stored and accessed using appropriate access controls.

AWS credentials should not be committed to the repository.

Use environment variables or the AWS credential mechanisms provided by the deployment environment instead.

## Important Limitation

This project is primarily a machine-learning and portfolio project.

A pneumonia classifier should not be treated as a medical diagnostic system without extensive clinical validation, appropriate datasets, regulatory review, and expert oversight.

## Possible Improvements

Possible future extensions include:

* Improved dataset management
* More extensive model evaluation
* Additional classification metrics
* Confusion matrix visualization
* Model versioning
* Automated training workflows
* Improved AWS deployment
* API documentation
* Monitoring and logging
* More comprehensive inference testing

## Project Purpose

The purpose of this project is to demonstrate an end-to-end computer-vision workflow for pneumonia detection, combining deep learning with an AWS-oriented infrastructure.

The project covers:

* Image classification
* Transfer learning
* Model training
* Evaluation
* Inference
* GPU acceleration
* AWS S3 integration
* Visual presentation of results

## Author

**Markus**

