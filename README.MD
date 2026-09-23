# AWS Pneumonia Detection

A deep-learning project for detecting pneumonia from chest X-ray images using **TensorFlow/Keras** and **EfficientNetB0**.

The project contains the training and inference components, configuration files, utilities, and screenshots documenting the model and application workflow.

## ✨ Features

* Pneumonia classification from chest X-ray images
* Transfer learning with **EfficientNetB0**
* TensorFlow / Keras training pipeline
* Configurable training settings
* GPU support when available
* Separate training and inference components
* Local and AWS-oriented configuration
* Model evaluation
* Saved trained model artifacts
* Visual results and screenshots

## 🛠️ Technology Stack

* **Python**
* **TensorFlow**
* **Keras**
* **EfficientNetB0**
* **NumPy**
* **Pillow**
* **FastAPI**
* **AWS S3**

## 🧠 Machine Learning Workflow

```text
Chest X-Ray Dataset
        │
        ▼
 Data Loading
        │
        ▼
 Preprocessing
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
 Trained Model
        │
        ▼
 Inference
```

## 📂 Project Structure

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

The repository is organized around configuration, training, inference, and utility code. The `screenshots/` directory contains visual results from the project.

## ⚙️ Configuration

Configuration is separated from the training implementation so that the project can be adapted to different environments.

The configuration supports local development, GPU execution, and AWS-oriented workflows.

This keeps environment-specific paths and settings outside the main training logic.

## 🏋️ Training

The training implementation uses **TensorFlow/Keras** and **EfficientNetB0**.

The general workflow is:

```text
1. Load configuration
2. Load dataset
3. Prepare training/validation/test data
4. Preprocess images
5. Initialize EfficientNetB0
6. Train the classifier
7. Evaluate the model
8. Save the trained model
```

The project can use a GPU when TensorFlow detects a compatible CUDA environment.

A GPU test utility is included:

```bash
python gpu_test.py
```

## 🔬 Model

The classifier uses **EfficientNetB0** as its base architecture.

EfficientNetB0 is used as a transfer-learning model and adapted to the pneumonia classification task.

The training pipeline prepares the image data, trains the classifier, evaluates the resulting model, and saves the trained model artifact.

## 📊 Evaluation

The project separates the available data into training, validation, and test workflows.

```text
Training Data
      │
      ▼
Model Training
      │
      ▼
Validation
      │
      ▼
Final Evaluation
      │
      ▼
Prediction
```

The evaluation results are documented visually in the repository screenshots.

## 🖼️ Results & Screenshots

The repository contains screenshots showing the different stages and results of the project.

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

These screenshots provide a visual overview of the model, training process, dashboard, evaluation, and prediction workflow.

## 🔎 Inference

The `inference/` directory contains the inference-related implementation.

The intended workflow is:

```text
Input X-Ray
    │
    ▼
Image Preprocessing
    │
    ▼
Trained EfficientNet Model
    │
    ▼
Prediction
```

The inference workflow is separated from training so that an already trained model can be used without running the complete training process again.

## ☁️ AWS Workflow

The project also contains configuration for AWS-oriented workflows.

AWS S3 can be used for storing datasets and model artifacts when the project is executed in a cloud environment.

A typical workflow is:

```text
AWS S3
  │
  ▼
Training Environment
  │
  ▼
TensorFlow / EfficientNetB0
  │
  ▼
Trained Model
  │
  ▼
Model Artifact
  │
  ▼
Inference
```

The project can also be configured for local development and GPU-based training.

## 💻 Local Development

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

The exact TensorFlow GPU setup depends on the operating system, GPU, CUDA, and TensorFlow version.

## 🧪 GPU Test

The repository contains:

```text
gpu_test.py
```

which can be used to check whether the local TensorFlow environment can access a compatible GPU.

Run:

```bash
python gpu_test.py
```

## 🔐 Security & Data Handling

Chest X-ray datasets can contain sensitive medical information.

When using real-world medical datasets or cloud storage, appropriate access controls and data-protection measures are required.

For AWS deployments, this includes:

* restricted S3 permissions
* secure IAM policies
* private buckets where appropriate
* avoiding credentials in source code
* appropriate handling of medical data

The project is intended as a technical and portfolio demonstration and should not be interpreted as a clinically validated diagnostic system.

## ⚠️ Important Limitation

This project demonstrates an image-classification workflow using a medical dataset.

A machine-learning model trained for this purpose should **not be treated as a medical diagnosis system** without extensive clinical validation, appropriate datasets, regulatory review, and real-world evaluation.

## 🔮 Possible Improvements

Potential future improvements include:

* additional model evaluation metrics
* confusion matrix visualization
* ROC/AUC analysis
* explainability with Grad-CAM
* improved experiment tracking
* automated model evaluation
* containerized inference
* dedicated API deployment
* CI/CD for the ML pipeline
* additional AWS automation

## 🎯 Project Purpose

The project demonstrates an end-to-end computer-vision workflow:

```text
Dataset
   ↓
Preprocessing
   ↓
Transfer Learning
   ↓
Training
   ↓
Evaluation
   ↓
Model Artifact
   ↓
Inference
```

It combines **deep learning, computer vision, Python, and AWS-oriented workflows** in a single portfolio project.

---

**Project:** AWS Pneumonia Detection
**Author:** Markus

