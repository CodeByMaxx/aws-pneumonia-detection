# AWS Pneumonia Detection

A deep-learning project for detecting pneumonia from chest X-ray images using **TensorFlow/Keras** and **EfficientNetB0**.

The project contains the training and inference components, configuration files, utilities, and screenshots of the resulting application/model workflow.

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
* Example results and screenshots

## 🛠️ Technology Stack

* **Python**
* **TensorFlow**
* **Keras**
* **EfficientNetB0**
* **NumPy**
* **Pillow**
* **FastAPI-oriented inference components**
* **AWS S3** for cloud-oriented data/model workflows

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
├── training/
├── utils/
├── .gitignore
├── gpu_test.py
├── requirements.txt
├── README.md
└── ...
```

The repository is organized around configuration, training, inference, and utility code. The `screenshots/` directory contains visual results from the project.

## ⚙️ Configuration

Configuration is separated from the training implementation so that the project can be adapted to different environments.

The configuration can be used for local development, GPU execution, and AWS-oriented workflows.

This keeps paths and environment-specific settings outside the core training logic.

## 🏋️ Training

The training implementation uses TensorFlow/Keras and EfficientNetB0.

A typical workflow is:

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

A small GPU test utility is also included:

```bash
python gpu_test.py
```

## 🔬 Model

The classifier uses **EfficientNetB0** as the base architecture.

EfficientNet provides a pretrained convolutional neural-network architecture that can be adapted to the pneumonia classification task through transfer learning.

The training pipeline is responsible for preparing the model and training it on the chest X-ray dataset.

## 📊 Evaluation

Evaluation is an important part of the workflow because training accuracy alone does not describe how well the model performs on unseen X-ray images.

The project therefore separates:

```text
Training Data
Validation Data
Test Data
```

so that the trained model can be evaluated on data that was not used directly for parameter optimization.

## 🖼️ Results & Screenshots

The repository contains screenshots documenting the results and visual output of the project.

### Result 1

![Pneumonia Detection Result](screenshots/1.png)

### Result 2

![Pneumonia Detection Result](screenshots/2.png)

### Result 3

![Pneumonia Detection Result](screenshots/3.png)

### Result 4

![Pneumonia Detection Result](screenshots/4.png)

> The screenshots are included as visual project results. They provide a quick way to inspect the application/model workflow without having to run the complete training pipeline.

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

Inference is separated from the training code so that an already trained model can be used without running the complete training pipeline again.

## ☁️ AWS Workflow

The project also contains configuration for AWS-oriented workflows.

AWS S3 can be used for storing datasets and model artifacts when the project is executed in a cloud environment.

A typical cloud workflow can be represented as:

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
```

The repository also keeps local configuration possible, which makes development and experimentation easier before moving workloads to AWS.

## 💻 Local Development

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The exact TensorFlow GPU setup depends on the operating system, GPU, CUDA, and TensorFlow version.

## 🧪 GPU Test

The repository contains:

```text
gpu_test.py
```

which can be used to verify whether the local TensorFlow environment can access a compatible GPU.

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

The project itself is intended as a technical/portfolio demonstration and should not be interpreted as a clinically validated diagnostic system.

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

