# AWS Pneumonia Detection Model

Deep Learning project for detecting pneumonia from chest X-Ray images using **TensorFlow** and **EfficientNetB0**.

The project supports:

* Local GPU training
* Model evaluation
* Local inference
* AWS S3 model storage
* Optional AWS EC2 GPU training

---

# Project Structure

```text
aws-pneumonia-detection/

├── data/
│   └── raw/
│       └── chest_xray/
│           ├── train/
│           ├── test/
│           └── val/

├── models/
│   └── pneumonia_model.keras

├── training/
│   ├── preprocess.py
│   ├── start_training.py
│   ├── evaluate.py
│   └── upload_model.py

├── inference/
│   ├── predict.py
│   └── download_model.py

├── requirements.txt
└── README.md
```

---

# Project Workflow

```text
Dataset

    ↓

Preprocessing

    ↓

TensorFlow Training

    ↓

EfficientNetB0 Model

    ↓

pneumonia_model.keras

    ↓

Amazon S3 Storage

    ↓

Inference / Prediction
```

---

# Local Setup and Training

The model can be trained locally using an NVIDIA GPU or moved to AWS GPU infrastructure.

---

## 1. Clone Repository

```bash
git clone https://github.com/CodeByMaxx/aws-pneumonia-detection.git

cd aws-pneumonia-detection
```

---

## 2. Create Python Environment

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

## 3. Install Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Main dependencies:

* TensorFlow
* NumPy
* Pandas
* Scikit-Learn
* OpenCV
* boto3

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

# GPU Verification

Check TensorFlow GPU support:

```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
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
2.16.1
```

---

# Training Pipeline

## Preprocessing

Run:

```bash
python3 training/preprocess.py
```

Example output:

```text
Found 5216 files belonging to 2 classes.
```

---

## Start Training

Run:

```bash
python3 training/start_training.py
```

The model uses:

* TensorFlow
* Transfer Learning
* EfficientNetB0
* Binary Classification

Training output:

```text
Epoch 1/10

loss:
accuracy:
val_loss:
val_accuracy:
```

After training:

```text
models/

└── pneumonia_model.keras
```

---

# Model Evaluation

Evaluate the trained model:

```bash
python3 training/evaluate.py
```

The evaluation provides:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

For medical classification, **Recall** is especially important because false negatives should be minimized.

---

# Local Inference

Run prediction:

```bash
python3 inference/predict.py \
data/raw/chest_xray/test/PNEUMONIA/example.jpeg
```

Example:

```text
Prediction:

PNEUMONIA

Confidence:
95.40%
```

---

# AWS Model Storage Setup

The project uses **Amazon S3** to store trained models.

AWS workflow:

```text
Training

    ↓

pneumonia_model.keras

    ↓

Amazon S3 Bucket

    ↓

Download Model

    ↓

Inference
```

---

# AWS Account Setup

## 1. Create AWS Account

Create an AWS account:

https://aws.amazon.com/

Open the AWS Management Console after login.

---

# IAM Configuration

## 2. Create IAM User

Do not use the AWS root account.

Create a dedicated IAM user:

```text
AWS Console

→ IAM

→ Users

→ Create user
```

Example:

```text
pneumonia-detection-user
```

Enable programmatic access.

The user requires access keys for AWS CLI authentication.

---

## 3. Configure IAM Permissions

The user requires S3 permissions.

Example policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": "*"
        }
    ]
}
```

Attach the policy to the IAM user.

For production environments, restrict permissions to the required S3 bucket only.

---

## 4. Create AWS Access Keys

Open:

```text
AWS Console

→ IAM

→ Users

→ pneumonia-detection-user

→ Security credentials
```

Create a new access key.

You receive:

```text
AWS Access Key ID

AWS Secret Access Key
```

Example environment variables:

```text
AWS_ACCESS_KEY_ID

AWS_SECRET_ACCESS_KEY
```

Store them securely.

Never upload credentials to GitHub.

---

# AWS CLI Setup

## 5. Install AWS CLI

Install:

https://aws.amazon.com/cli/

Verify:

```bash
aws --version
```

Example:

```text
aws-cli/2.x.x
```

---

## 6. Configure AWS CLI

Run:

```bash
aws configure
```

Enter:

```text
AWS Access Key ID:
YOUR_ACCESS_KEY

AWS Secret Access Key:
YOUR_SECRET_KEY

Default region name:
eu-central-1

Default output format:
json
```

---

## 7. Test AWS Connection

Verify authentication:

```bash
aws sts get-caller-identity
```

Example:

```json
{
    "UserId": "xxxxxxxx",
    "Account": "xxxxxxxx",
    "Arn": "arn:aws:iam::xxxxxxxx:user/pneumonia-detection-user"
}
```

If the IAM user information is returned, the AWS configuration is complete.

---

# Amazon S3 Model Storage

## 8. Create S3 Bucket

Create a bucket:

```text
AWS Console

→ S3

→ Create bucket
```

Example:

```text
codebymaxx-pneumonia-models
```

Bucket names must be globally unique.

Recommended structure:

```text
s3://codebymaxx-pneumonia-models/

└── models/

    └── pneumonia_model.keras
```

---

# Upload Model

After successful training:

```bash
python3 training/upload_model.py
```

Uploads:

```text
models/pneumonia_model.keras
```

to:

```text
s3://YOUR_BUCKET_NAME/models/pneumonia_model.keras
```

---

# Download Model

For inference:

```bash
python3 inference/download_model.py
```

The model is stored locally:

```text
models/

└── pneumonia_model.keras
```

---

# AWS Security

Never commit sensitive files:

```text
.aws/
.env
credentials
```

Never upload:

* AWS Access Keys
* AWS Secret Keys
* AWS credentials
* Private configuration files

For production environments use:

* IAM Roles
* EC2 Instance Profiles
* AWS Secrets Manager

---

# Optional AWS EC2 GPU Training

For larger workloads, training can run on AWS GPU instances.

Recommended instances:

```text
g4dn.xlarge

g5.xlarge
```

Workflow:

```text
AWS EC2 GPU Instance

        ↓

NVIDIA Drivers

        ↓

CUDA + TensorFlow

        ↓

Dataset Download

        ↓

Model Training

        ↓

Upload Model to S3
```

---

# Complete End-to-End Workflow

```text
Developer Machine

        ↓

Prepare Dataset

        ↓

Train Model

        ↓

Evaluate Model

        ↓

Test Prediction

        ↓

Upload Model

        ↓

Amazon S3

        ↓

Download Anywhere

        ↓

Inference
```

