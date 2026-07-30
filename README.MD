## AWS Setup for Pneumonia Detection Model

This project uses AWS services to store and manage the trained pneumonia detection model.

The AWS workflow:

```text
Training on GPU
        |
        ↓
pneumonia_model.keras
        |
        ↓
Amazon S3 Bucket
        |
        ↓
Download Model
        |
        ↓
Inference / Prediction
```

---

# 1. Create an AWS Account

Create an AWS account if you do not already have one:

https://aws.amazon.com/

After login, open the AWS Management Console.

---

# 2. Create an IAM User

For security reasons, do not use the AWS root account for this project.

Create a dedicated IAM user:

1. Open:

```text
AWS Console → IAM → Users → Create user
```

2. Create a user, for example:

```text
pneumonia-detection-user
```

3. Enable access for programmatic usage.

The user needs access keys for AWS CLI authentication.

---

# 3. Create IAM Permissions

The user requires access to Amazon S3.

Create a policy with the following permissions:

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

Attach this policy to the IAM user.

For production environments, restrict the permissions to the specific S3 bucket.

---

# 4. Create AWS Access Keys

Open:

```text
AWS Console
→ IAM
→ Users
→ pneumonia-detection-user
→ Security credentials
```

Create a new access key.

You will receive:

```text
AWS Access Key ID
AWS Secret Access Key
```

Example:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```

Store these values securely.

Do not upload them to GitHub.

---

# 5. Install AWS CLI

Install AWS CLI:

https://aws.amazon.com/cli/

Check installation:

```bash
aws --version
```

Example:

```text
aws-cli/2.x.x
```

---

# 6. Configure AWS CLI

Configure your AWS credentials:

```bash
aws configure
```

Enter the credentials from your IAM user:

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

# 7. Test AWS Connection

Verify that AWS CLI can authenticate:

```bash
aws sts get-caller-identity
```

Example response:

```json
{
    "UserId": "xxxxxxxx",
    "Account": "xxxxxxxx",
    "Arn": "arn:aws:iam::xxxxxxxx:user/pneumonia-detection-user"
}
```

If the command returns your IAM user information, the AWS configuration is complete.

---

# 8. Create an S3 Bucket

Create an S3 bucket for storing the trained model.

Open:

```text
AWS Console
→ S3
→ Create bucket
```

Example bucket name:

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

# 9. Upload the Trained Model

After successful training, the model is saved locally:

```text
models/
└── pneumonia_model.keras
```

Upload the model:

```bash
python3 training/upload_model.py
```

The script uploads:

```text
models/pneumonia_model.keras
```

to:

```text
s3://YOUR_BUCKET_NAME/models/pneumonia_model.keras
```

---

# 10. Download Model from S3

For inference, the model can be downloaded from S3:

```bash
python3 inference/download_model.py
```

The model will be stored locally:

```text
models/
└── pneumonia_model.keras
```

---

# 11. AWS Security

Never commit sensitive AWS files.

Add these entries to `.gitignore`:

```text
.aws/
.env
```

Never upload:

* AWS Access Keys
* AWS Secret Keys
* AWS credentials
* private configuration files

For production environments use:

* IAM Roles
* EC2 Instance Profiles
* AWS Secrets Manager

---

# 12. Optional: GPU Training on AWS EC2

The model can also be trained on an AWS GPU instance.

Recommended instance types:

```text
g4dn.xlarge
g5.xlarge
```

Workflow:

```text
EC2 GPU Instance

        ↓

Install NVIDIA Drivers
        ↓
Install CUDA + TensorFlow
        ↓
Download Dataset
        ↓
Train Model
        ↓
Upload Model to S3
```

This allows scalable GPU training without using local hardware.
# AWS Pneumonia Detection Model

Deep Learning project for detecting pneumonia from chest X-Ray images using TensorFlow and EfficientNetB0.

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

# Local Setup and Training

The project can be trained locally using an NVIDIA GPU or moved to AWS GPU infrastructure.

Workflow:

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

AWS S3 Storage
```

---

# 1. Clone Repository

```bash
git clone https://github.com/CodeByMaxx/aws-pneumonia-detection.git

cd aws-pneumonia-detection
```

---

# 2. Create Python Environment

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate environment:

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

# 4. Dataset Preparation

This project uses the Kaggle Chest X-Ray Pneumonia dataset.

Download dataset:

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

# 5. Check GPU Availability

The project supports NVIDIA GPUs.

Check TensorFlow GPU detection:

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

# 6. Dataset Preprocessing

Run:

```bash
python3 training/preprocess.py
```

Example output:

```text
Found 5216 files belonging to 2 classes.
```

---

# 7. Start Training

Start model training:

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

For medical classification, Recall is especially important because false negatives should be minimized.

---

# Local Inference

Run prediction on a single X-Ray image:

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

# AWS Setup for Pneumonia Detection Model

The project uses AWS S3 for storing trained models.

AWS workflow:

```text
Local Training

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

# 1. Create AWS Account

Create an AWS account:

https://aws.amazon.com/

Login to the AWS Management Console.

---

# 2. Create IAM User

Do not use the AWS root account.

Create a dedicated user:

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

Create an access key for programmatic access.

---

# 3. IAM Permissions

The user requires S3 access.

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

For production environments, restrict access to the required bucket only.

---

# 4. Create AWS Access Keys

Go to:

```text
IAM

→ Users

→ pneumonia-detection-user

→ Security credentials
```

Create:

```text
Access Key
```

You receive:

```text
AWS Access Key ID

AWS Secret Access Key
```

Keep these credentials private.

Never upload them to GitHub.

---

# 5. Install AWS CLI

Install:

https://aws.amazon.com/cli/

Check:

```bash
aws --version
```

---

# 6. Configure AWS CLI

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

# 7. Test AWS Connection

Run:

```bash
aws sts get-caller-identity
```

Successful output shows your IAM user information.

---

# 8. Create S3 Bucket

Create:

```text
AWS Console

→ S3

→ Create bucket
```

Example:

```text
codebymaxx-pneumonia-models
```

Recommended structure:

```text
s3://codebymaxx-pneumonia-models/

└── models/

    └── pneumonia_model.keras
```

---

# Upload Model to S3

After training:

```bash
python3 training/upload_model.py
```

The model will be uploaded:

```text
models/pneumonia_model.keras
```

to:

```text
S3 Bucket

└── models/

    └── pneumonia_model.keras
```

---

# Download Model from S3

Download the model:

```bash
python3 inference/download_model.py
```

The file is stored locally:

```text
models/

└── pneumonia_model.keras
```

---

# AWS Security

Never commit:

```text
.aws/
.env
credentials
```

Never upload:

* AWS Access Keys
* AWS Secret Keys
* Private credentials

For production use:

* IAM Roles
* EC2 Instance Profiles
* AWS Secrets Manager

---

# Optional AWS EC2 GPU Training

For larger training workloads the project can run on AWS GPU instances.

Recommended:

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

# Complete Workflow

```text
Developer Machine

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

