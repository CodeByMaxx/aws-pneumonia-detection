from pathlib import Path
import boto3
from botocore.exceptions import ClientError


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "pneumonia_model.keras"
)

BUCKET_NAME = "codebymaxx-pneumonia-models"

S3_KEY = (
    "models/pneumonia_model.keras"
)


def upload_model():

    s3 = boto3.client(
        "s3"
    )

    try:

        print("Uploading model...")

        s3.upload_file(
            str(MODEL_PATH),
            BUCKET_NAME,
            S3_KEY
        )

        print(
            "Upload successful!"
        )

        print(
            f"s3://{BUCKET_NAME}/{S3_KEY}"
        )


    except ClientError as e:

        print(
            "Upload failed:"
        )

        print(e)


if __name__ == "__main__":

    upload_model()
