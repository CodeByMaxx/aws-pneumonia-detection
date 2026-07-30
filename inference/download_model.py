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


def download_model():

    s3 = boto3.client("s3")

    MODEL_PATH.parent.mkdir(
        exist_ok=True
    )

    try:

        print("Downloading model from S3...")

        s3.download_file(
            BUCKET_NAME,
            S3_KEY,
            str(MODEL_PATH)
        )

        print("Download successful!")
        print(
            f"Saved to: {MODEL_PATH}"
        )


    except ClientError as e:

        print("Download failed:")
        print(e)


if __name__ == "__main__":
    download_model()
