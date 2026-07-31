import boto3

from utils.config_loader import load_config



config = load_config()


s3 = boto3.client(

    "s3",

    region_name=config["aws"]["region"]

)



s3.download_file(

    config["aws"]["bucket"],

    config["aws"]["s3_key"],

    config["model"]["path"]

)


print(
    "Model downloaded"
)
