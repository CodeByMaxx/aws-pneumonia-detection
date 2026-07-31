import boto3

from utils.config_loader import load_config



config = load_config()


s3 = boto3.client(
    "s3",
    region_name=config["aws"]["region"]
)



s3.upload_file(

    config["model"]["path"],

    config["aws"]["bucket"],

    config["aws"]["s3_key"]

)


print(
    "Upload successful"
)


print(
    f"s3://{config['aws']['bucket']}/{config['aws']['s3_key']}"
)
