import json
from pathlib import Path


def load_config():

    config_path = Path(__file__).parent.parent / "config" / "config.json"

    with open(config_path, "r") as file:
        config = json.load(file)

    return config
