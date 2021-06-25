from urllib import request
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import os
import CodeSummary
from pathlib import Path
from tqdm import tqdm


current_path = Path(os.path.expanduser("~"), "Documents", ".models")

models = {
    "LAMNER": {
        "server_location": "http://52.24.120.181/lamner.zip",
        "local_location": "lamner",
        "root_folder_names": [
            "custom_embeddings",
            "custom_embeddings_decoder",
            "custom_embeddings_semantic_encoder",
            "custom_embeddings_syntax_encoder",
            "best-seq2seq.pt",
        ],
    }
}


def get_models(download_models=None):
    """
    This runs through all models and downloads them from
    remote servers. To add a new model, simply append
    the model to the 'models' dict with a server location,
    that contains a zip file that can be extracted and a
    local location to download and unzip this file to. There
    is an optional root_folder_names which specifies what will
    be downloaded, and will stop the model from being re-downloaded.

    If you want to just download a subset of models simply
    specify the models name.
    """

    down_models = {}
    if download_models:
        for i in download_models:
            if i not in models:
                raise KeyError(
                    "{} does not exist, please chose from the following models: {}".format(
                        i, [i for i in models.keys()]
                    )
                )
            else:
                down_models[i] = models[i]
    else:
        down_models = models

    for model_name, model_info in down_models.items():
        # Only download if the file doesn't already exist
        exists = False
        if model_info["root_folder_names"]:
            for folder in model_info["root_folder_names"]:
                if Path(current_path, model_info["local_location"], folder).exists():
                    exists = True
                    break
        if not exists:
            print(f"Downloading {model_name} model...")
            with request.urlopen(model_info["server_location"]) as url:
                length = int(url.headers.get("content-length"))
                stream = BytesIO()
                block_size = max(4096, length // 20)
                with tqdm(total=length, position=0) as tq:
                    while True:
                        data = url.read(block_size)
                        if not data:
                            break
                        stream.write(data)
                        tq.update(block_size)
                print(f"Extracting {model_name} model...")
                with ZipFile(stream) as zipped:
                    zipped.extractall(Path(current_path, model_info["local_location"]))
