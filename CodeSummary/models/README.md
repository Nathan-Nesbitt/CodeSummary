# Models

This is a folder full of the different models that can be run within the API.
You don't have to store the model in this directory to use it, you simply need
to have an object that initializes the model, and a function that takes in a 
string input and returns a string form of the prediction. If you want to install
the model within the library, we do not want to store files that are larger than
100MB on GitHub so we can simply modify the models script to download the files
from a server.

## Adding a model

To use the model in the REST API, you simply need to use the `Model` class. It 
takes in a few parameters

`name` - The name of the model
`description` - The description of the model
`model` - The model object for predictions
`predict` - The function that makes the prediction. This must take a string 
    input and return a string output. 

You then can modify the `models.py` file to download the required zipped model

```
models = {
    Model Name: {
        "server_location": Location where the model is being hosted,
        "local_location": Local file where you will save the model,
        "root_folder_names": [
            This is the main folder names within the zip file, this allows
            us to avoid re-downloading the files if they already exist on 
            the local system.
        ]
    }
}
```

For example the LAMNER model is described as the following which downloads and
unzips the files to the lamner directory.

models = {
    "LAMNER": {
        "server_location": "",
        "local_location": "lamner",
        "root_folder_names": [
            "custom_embeddings",
            "custom_embeddings_decoder",
            "custom_embeddings_semantic_encoder",
            "custom_embeddings_syntax_encoder",
            "best-seq2seq.pt"
        ]
    }
}

## Using the model

Since you don't always want all the models, you can simply import the models
using the `get_models` script. Then create a dict of `Model` objects, which take
a name, a description, a model, and a translation. 

For example you can install and run the lamner model as defined in the `Examples.py`
file. 

```py
from .server.REST import REST
from .models.Model import Model
from .models.lamner.Lamner import Lamner
from .models.models import get_models

class LamnerExample:
    """
    Example class for the LAMNER NLP model.
    """
    def __init__(self):
        """
        Creates a basic object
        """
        super().__init__()

        # Downloads the lamner model
        get_models(["LAMNER"])

        lamner = Lamner()
        
        models = {
            "lamner": Model(
                "LAMNER", "LAMNER - Code Summarization", lamner.translate
            )
        }
        
        self.rest_server = REST(models)
```

If you want to run multiple models at the same time you can just add more Model
objects:

```py
from .server.REST import REST
from .models.Model import Model
from .models.lamner.Lamner import Lamner
from .models.models import get_models

class Example:
    """
    Example of multiple models.
    """
    def __init__(self):
        """
        Creates a basic object
        """
        super().__init__()

        # Downloads the lamner model
        get_models(["LAMNER"])

        # Running two instances of LAMNER for some reason
        lamner_1 = Lamner()
        lamner_2 = Lamner()

        # Defines two models running on the server
        models = {
            "lamner": Model(
                "LAMNER", "LAMNER - Code Summarization", lamner_1.translate
            ),

            "lamner_2": Model(
                "LAMNER_2", "LAMNER - Code Summarization", lamner_2.translate
            )
        }
        
        self.rest_server = REST(models)
```