# CodeSummary
Deploy models as a local REST API with ease.

## Installation
The easiest way to install the `CodeSummary` library, install it via PIP:

```shell
pip install CodeSummary
```

This must be run on python 3.7.

## Running

At the moment you can run the `main.py` file which starts a basic server locally.
If you want to run this on a server you can use gunicorn.

## Running your own models

This library is set up so that you can pass in any model that is initialized
using an object and has a method that accepts a string parameter and returns a
single prediction.

For example, for LAMNER we simply initialize a new class that extends the 
original class, initializes the `translate` method which is then passed into
the Model object along with any. 

As you may want to run multiple models at the same time on the same machine, 
you can specify multiple models and pass them into the `REST` object to be 
served.

```py
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
                "LAMNER", "LAMNER - Code Summarization", None, lamner.translate
            )
        }

        self.rest_server = REST(models, debug=True)
```

If you want to run multiple models at the same time you can do the following:

```py
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
                "LAMNER", "LAMNER - Model One Example", lamner_1.translate
            ),

            "lamner_2": Model(
                "LAMNER_2", "LAMNER - This is a second instance of the model", lamner_2.translate
            )
        }
        
        self.rest_server = REST(models)

```