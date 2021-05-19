# CodeSummary
Deploy models as a local REST API with ease.

## Installation
The easiest way to install the `CodeSummary` library, install it via PIP:

```shell
pip install CodeSummary
```

This must be run on python 3.7.

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
class LamnerExample(Lamner):
    """
    Example class for the LAMNER NLP model.
    """
    def __init__(self):
        """
        Creates a basic object
        """
        super().__init__()

        models = {
            "lamner": Model(
                "LAMNER", "LAMNER - Code Summarization", None, self.translate
            )
        }

        self.rest_server = REST(models, debug=True)
```