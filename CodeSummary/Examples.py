from .server.REST import REST
from .models.Model import Model
from .models.lamner.Lamner import Lamner
from .models.models import get_models


class BasicTutorial:
    """
    Class for each model that is passed in.
    """

    def __init__(self):
        """
        Creates a basic object
        """
        super().__init__()

        # Ensures the models are installed
        get_models()

        # Creates the models
        def predict(value):
            return "You have successfully send the data: {}".format(value)

        models = {
            "TestModel": Model(
                "TestModel", "A test model to show this can work", predict
            )
        }

        self.rest_server = REST(models, debug=True)


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
                "LAMNER",
                """
                LAMNER is a deep learning model based on the transformers architecture developped at UBC.
                It takes in a function, parses the text, and produces a description of the code segment.
                It is intended to be used for full Java functions.
                """,
                lamner.translate,
            )
        }

        self.rest_server = REST(models)


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
                "LAMNER",
                """
                LAMNER is a deep learning model based on the transformers architecture developped at UBC.
                It takes in a function, parses the text, and produces a description of the code segment.
                It is intended to be used for full Java functions.
                """,
                lamner_1.translate,
            ),
            "lamner_2": Model(
                "LAMNER_2",
                "This is the second version of lamner that is currently running.",
                lamner_2.translate,
            ),
        }

        self.rest_server = REST(models)
