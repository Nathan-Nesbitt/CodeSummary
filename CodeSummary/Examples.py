from .server.REST import REST
from .models.Model import Model
from .models.lamner.Lamner import Lamner


class BasicTutorial:
    """
    Class for each model that is passed in.
    """

    def __init__(self):
        """
        Creates a basic object
        """
        super().__init__()

        # Creates the models
        def predict(value):
            return "You have successfully send the data: {}".format(value)

        models = {
            "TestModel": Model(
                "TestModel", "A test model to show this can work", None, predict
            )
        }

        rest_server = REST(models, debug=True)

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