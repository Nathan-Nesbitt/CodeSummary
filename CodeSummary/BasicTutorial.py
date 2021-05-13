from .server.REST import REST
from .models.Model import Model


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
