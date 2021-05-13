import json


class Model:
    """
    Class for each model that is passed in.
    """

    def __init__(self, name, description, model, predict):
        super().__init__()
        self.name = name
        self.description = description
        self.model = model
        self.predict = predict

    def predict(self, input):
        self.predict(input)

    def __str__(self):
        return json.dumps({"name": self.name, "description": self.description})
