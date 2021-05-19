from flask import Flask, request
import json


def REST(models, debug=False):
    app = Flask(__name__, template_folder="templates")

    @app.route("/models/<model>", methods=["GET"])
    def route_model_description(model):
        """
            Gets the description of a single model. This is useful if you
            want to provide a user with a full description in one page.
        """
        if model in models:
            return {
                "model_name": models[model].name,
                "model_description": models[model].description,
            }
        return "Model does not exist.", 404

    @app.route("/models/<model>", methods=["POST"])
    def route_model_prediction(model):
        """
            This function takes in the model name and a single request parameter

        """
        if model in models:
            if request.form:
                return models[model].predict(request.form["input"]), 200
            else:
                return "No code sent to the server.", 401
        return "Model does not exist.", 404

    @app.route("/models", methods=["GET"])
    def route_models():
        """
            This returns a list of full descriptions of all of the models that
            are currently running on the REST server.
        """
        vals = {}
        for key, value in models.items():
            vals[key] = str(value)
        return vals

    # Starts the flask server
    app.run(debug=False, host="0.0.0.0", port=3000)
