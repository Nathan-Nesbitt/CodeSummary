from flask import Flask, request
import json


def REST(models, debug=False):
    app = Flask(__name__, template_folder="templates")

    @app.route("/models/<model>", methods=["GET"])
    def route_model_description(model):
        if model in models:
            return {
                "model_name": models[model].name,
                "model_description": models[model].description,
            }
        return "Model does not exist.", 404

    @app.route("/models/<model>", methods=["POST"])
    def route_model_prediction(model):
        if model in models:
            user_input = request.form["input"]
            return models[model].predict(user_input), 200
        return "Model does not exist.", 404

    @app.route("/models", methods=["GET"])
    def route_models():
        vals = {}
        for key, value in models.items():
            vals[key] = str(value)
        return vals

    # Starts the flask server
    app.run(debug=False, host="0.0.0.0", port=3000)
