from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
import json


def REST(models, debug=False):
    app = Flask(__name__, template_folder="templates")
    cors = CORS(app)
    app.config["CORS_HEADERS"] = "Content-Type"

    @app.route("/models/<model>", methods=["GET"])
    @cross_origin()
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
        return {"error": True, "response": "Model does not exist."}, 404

    @app.route("/models/<model>", methods=["POST"])
    @cross_origin()
    def route_model_prediction(model):
        """
        This function takes in the model name and a single request parameter

        """
        if model in models:
            if request.form:
                return {
                    "error": False,
                    "response": models[model].predict(request.form.get("input")),
                }, 200
            else:
                return {"error": True, "response": "No code sent to the server."}, 401
        return {"error": True, "response": "Model does not exist."}, 404

    @app.route("/models", methods=["GET"])
    @cross_origin()
    def route_models():
        """
        This returns a list of full descriptions of all of the models that
        are currently running on the REST server.
        """
        vals = {}
        for key, value in models.items():
            vals[key] = str(value)
        return {"error": False, "response": vals}, 200

    @app.route("/", methods=["GET"])
    def main():
        """
        This just serves up the main page.
        """
        vals = [i for i in models.keys()]

        return render_template("index.html", models=json.dumps(vals))

    return app
