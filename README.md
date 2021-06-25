# CodeSummary
Deploy models as a local REST API with ease.

## Installation
If you want to run the server as a web server run the following:

1. Clone the repository on your own machine.
2. Create a virtual environment in the root directory `python -m venv venv`
3. Activate virtual environment `. venv/bin/activate` or `.\venv\Scripts\activate` on Windows
4. Install CodeSummary `pip install .`

If you want to install the `CodeSummary` library alone, install it via PIP
for Python 3.7:

```shell
pip install git+https://github.com/Nathan-Nesbitt/CodeSummary
```

## Running Basic Example

This is a demo server that loads 2 
versions of the LAMNER model. For more information see the license and 
associated information in the models directory. You can view this server by
visiting `localhost:3000`

### Development

If you want to run the local server that is included in the git repository,
you can simply run `export FLASK_APP=main:server` on Linux to set the environment
variables, or `$env:FLASK_APP = "main:server"` on Windows. 

You can then start the server by running `flask run --port 3000`. 

### Production

If you want to run this in production you should instead use gunicorn, which
can be done by running `sh gunicorn.sh` on Linux. You will need to install 
gunicorn first which can be done by running `pip install gunicorn`.

## Using the API

The API has a very simple syntax. There is 3 options:

1. GET /models
2. GET /models/\<model_id\>
3. POST /models/\<model_id\>

### GET /models

This endpoint simply lists all of the models currently running on the server.
It returns the following format:

```JSON
{
    "error": false,
    "response": {
        "id": {
            "name": "",
            "description": ""
        },
        ...
    }
}
```

For example, this is the default `main.py` script return values would be the 
following for two of the exact same models.

```JSON
{
    "error": false,
    "response": {
        "lamner": {
            "name": "LAMNER",
            "description": "LAMNER - Code Summarization"
        },
        "lamner_2":{
            "name": "LAMNER_2", 
            "description": "LAMNER - Code Summarization"
        }
    }
}
```

### GET /models/\<model_id\>

This gets specific information about a model. This is not used in the main 
script as we care about all of the models, not one specific model. This is more
if you already know the ID of your model and want to only load information about
it.

### POST /models/\<model_id\>

This endpoint takes in the following parameters using x-www-form-urlencoded:

```JSON
input: "String that is passed into the model"
```

It then returns a raw return, which contains the string response from the model.

```JSON
{
    "error": <true/false>,
    "response": <response from model>
}
```

## Adding Your Own Models

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
                "LAMNER", "LAMNER - Code Summarization", lamner.translate
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

## Docker
Since you may want to deploy this using docker and Kubernetes to allow for better
scaling you can use the included scripts. To do this locally you can do the 
following:

1. Install and start docker ([see here for more info](https://docs.docker.com/engine/install/))

2. Build the image `docker build -t codesummary . ` (be aware this is an insanely taxing process)

3. Run the container locally by running `docker run -d -p 3000:3000 codesummary` (make sure that apache/nginx is not running in the background)

4. Visit [http://localhost:3000](http://localhost:3000) to view the project locally.

If you want to see if it is running you can type `docker ps` or if you want
to kill it you can run `docker stop <process ID>`. 

## Deployment
The apache/nginx configuration is up to the deployer, the core concept is that 
you need to reverse proxy to the app listning on port 3000. 

Kubernetes deployment is also up to you, as it could be a whole project in itself.