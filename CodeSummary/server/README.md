# Server

This directory contains all of the files required to serve up a model with
3 main endpoints:

1. GET <URL>/models - Gets all of the models
2. GET <URL>/models/\<model_id\> - Gets a specific models' information
3. POST <URL>/models/\<model_id\> - Gets the response from a specific model

This directory also contains a simple HTML page that queries the API and
displays some of it's functionality. This can be viewed on localhost:3000.