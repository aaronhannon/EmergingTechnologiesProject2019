# EmergingTechnologiesProject2019
### Name: Aaron Hannon

# Project Summary
The aim of this project was to create a web application that allows the user to draw a number from 0 - 9 and the server returns a prediction using the keras deep learning library. Once the user clicks the send button the Image that the user made gets sent to the python flask server via a post request. The server does some pre-processing before making the prediction. Once the image has been processed it is inputted into the keras model. If the model does not exist then it trains it. If it does, the keras neural network returns an array of probabilities. The prediction is the index of the highest number in that array and that is returned to the user and displayed.

# Required Imports

If an import error occurs when trying to run pip install or conda install the required package. Below I have listed the imports in both the flaskapp python file and the kerasModel python file. However there is a model.h5 file so you should not have to re-train the model.

## File:flaskapp.py

``` from flask import Flask, json, jsonify, render_template, request
import numpy as np
from kerasModel import prediction
import sys
from PIL import Image
import re
import base64
from keras.models import load_model
import io
from io import BytesIO
```

## File:kerasModel.py
```
import gzip
import numpy as np
import keras as kr
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.models import load_model
import sklearn.preprocessing as pre
import matplotlib.pyplot as plt
```
# How it works
View comments in the following files to see how everything works
- ```flaskapp.py```
- ```kerasModel.py```
- ```index.html```

# How to run
## Application:
- Clone the repository using: 
    - ``` git clone https://github.com/aaronhannon/EmergingTechnologiesProject2019 ```
- Navigate to the FlaskApp directory.
- Run the flask server using:
    - ``` python flaskapp.py```
- Go to localhost:5000
- Draw a number from 0-9
- See Prediction


## Jupyter Notebook: 
- Open notebook using:
    - ``` jupyter lab ```


# References & Research
## Flask App:
- Simple flask server:
    - http://flask.palletsprojects.com/en/1.1.x/quickstart/#a-minimal-application
- Fix issue with CORS:
    - https://stackoverflow.com/a/28339918
- Return a static page:
    - http://flask.palletsprojects.com/en/1.1.x/quickstart/#static-files
- Regular Expression Documentation:
    - https://docs.python.org/2/library/re.html
- Decode image from post request:
    - https://stackoverflow.com/questions/26070547decoding-base64-from-post-to-use-in-pil
- Changing image mode from RGB to L:
    - https://pillow.readthedocs.io/en/stable/handbook/tutorial.html#converting-between-modes
- Resizing the image to 28x28:
    - https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize
- Changing the threshold of the image:
    - https://www.geeksforgeeks.org/python-pil-image-point-method/
- Reshape and flatten the array:
    - https://numpy.org/devdocs/reference/generated/numpy.reshape.html

## Index.html:
- Draw on a canvas with a mouse, color picker and pen size:
    - http://www.zsoltnagy.eu/javascript-tech-interview-exercise-7-painting-on-an-html5-canvas/
    - https://codepen.io/zsolt555/pen/rpPXOB
- Post Request: 
    - https://api.jquery.com/jQuery.post/

## KerasModel Training:
- High accuracy model:
    - https://nextjournal.com/gkoehler/digit-recognition-with-keras
- Processing of image and label data from MNIST API:
    https://nbviewer.jupyter.org/github/ianmcloughlin/jupyter-teaching-notebooks/blob/master/mnist.ipynb
