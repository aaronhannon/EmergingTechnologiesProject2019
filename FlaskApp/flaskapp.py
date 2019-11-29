#Imports necessary to run the server
from flask import Flask, json, jsonify, render_template, request
import numpy as np
from kerasModel import prediction
import sys
from PIL import Image
import re
import base64
from keras.models import load_model
import gzip
import io
import cv2
from io import BytesIO

app = Flask(__name__)

#GET request: When 127.0.0.1:5000 is entered into the URL this method returns the index.html file.
@app.route("/")
def home():
    return app.send_static_file('index.html')

#POST request: When the user sends the image as a post request this method is run.
@app.route('/', methods=['POST'])
def getImage():
    
    #Retrieving Base64 image data
    image_b64 = request.values['imageBase64']
    #re.sub replaces "^data:image/.+;base64," with an empty string so all you are left with is the image data
    image_data = re.sub('^data:image/.+;base64,', '', image_b64)

    #Decodes the image from Base64
    decode_Image = base64.b64decode(image_data)

    #Converting the decoded image to bytes and grayscaling it using PIL (Python Image Library)
    img = Image.open(BytesIO(decode_Image)).convert("L")
    
    #Resizes the image to the required size of 28x28 using Bilinear mode. I used this mode because during testing it seemed
    #to be the most fast and accurate.
    img = img.resize((28, 28), Image.BILINEAR)
    
    #The MNIST images that the keras model was trained with are white numbers with a black background.
    # .convert('L') converts it to grayscale which is not exactly what I want so I convert the grays to white using this lambda expression
    threshold = 0  
    img = img.point(lambda p: p > threshold and 255)  
    
    #The image then must be flattened and reshaped into a single array with 784 uint8's and then divided by 255 to get 1s or < 1
    grayArray = np.ndarray.flatten(np.array(img)).reshape(1,784).astype(np.uint8) / 255


    #Prints out the image with 0s and .s for testing.
    counter = 0
    #Print Contents
    for i in grayArray[0]:
        #print(i,end=" ")
        if i == 1:
           print(".",end="")
        else:
           print("0",end="")
        counter +=1

        if counter == 28:
            print("\n")
            counter = 0
    
    #This calls the prediction method in kerasModel.py which returns the argmax of the array of predictions
    labelPredict = prediction(grayArray)

    #Prints Prediction
    print("PREDICTION: ")
    print(labelPredict)

    #Sends prediction back to the user.
    return str(labelPredict)


    
if __name__ == "__main__":
    app.run(debug = True, threaded = False)