from flask import Flask, json, jsonify, render_template, request
import numpy as np
from kerasModel import prediction
import sys
import matplotlib.pyplot as plt
from PIL import Image
import re
import base64
from keras.models import load_model
import gzip
import io
import cv2
from io import StringIO
from io import BytesIO

app = Flask(__name__)

@app.route("/")
def home():
    return app.send_static_file('index.html')

@app.route('/', methods=['POST'])
def getImage():
    
    image_b64 = request.values['imageBase64']
    
    image_data = re.sub('^data:image/.+;base64,', '', image_b64)

    decode_Image = base64.b64decode(image_data)

    img = Image.open(BytesIO(decode_Image))
    img = img.save("img.png")
    imge = cv2.imread("img.png")
    gray = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)
    
    grayArray = ~np.array(list(gray)).reshape(1, 784).astype(np.uint8) /255


    print(grayArray)
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
    
    

    

    labelPredict = prediction(grayArray)


    print("PREDICTION: ")
    print(labelPredict)
    #print(labelPredict.argmax())

    with gzip.open('mnist_data/t10k-images-idx3-ubyte.gz', 'rb') as f:
        test_img = f.read()

    with gzip.open('mnist_data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        test_lbl = f.read()



    test_img = ~np.array(list(test_img[16:])).reshape(10000, 784).astype(np.uint8) /255
    
    # for i in test_img[6:7][0]:
    # #print(i,end=" ")
    #     if i == 1:
    #         print(".",end="")
    #     else:
    #         print("0",end="")
    #     counter +=1

    #     if counter == 28:
    #         print("\n")
    #         counter = 0

    #print(test_img[5:6])

    output = prediction(test_img[5:6])

    print("TEST PREDICTION")
    print(output)
    #print(output.argmax())

    return app.send_static_file('index.html')
    
if __name__ == "__main__":
    app.run(debug = True, threaded = False)