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
from io import StringIO
from io import BytesIO

app = Flask(__name__)

@app.route("/")
def home():
    return app.send_static_file('index.html')

@app.route('/', methods=['POST'])
def getImage():
    
    image_b64 = request.values['imageBase64']
    
    print(image_b64)
    
    image_data = re.sub('^data:image/.+;base64,', '', image_b64)

    print(image_data)

    decode_Image = base64.b64decode(image_data)

    decode_Image = ~np.array(list(decode_Image)).astype(np.uint8) / 255.0
    decode_Image.resize(1,784)

    # #print(test_img)

    # img = Image.open(BytesIO(decode_Image))
    
    # #img.show()

    # image_np = np.array(img)

    # pixels = np.resize(image_np, (1, 784)).astype(np.uint8) / 255

    
    model = load_model("model.h5")

    labelPredict = model.predict(decode_Image)

    print(labelPredict)
    print(labelPredict.argmax())

    with gzip.open('mnist_data/t10k-images-idx3-ubyte.gz', 'rb') as f:
        test_img = f.read()

    with gzip.open('mnist_data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        test_lbl = f.read()



    test_img = ~np.array(list(test_img[16:])).reshape(10000, 784).astype(np.uint8) / 255.0
    

    output = model.predict(test_img[5:6])

    print(output)
    print(output.argmax())

    # for row in pixels:
    #     for elem in row:
    #         if elem > 127:
    #             print("0", end=' ')
    #         else:
    #             print(".",end=' ')
    #     print()


    #print(pixels)

    # pred = prediction(pixels)

    # print("Prediction: ")
    # print(pred)

    #test_img = ~np.array(list(decode_Image)).reshape(10000, 784).astype(np.uint8) / 255.0
    
    
    
    #data = request.get_data()

    #data.show()
    #dimensions = (4, 784)
    #print(data)
    #print(data[22:])

    #print(type(data))
    #print(data)
    #print(data[38:])
    #pixels = np.asarray(data[38:], dtype='uint8')

    #decode_Image = base64.b64decode(data[38:])

    #decode_Image.show()

    #print(decode_Image)

    #print(type(decode_Image))

    #print(decode_Image)

    # img = Image.open(BytesIO(decode_Image))
    
    # img.show()

    # print(img)

    # test_img = ~np.array(list(data[38:])).reshape(1, 784).astype(np.uint8) / 255.0

    # print(test_img)

    # image is (28, 28)
    #img = img.resize(dimensions, Image.ANTIALIAS)
    # # pixels.shape == (28, 28, 4)
    # pixels = np.asarray(img, dtype='uint8')

    # for i in pixels:
    #     print(i)
    # # force (28, 28)
    #pixels = np.resize(pixels, (1, 784))

    

    #pred = prediction(pixels)

    # print("Prediction: ")
    # print(pred)

    return app.send_static_file('index.html')
    
if __name__ == "__main__":
    app.run(debug=True)