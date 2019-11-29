import gzip
import numpy as np
import keras as kr
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.models import load_model
import sklearn.preprocessing as pre
import matplotlib.pyplot as plt

def build():

    #REFERENCE TO KERAS MODEL
    #https://nextjournal.com/gkoehler/digit-recognition-with-keras
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))                            
    model.add(Dropout(0.2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    with gzip.open('mnist_data/train-images-idx3-ubyte.gz', 'rb') as f:
        train_img = f.read()

    with gzip.open('mnist_data/train-labels-idx1-ubyte.gz', 'rb') as f:
        train_lbl = f.read()
        
    train_img = np.array(list(train_img[16:])).reshape(60000, 28, 28).astype(np.uint8) / 255.0
    train_lbl =  np.array(list(train_lbl[ 8:])).astype(np.uint8)

    inputs = train_img.reshape(60000, 784)

    encoder = pre.LabelBinarizer()
    encoder.fit(train_lbl)
    outputs = encoder.transform(train_lbl)

    model.fit(inputs, outputs,batch_size=128, epochs=20,verbose=2)
    model.save("model.h5")

    return model

def prediction(image):
    #If the model has already been trained then it loads it in otherwise it trains it from the start
    try:
        model = load_model("model.h5")
    except:
        print("LOADING FAILED....")
        model = build()

    #Makes prediction
    labelPredict = model.predict(image)
    
    #returns the index of the largest number in the array which is the prediction
    return labelPredict.argmax()


