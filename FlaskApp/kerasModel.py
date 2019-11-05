import gzip
import numpy as np
import keras as kr
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.models import load_model
import sklearn.preprocessing as pre
import matplotlib.pyplot as plt


with gzip.open('mnist_data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    file_content = f.read()

with gzip.open('mnist_data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    label_content = f.read()

# print(int.from_bytes(file_content[0:4], byteorder='big'))
# print(label_content[8])
            
#first 16 bytes are headers, 28 x 28 = 784
#_prev_image = 16
#_next_image = 800
#label_index = 8
# for i in range(20):

#     l = file_content[_prev_image:_next_image]
#     _prev_image = _next_image
#     _next_image = _next_image + 784

#     image = ~np.array(list(l)).reshape(28,28).astype(np.uint8)
#     print("=========")
#     print("LABLE = " + "{}".format(label_content[label_index]))
#     print("=========")
#     label_index += 1

#     for row in image:
#         for elem in row:
#             if elem > 127:
#                 print("0", end=' ')
#             else:
#                 print(".",end=' ')
#         print()

def build():
    #Start a neural network, building it by layers.
    #model = kr.models.Sequential()

    # # Add a hidden layer with 1000 neurons and an input layer with 784.
    # model.add(kr.layers.Dense(units=600, activation='linear', input_dim=784))
    # model.add(kr.layers.Dense(units=400, activation='relu'))
    # # Add a three neuron output layer.
    # model.add(kr.layers.Dense(units=10, activation='softmax'))

    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))                            
    model.add(Dropout(0.2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    # Build the graph.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    with gzip.open('mnist_data/train-images-idx3-ubyte.gz', 'rb') as f:
        train_img = f.read()

    with gzip.open('mnist_data/train-labels-idx1-ubyte.gz', 'rb') as f:
        train_lbl = f.read()
        
    train_img = ~np.array(list(train_img[16:])).reshape(60000, 28, 28).astype(np.uint8) / 255.0
    train_lbl =  np.array(list(train_lbl[ 8:])).astype(np.uint8)

    inputs = train_img.reshape(60000, 784)

    encoder = pre.LabelBinarizer()
    encoder.fit(train_lbl)
    outputs = encoder.transform(train_lbl)

    model.fit(inputs, outputs,batch_size=128, epochs=20,verbose=2)
    #model.fit(inputs, outputs, epochs=4, batch_size=100)
    model.save("model.h5")

    return model

#print(type(encoder))

#print(train_lbl[0], outputs[0])

# for i in range(10):
#     print(i, encoder.transform([i]))

def prediction(image):
    try:
        #print("LOADING.....")
        model = load_model("model.h5")
    except:
        print("LOADING FAILED....")
        model = build()


    
    labelPredict = model.predict(image)
    
    
    return labelPredict.argmax()


