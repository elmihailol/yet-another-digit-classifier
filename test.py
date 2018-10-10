import os
from pprint import pprint

import cv2
import joblib
import numpy
from keras.datasets import mnist
from keras.engine.saving import load_model
from scipy.misc import imresize

HEIGHT = 28
WIDTH = 28

def add_noise(x, noise_factor=0.1):
    x = x + numpy.random.randn(*x.shape) * noise_factor
    x = x.clip(0., 5.)
    return x

def get_im(path):
    # Load as grayscale
    img = cv2.imread(path, 0)
    # Reduce size
    resized = cv2.resize(img, (HEIGHT, WIDTH))
    return resized


(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = load_model("models/model.h5")
lb = joblib.load("models/lb.sav")
print(lb.classes_)
import matplotlib.pyplot as plt
kek = 1

from scipy.misc import imread, imresize,imshow
while 1:
    inp = "images/"+input()
    dataX = []
    binary_image = None

    x = imread(inp, mode='L')
    #x = numpy.invert(x)
    x = imresize(x, (28, 28))
    x = x.reshape(1, 28, 28, 1)/255
    x = add_noise(x)
    xx = x.tolist()
    pred = list(model.predict(x)[0])
    map_output = {}
    for i in range(len(pred)):
        print(pred[i],  lb.classes_[i])
    pprint(map_output)