import heapq
from scipy.misc import imread
import joblib
from keras.engine.saving import load_model
from scipy.misc import imresize

from helpers import add_noise

HEIGHT = 28
WIDTH = 28

model = load_model("models/model.h5")
lb = joblib.load("models/lb.sav")
print(lb.classes_)

while 1:
    try:
        print("\nРаположение изображения:")
        inp = input()

        x = imread(inp, mode='L')
        x = imresize(x, (28, 28))

        x = x.reshape(1, 28, 28, 1)/255

        x = add_noise(x)

        pred = model.predict(x)[0]

        map_output = {}
        for i in range(len(pred)):
            print(lb.classes_[i], "\t",pred[i])

        max_pred = heapq.nlargest(1, range(len(pred)), pred.take)[0]
        print("Число:", lb.classes_[max_pred])

    except Exception as e:
        print(repr(e))