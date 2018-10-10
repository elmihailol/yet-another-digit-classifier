import joblib
from keras import Sequential
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, np, GaussianDropout, Activation
from keras_preprocessing.image import ImageDataGenerator

from helpers import add_noise

batch_size = 512
steps_per_epoch = 256
num_classes = 10
epochs = 12
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train = np.clip(x_train, 0., 0.3)
x_test = np.clip(x_test, 0., 0.3)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

lb = joblib.load("lb.sav")
y_train = lb.transform(y_train)
y_test = lb.transform(y_test)

x_train = x_train.reshape((-1, img_rows, img_cols, 1))
x_test = x_test.reshape((-1, img_rows, img_cols, 1))

x_train = add_noise(x_train)

input_shape = (img_rows, img_cols, 1)

model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=True,
    rotation_range=90,
    zoom_range=0.4,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False,
    vertical_flip=False)

max_acc = 0

h = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        epochs=epochs, steps_per_epoch=steps_per_epoch,
                        verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
joblib.dump(lb, "models/lb.sav")
model.save("models/model.h5")
print('Test loss:', score[0])
print('Test accuracy:', score[1])
