
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout, MaxPooling2D, Conv2D, BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

mean = X_train.mean()
std = X_train.std()

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


model = Sequential()

model.add(Conv2D(32, (3,3), padding='same', input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(32, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()

#model.load_weights('cifar10.h5')

tbcallback = TensorBoard(
    log_dir='./logs',
    histogram_freq=0, write_graph=True, write_images=False)
    
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True)

datagen.fit(X_train)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit_generator(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=50, callbacks=[tbcallback],
    validation_data=(X_test, y_test))


model.save_weights('cifar10.h5', overwrite=True)