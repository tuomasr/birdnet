import os

import numpy as np

from keras.models import Sequential
from keras.preprocessing import image as image_utils
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization


# modify if necessary
DIM_ORDERING = 'tf'
DATA_ROOT_DIR = '../CUB_200_2011/'


def read_images(image_target_size):
    image_root_dir = DATA_ROOT_DIR + 'images/'
    subdirs = os.listdir(image_root_dir)

    images = []

    for subdir in sorted(subdirs):
        files = os.listdir(os.path.join(image_root_dir, subdir))

        for file in sorted(files):
            image_path = os.path.join(image_root_dir, subdir, file)

            image = image_utils.load_img(image_path, target_size=image_target_size)
            image = image_utils.img_to_array(image, dim_ordering=DIM_ORDERING)
            image /= 255 # scale the RGB values to [0, 1]
            images.append(image)

    images = np.stack(images, axis=0)
    
    return images


def read_class_labels():
    with open(DATA_ROOT_DIR + 'image_class_labels.txt', 'r') as f:
        lines = f.read().splitlines()
        # decrement by one to obtain labels starting from zero
        y = [int(line.split(' ')[1])-1 for line in lines]

    return np.array(y)


def train_test_split(X, y, split=0.1):
    num_train_examples = int(len(X)*(1-split))
    num_output_classes = len(set(y))

    # data is ordered so shuffle it
    shuffled_indices = np.random.permutation(len(X))
    X = X[shuffled_indices, :, :, :]
    y = y[shuffled_indices]

    X_train = X[:num_train_examples, :, :, :]
    y_train = y[:num_train_examples]

    X_test = X[num_train_examples:, :, :, :]
    y_test = y[num_train_examples:]

    # convert class integers to one hot vectors
    Y_train = np_utils.to_categorical(y_train, num_output_classes)
    Y_test = np_utils.to_categorical(y_test, num_output_classes)

    print('X_train', X_train.shape)
    print('Y_train', Y_train.shape)
    print('X_test', X_test.shape)
    print('Y_test', Y_test.shape)

    return (X_train, Y_train), (X_test, Y_test)

def main():
    # read input and output data
    image_target_size = (80, 80)
    image_width = image_target_size[0]
    image_height = image_target_size[1]

    X = read_images(image_target_size)
    y = read_class_labels()

    assert len(X) == len(y)

    # split data to train and test sets
    (X_train, Y_train), (X_test, Y_test) = train_test_split(X, y)

    num_output_classes = Y_train.shape[1]

    batch_size = 128
    num_epoch = 60

    model = Sequential()

    # convolution 1
    model.add(Convolution2D(64, 3, 3, border_mode='valid',
                            input_shape=(image_width, image_height, 3),
                            dim_ordering=DIM_ORDERING))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # convolution 2
    model.add(Convolution2D(32, 3, 3, border_mode='valid',
                            dim_ordering=DIM_ORDERING))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering=DIM_ORDERING))

    # convolution 3
    model.add(Convolution2D(32, 3, 3, border_mode='valid',
                            dim_ordering=DIM_ORDERING))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering=DIM_ORDERING))
    model.add(Dropout(0.2))

    # flattening
    model.add(Flatten())

    # fully connected
    model.add(Dense(600))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # output layer
    model.add(Dense(num_output_classes))
    model.add(Activation('softmax'))

    # fit
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=num_epoch,
              validation_data=(X_test, Y_test))

    # evaluate
    accuracy = model.evaluate(X_test, Y_test)
    print('Model performance:', accuracy)

main()
