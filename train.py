""" This file is implementing the model for NVIDIA CNN for DAVE-2
in order to train the CarND simulator.
"""
import os

import cv2
import pandas as pd
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import cgitb
from keras.models import Sequential, load_model
from keras.layers import Lambda, Convolution2D, Flatten, Dense, Cropping2D
from keras.optimizers import Adam
from sklearn.model_selection import ShuffleSplit

flags = tf.app.flags
FLAGS = flags.FLAGS

# Command line flags
flags.DEFINE_string('drive_data', '', 'Path to folder which contains drive data from the simulator')
flags.DEFINE_string('out', './model.h5', 'Output path from model file.')
flags.DEFINE_float('learning_rate', 1e-3, 'The learning rate of the model.')
flags.DEFINE_integer('epochs', 10, 'The number of epochs used for training the model')
flags.DEFINE_integer('batch_size', 100, 'The number of samples in each batch')
flags.DEFINE_boolean('load_model', True, 'If `True` load model from last runs.')
flags.DEFINE_boolean('trace', False, 'If `True` display extended tracebacks on exception.')

def model():
    """
    Creates the NVIDIA CNN model for DAVE-2.
    The network get as input 66x200 color images
    :return: a Keras model of the network
    """

    def create_resize_func(size):
        """
        Creates a resize function of images using keras tf backend.
        This function is meant to be used in Lambda layer.
        :param size: The target size for the images
        :return: A function that can resize a tensor with images
        """
        def resize(x):
            from keras.backend import tf as ktf
            return ktf.image.resize_images(x, size)

        return resize

    m = Sequential(name='NVIDIA_DAVE_2')
    # Crop the image to 89x270
    m.add(Cropping2D(cropping=((45, 26), (20, 30)), input_shape=(160, 320, 3)))
    # Resize the image
    m.add(Lambda(create_resize_func((66, 200))))
    # Normalization layer
    m.add(Lambda(lambda im: im / 255.0 - 0.5))
    # Convolution layer 1 - out = 31x98x24
    m.add(Convolution2D(24, (5, 5), strides=(2, 2), activation='relu'))
    # Convolution layer 2 - out = 14x47x36
    m.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='relu'))
    # Convolution layer 3 - out = 5x22x48
    m.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='relu'))
    # Convolution layer 5 - out = 3x20x64
    m.add(Convolution2D(64, (3, 3), activation='relu'))
    # Convolution layer 6 - out = 1x18x64
    m.add(Convolution2D(64, (3, 3), activation='relu'))
    # Flatten layer - out = 1x1152
    m.add(Flatten())
    # Fully connected 1
    m.add(Dense(100, activation='relu'))
    # Fully connected 2
    m.add(Dense(50, activation='relu'))
    # Fully connected 3
    m.add(Dense(10, activation='relu'))
    # Out layer
    m.add(Dense(1))

    return m


def load_drive_data():
    """
    Loads the drive data from the simulator
    :return: a generator with
    """
    data_folder = FLAGS.drive_data
    assert os.path.isdir(data_folder), '"{}" is not a folder'.format(data_folder)
    for curr, dirs, files in os.walk(data_folder):
        if 'IMG' in dirs and 'driving_log.csv' in files:
            print('Loading driving data from {}'.format(curr))
            driving_log = pd.read_csv(curr + os.sep + 'driving_log.csv', names=['center', 'left', 'right', 'steering',
                                                                                'throttle', 'break', 'speed'])
            # Correct the data set image paths
            driving_log[['center', 'left', 'right']] = driving_log[['center', 'left', 'right']]\
                .applymap(lambda path: curr + os.sep + 'IMG' + os.sep + path.split('\\')[-1])

            yield driving_log


def extract_features_and_labels(driving_log):
    """
    Creates feature set and labels set from `driving_log`.
    The features set contains the paths to the images and
    the labels set contains the steering angles.
    :param driving_log: The drive data set
    :return: tuple with the features and labels sets.
    """
    # Steering factors for left and right images
    left_factor = 0.2
    right_factor = -0.2

    # Calculate straight drop - the number of straight steering samples to drop
    straight_max_val = 0.85
    n_straight_frames = driving_log.query('{} <= steering <= {}'.format(-straight_max_val, straight_max_val)).count()[1]
    straight_drop = 0.7 * n_straight_frames

    features, labels = [], []
    print('Extracting features & labels from data....', end='')
    for idx, row in driving_log.iterrows():
        steering = row[3]

        # Remove some of the samples of straight steering in order to prevent
        # the model from biasing towards no steering.
        if -straight_max_val < steering < straight_max_val:
            if straight_drop > 0:
                straight_drop -= 1
            else:
                continue

        features += [mpimg.imread(row[0])]  # center image
        features += [mpimg.imread(row[1])]  # left image
        features += [mpimg.imread(row[2])]  # right image

        labels += [steering]
        labels += [steering + left_factor]
        labels += [steering + right_factor]

    print('Done.')
    return np.array(features), np.array(labels, dtype=np.float32)


class DriveDataSplitAndProcess:
    """
     This class is made especially for the driving data from the CarND simulator,
     this class creates training and validation generators, while in training the data
     is augmented by adding flipped images.
    """
    __slots__ = ['_splitter', '_batch_size', '_features',
                 '_labels', '_steps_per_epoch_train',
                 '_steps_per_epoch_valid', '_validation']

    def __init__(self, epochs, batch_size, features, labels, test_size=0.2):
        """
        Initialized this class.
        :param epochs: The number of epochs - indicates how many shuffles there be.
        :param batch_size: The size of each batch sent into the model
        :param features: An array of features to be processed
        :param labels: An array of labels matching the number of features.
        :param test_size: Represent the proportion of the data set to include in the test split.
        """
        assert features.shape[0] == labels.shape[0], 'The number of features must match the number of labels.'

        n_examples = labels.shape[0]
        n_train_examples, n_valid_examples = n_examples * (1-test_size), n_examples * test_size

        self._splitter = ShuffleSplit(n_splits=epochs, test_size=test_size)
        self._batch_size = batch_size
        self._features = features
        self._labels = labels
        self._validation = []

        # Calculate number of steps per epoch for training and validation sets
        self._steps_per_epoch_train = int(n_train_examples / batch_size)
        self._steps_per_epoch_valid = int(n_valid_examples / batch_size)

    def steps_per_epoch(self, validation=False):
        """
        Calculates the number steps per epoch
        :param validation: False for training and True for validation
        :return: The number of steps per epoch
        """
        if validation:
            return self._steps_per_epoch_valid
        else:
            # For training step calculation, multiply n_splits by two in order to include the augmented data batches.
            return self._steps_per_epoch_train * 2

    def train_data_gen(self):
        """
        Creates training generator for keras `fit_generator` method
        :return: a generator
        """
        features, labels, batch_size = self._features, self._labels, self._batch_size

        while True:
            # Empty validation list every shuffle
            self._validation.clear()
            for train_idx_list, valid_idx_list in self._splitter.split(features, labels):
                self._validation += list(valid_idx_list)

                for batch_idx in range(self._steps_per_epoch_train):
                    batch_idx_lst = train_idx_list[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                    X_train, y_train = features[batch_idx_lst], labels[batch_idx_lst]
                    yield X_train, y_train
                    # Return horizontal flipped version of each image
                    yield np.array(list(map(lambda x: cv2.flip(x, 1), X_train))), (-1)*y_train

    def validation_data_gen(self):
        """
        Creates validation generator for keras `fit_generator` method
        :return: a generator
        """
        features, labels, batch_size = self._features, self._labels, self._batch_size
        while True:
            for batch_idx in range(self._steps_per_epoch_valid):
                batch_idx_lst = self._validation[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                yield features[batch_idx_lst], labels[batch_idx_lst]


def main(_):

    # Enable extended backtraces
    if FLAGS.trace:
        cgitb.enable(format='text')

    # Load or initalize the model
    model_loaded = False
    if FLAGS.load_model:
        model_path = FLAGS.out
        if os.path.exists(model_path):
            print('Loading model from {}'.format(model_path))
            m = load_model(model_path)
            model_loaded = True
        else:
            print("Couldn't find the model file {}".format(model_path))
    if not model_loaded:
        print('Creating model with learning rate of {}'.format(FLAGS.learning_rate))
        m = model()
        adam = Adam(lr=FLAGS.learning_rate)
        m.compile(optimizer=adam, loss='mse', metrics=['accuracy'])

    # Traing the model
    for data in load_drive_data():
        features, labels = extract_features_and_labels(data)
        processor = DriveDataSplitAndProcess(FLAGS.epochs, FLAGS.batch_size, features, labels)

        m.fit_generator(processor.train_data_gen(), processor.steps_per_epoch(), epochs=FLAGS.epochs,
                        validation_data=processor.validation_data_gen(),
                        validation_steps=processor.steps_per_epoch(validation=True))
        print('Writing model to {}'.format(FLAGS.out))
        m.save(FLAGS.out)


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run(main)
