""" This file is implementing the model for NVIDIA CNN for DAVE-2
in order to train the CarND simulator.
"""
import os

import cv2
import pandas as pd
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Lambda, Convolution2D, Flatten, Dense, Cropping2D
from keras.backend import tf as ktf
from sklearn.model_selection import StratifiedShuffleSplit

flags = tf.app.flags
FLAGS = flags.FLAGS

# Command line flags
flags.DEFINE_string('drive_data', '', 'Path to folder which contains drive data from the simulator')
flags.DEFINE_string('out', './model.h5', 'Output path from model file.')
flags.DEFINE_integer('epochs', 10, 'The number of epochs used for training the model')
flags.DEFINE_integer('batch_size', 100, 'The number of samples in each batch')

def model():
    """
    Creates the NVIDIA CNN model for DAVE-2.
    The network get as input 66x200 color images
    :return: a Keras model of the network
    """

    m = Sequential(name='NVIDIA_DAVE_2')
    # Crop the image to 89x270
    m.add(Cropping2D(cropping=((45, 26), (20, 30)), input_shape=(160, 320, 3)))
    # Resize the image
    m.add(Lambda(lambda im: ktf.image.resize_images(im, (66, 200))))
    # Normalization layer
    m.add(Lambda(lambda im: im / 255.0 - 0.5))
    # Convolution layer 1 - out = 31x98x24
    m.add(Convolution2D(24, (5, 5), strides=(2, 2)))
    # Convolution layer 2 - out = 14x47x36
    m.add(Convolution2D(36, (5, 5), strides=(2, 2)))
    # Convolution layer 3 - out = 5x22x48
    m.add(Convolution2D(48, (5, 5), strides=(2, 2)))
    # Convolution layer 5 - out = 3x20x64
    m.add(Convolution2D(64, (3, 3)))
    # Convolution layer 6 - out = 1x18x64
    m.add(Convolution2D(64, (3, 3)))
    # Flatten layer - out = 1x1152
    m.add(Flatten())
    # Fully connected 1
    m.add(Dense(100))
    # Fully connected 2
    m.add(Dense(50))
    # Fully connected 3
    m.add(Dense(10))
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
                .applymap(lambda path: curr + os.sep + 'IMG' + os.sep + os.path.basename(path))

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
    straight_max_val = 0.085
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

    _validation = []

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

        self._splitter = StratifiedShuffleSplit(n_splits=epochs, test_size=test_size)
        self._batch_size = batch_size
        self._features = features
        self._labels = labels

        # Calculate number of steps per epoch for training and validation sets
        # For training step calculation, multiply n_splits by two in order to include the augmented data batches.
        self._steps_per_epoch_train = int(n_train_examples / batch_size) * 2
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

            return self._steps_per_epoch_train

    def train_data_gen(self):
        """
        Creates training generator for keras `fit_generator` method
        :return: a generator
        """
        features, labels, batch_size = self._features, self._labels, self._batch_size

        # Use np.digitize in order to arrange the labels in smaller groups for the splitter, otherwise the `split`
        # function will return an error stating that there are classes with only one value.
        for train_idx_list, valid_idx_list in self._splitter.split(features, np.digitize(labels,
                                                                                         bins=np.arange(-1, 1, 0.2))):
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
        for batch_idx in range(self._steps_per_epoch_valid):
            batch_idx_lst = self._validation[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            yield features[batch_idx_lst], labels[batch_idx_lst]


def main(_):
    m = model()

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
