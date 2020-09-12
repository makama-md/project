
import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_data(path):
    path = 'new'

    train_x, train_y = glob(os.path.join(path, "train/images/*")), glob(os.path.join(path, "train/masks/*"))
    valid_x, valid_y = glob(os.path.join(path, "val/images/*")), glob(os.path.join(path, "val/masks/*"))
    test_x, test_y  = glob(os.path.join(path, "test/images/*")), glob(os.path.join(path, "test/masks/*"))

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # x = cv2.resize(x, (128, 128))
    x = x/255.0
    x = np.expand_dims(x, axis=-1)
    # print(f"the shape of x is {x.shape}")
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # x = cv2.resize(x, (128, 128))
    x = x/255.0
    x = np.expand_dims(x, axis=-1)
    # print(f"the shape of x is {x.shape}")
    return x


def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([128, 128, 1])
    y.set_shape([128, 128, 1])
    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset
