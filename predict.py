
import os
import numpy as np
import cv2
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import f1_score
from tensorflow.keras.utils import CustomObjectScope
from tqdm import tqdm
from data import load_data, tf_dataset
from train import f1, iou

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x/255.0
    # x = np.expand_dims(x, axis=-1)

    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = np.expand_dims(x, axis=-1)
    return x

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

if __name__ == "__main__":
    # Dataset
    path = "new"
    batch_size = 32
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

    test_dataset = tf_dataset(test_x, test_y, batch=batch_size)

    test_steps = (len(test_x)//batch_size)
    if len(test_x) % batch_size != 0:
        test_steps += 1

    with CustomObjectScope({'iou': iou}, {'f1_score': f1}):
        model = tf.keras.models.load_model("files/model.h5")

    model.evaluate(test_dataset, steps=test_steps)

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        x = read_image(x)
        y = read_mask(y)
        y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5
        if len(x.shape)==3:
          h, w, _ = x.shape
          x = x.reshape(h,w)
        elif len(x.shape)==2:
          h, w = x.shape
        else:
          raise NotImplementedError("The shape of x is wierd, should be either 2 channels or 3 got"+str(len(x.shape)))
        white_line = np.ones((h, 10)) * 255.0

        all_images = [
            x * 255.0, white_line,
            mask_parse(y)[:,:,1], white_line,
            mask_parse(y_pred)[:,:,1] * 255.0
        ]
        image = np.concatenate(all_images, axis=1)
        cv2.imwrite(f"results/{i}.png", image)
