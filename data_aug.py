import os
import random
from shutil import copyfile
import random
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
# from utils import create_dir
from data import load_data

from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    RandomRotate90,
    RandomScale
)

def augment_data(images, masks, save_path, augment=True):
    """ Performing data augmentation. """
    size = (128, 128)

    for image, mask in tqdm(zip(images, masks), total=len(images)):
        image_name = image.split("/")[-1].split(".")[0]
        mask_name = mask.split("/")[-1].split(".")[0]

        x = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        y = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        h, w = x.shape

        if augment == True:
            aug = HorizontalFlip(p=1)
            augmented = aug(image=x, mask=y)
            x1 = augmented['image']
            y1 = augmented['mask']

            aug = VerticalFlip(p=1)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            aug = RandomRotate90(p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            # aug = RandomScale(p=1.0, scale_limit=(0.5, 2.0))
            # augmented = aug(image=x, mask=y)
            # x4 = augmented['image']
            # y4 = augmented['mask']

            images = [x, x1, x2, x3]
            masks  = [y, y1, y2, y3]

        else:
            images = [x]
            masks  = [y]

        idx = 0
        for i, m in zip(images, masks):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            tmp_image_name = f"{image_name}_{idx}.png"
            tmp_mask_name  = f"{mask_name}_{idx}.png"

            image_path = os.path.join(save_path, "image/", tmp_image_name)
            mask_path = os.path.join(save_path, "mask/", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            idx += 1

def main():
    np.random.seed(42)
    path = 'new/'
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

    print("Train: ", len(train_x))
    print("Valid: ", len(valid_x))
    print("Test: ", len(test_x))


    # create_dir("../new_data/train/image/")
    # create_dir("../new_data/train/mask/")
    # create_dir("../new_data/valid/image/")
    # create_dir("../new_data/valid/mask/")
    # create_dir("../new_data/test/image/")
    # create_dir("../new_data/test/mask/")


    augment_data(train_x, train_y, "../new/train/", augment=True)
    augment_data(valid_x, valid_y, "../new/valid/", augment=False)
    augment_data(test_x, test_y, "../new/test/", augment=False)


if __name__ == "__main__":
    main()
