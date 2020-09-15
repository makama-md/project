import os
from shutil import copyfile
import random

root_dir = "CVC-612/"
images = 'images/'
masks = 'masks/'

count = 0
total = 15096


Real_val = 0.3

TRAIN_PER = 0.8
VAL_PER = 0.1
TEST_PER = 0.1


train_dir = 'new/train/'
val_dir = 'new/val/'
test_dir = 'new/test/'

shuffle = False

for subdir, dirs, files in os.walk(root_dir+images):
    if shuffle:
        random.shuffle(files)
    for file in sorted(files):
        if '.png' in file:
            count += 1

            image_folder = root_dir+images
            mask_folder = root_dir+masks

            if count <= total*TRAIN_PER:
                if not os.path.exists(train_dir+images) or not os.path.exists(train_dir+masks) :
                    os.makedirs(train_dir+images)
                    os.makedirs(train_dir+masks)

                if count <= total*Real_val:
                    copyfile(image_folder+file, train_dir+images+file)
                    copyfile(mask_folder+file, train_dir+masks+file)

            elif total*TRAIN_PER < count <= ((total*TRAIN_PER)+(total*VAL_PER)):
                if not os.path.exists(val_dir + images) or not os.path.exists(val_dir + masks):
                    os.makedirs(val_dir + images)
                    os.makedirs(val_dir + masks)


                copyfile(image_folder+file, val_dir+images+file)
                copyfile(mask_folder+file, val_dir+masks+file)

            else:
                if not os.path.exists(test_dir + images) or not os.path.exists(test_dir + masks):
                    os.makedirs(test_dir + images)
                    os.makedirs(test_dir + masks)


                copyfile(image_folder+file, test_dir+images+file)
                copyfile(mask_folder+file, test_dir+masks+file)

