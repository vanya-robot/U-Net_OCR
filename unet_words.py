import numpy as np
import cv2
import random
import math
from unet_lines import get_segmented_img


def roundup(x):
    return int(math.ceil(x / 10.0)) * 10


def pad_img(img):
    old_h, old_w = img.shape[0], img.shape[1]

    # Pad the height.

    # If height is less than 512 then pad to 512
    if old_h < 512:
        to_pad = np.ones((512 - old_h, old_w)) * 255
        img = np.concatenate((img, to_pad))
        new_height = 512
    else:
        # If height >512 then pad to nearest 10.
        to_pad = np.ones((roundup(old_h) - old_h, old_w)) * 255
        img = np.concatenate((img, to_pad))
        new_height = roundup(old_h)

    # Pad the width.
    if old_w < 512:
        to_pad = np.ones((new_height, 512 - old_w)) * 255
        img = np.concatenate((img, to_pad), axis=1)
        new_width = 512
    else:
        to_pad = np.ones((new_height, roundup(old_w) - old_w)) * 255
        img = np.concatenate((img, to_pad), axis=1)
        new_width = roundup(old_w) - old_w
    return img


def pad_seg(img):
    old_h, old_w = img.shape[0], img.shape[1]

    # Pad the height.

    # If height is less than 512 then pad to 512
    if old_h < 512:
        to_pad = np.zeros((512 - old_h, old_w))
        img = np.concatenate((img, to_pad))
        new_height = 512
    else:
        # If height >512 then pad to nearest 10.
        to_pad = np.zeros((roundup(old_h) - old_h, old_w))
        img = np.concatenate((img, to_pad))
        new_height = roundup(old_h)

    # Pad the width.
    if old_w < 512:
        to_pad = np.zeros((new_height, 512 - old_w))
        img = np.concatenate((img, to_pad), axis=1)
        new_width = 512
    else:
        to_pad = np.zeros((new_height, roundup(old_w) - old_w))
        img = np.concatenate((img, to_pad), axis=1)
        new_width = roundup(old_w) - old_w
    return img


def batch_generator(filelist, n_classes, batch_size):
    while True:
        X = []
        Y = []
        for i in range(batch_size):
            fn = random.choice(filelist)
            img = cv2.imread(f'/content/drive/MyDrive/Dataset_words/img/{fn}.jpg', 0)
            img = pad_img(img)
            ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

            img = cv2.resize(img, (512, 512))
            img = np.expand_dims(img, axis=-1)
            # img = np.stack((img,)*3, axis=-1)
            img = img / 255

            seg = cv2.imread(f'/content/drive/MyDrive/Dataset_words/mask/{fn}_mask.png', 0)
            seg = pad_seg(seg)
            seg = cv2.resize(seg, (512, 512))
            seg = np.stack((seg,) * 3, axis=-1)
            seg = get_segmented_img(seg, n_classes)

            X.append(img)
            Y.append(seg)
        yield np.array(X), np.array(Y)


