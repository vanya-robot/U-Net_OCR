import numpy as np
import string
from PIL import Image, ImageFont, ImageDraw
import argparse
import random
import os
import imgaug.augmenters as iaa
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', type=int)
parser.add_argument('--word_type', default='lowercase')
args = parser.parse_args()

global gray_back
kernel = np.ones((2, 2), np.uint8)
kernel2 = np.ones((1, 1), np.uint8)
punclist = ',.?:;'

# Character sets to choose from.
smallletters = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789"
capitalletters = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789"
digits = string.digits

# Base backgound.
backfilelist = os.listdir('./background/')
backgroud_list = []

for bn in backfilelist:
    fileloc = './background/' + bn
    backgroud_list.append(Image.fromarray(cv2.imread(fileloc, 0)))

# Different fonts to be used.
fonts_list = os.listdir('./fonts/')
fonts_list = ['./fonts/' + f for f in fonts_list]

# Lengths of the words.
word_lengths = []
for l in range(1, 21):
    word_lengths.append(l)

# Font size.
font_size = []
for l in range(10, 30):
    font_size.append(l)

file_counter = 0


def random_brightness(img):
    img = np.array(img)
    brightness = iaa.Multiply((0.2, 1.2))
    img = brightness.augment_image(img)
    return img


def dilation(img):
    img = np.array(img)
    img = cv2.dilate(img, kernel2, iterations=1)
    return img


def erosion(img):
    img = np.array(img)
    img = cv2.erode(img, kernel, iterations=1)
    return img


def blur(img):
    img = np.array(img)
    img = cv2.blur(img, ksize=(3, 3))
    return img


def fuse_gray(img):
    img = np.array(img)
    ht, wt = img.shape[0], img.shape[1]
    gray_back = cv2.imread('gray_back.jpg', 0)
    gray_back = cv2.resize(gray_back, (wt, ht))

    blended = cv2.addWeighted(src1=img, alpha=0.8, src2=gray_back, beta=0.4, gamma=10)
    return blended


def random_transformation(img):
    if np.random.rand() < 0.5:
        img = fuse_gray(img)
    elif np.random.rand() < 0.5:
        img = random_brightness(img)
    elif np.random.rand() < 0.5:
        img = dilation(img)
    elif np.random.rand() < 0.5:
        img = erosion(img)

    else:
        img = np.array(img)
    return Image.fromarray(img)


file = open('annotation.txt', 'a+')

file_counter = 0

for _ in range(args.n_samples):

    back_c = random.choice(backgroud_list).copy()
    start_cap = random.choice(capitalletters)
    filename = ''.join([random.choice(smallletters) for c in range(random.choice([5, 6, 7, 8, 9, 10, 11]))])
    font = ImageFont.truetype(random.choice(fonts_list), size=random.choice(font_size))
    if args.word_type == 'lowercase':
        word = ''.join([random.choice(smallletters) for b in range(random.choice(word_lengths))])
    elif args.word_type == 'uppercase':
        word = ''.join([random.choice(capitalletters) for b in range(random.choice(word_lengths))])
    elif args.word_type == 'firstcapital':
        word = ''.join([random.choice(smallletters) for b in range(random.choice(word_lengths) - 1)])
        word = start_cap + word
    elif args.word_type == 'digits':
        word = ''.join([random.choice(digits) for b in range(random.choice(word_lengths))])
    elif args.word_type == 'punctuation':
        word = ''.join([random.choice(smallletters) for b in range(random.choice(word_lengths))])
        word = word + str(random.choice(punclist))
    else:
        raise Exception("Invalid word choice.")
    w, h = font.getsize(word)[0], font.getsize(word)[1]
    back_c = back_c.resize((w + 5, h + 5))
    draw = ImageDraw.Draw(back_c)
    draw.text((0, 0), text=word, font=font, fill='rgb(0,0,0)')
    back_c = random_transformation(back_c)
    back_c.save(f'./images/{file_counter}.jpg')
    file.writelines(str(file_counter) + '.jpg' + ',' + word + '\n')
    file_counter += 1
