import numpy as np
from unet_lines import segment_into_lines
from pipeline_utils import segment_into_words
from crnn import recognize_words
from unet import unet

# Open image and segment into lines
directory = './example/'
filename = 'example.jpeg'

# Load line segmentation U-Net model
line_unet = unet(pretrained_weights='unet_lines.h5')

line_img_array = segment_into_lines(directory + filename, line_unet)
del line_unet

# Creating lists to store the line indexes,words list.
full_index_indicator = []
all_words_list = []
# Variable to count the total no of lines in page.
len_line_arr = 0

# Load word segmentation U-net model
word_unet = unet(pretrained_weights='unet_words.h5')

# Segment the lines into words and store as arrays.
for idx, im in enumerate(line_img_array):
    line_indicator, word_array = segment_into_words(im, idx, word_unet)
    for k in range(len(word_array)):
        full_index_indicator.append(line_indicator[k])
        all_words_list.append(word_array[k])
    len_line_arr += 1

all_words_list = np.array(all_words_list)
del word_unet

# Perform the recognition on list of list of words.
recognize_words(full_index_indicator, all_words_list, len_line_arr, directory, filename)
