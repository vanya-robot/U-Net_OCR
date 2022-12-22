import numpy as np
from PIL import Image
import cv2

image = cv2.imread('testingimage.png')
#img = np.array(image)
ori_img = image
ret, img = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
img = cv2.resize(img, (512, 512))
# Expanding the dimension to account for the batch dimension.
img = np.expand_dims(img, axis=-1)

print(img.shape())
