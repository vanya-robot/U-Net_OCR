import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random


def get_segmented_img(img, n_classes):
    """
    Loads in the segmented image and create suitable segmentation label.
    """
    seg_labels = np.zeros((512, 512, 1))
    img = cv2.resize(img, (512, 512))
    img = img[:, :, 0]
    cl_list = [0, 24]

    seg_labels[:, :, 0] = (img != 0).astype(int)

    return seg_labels


def visualize(img, seg_img):
    """
    Visualizes image
    """
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Image')
    plt.subplot(1, 2, 2)
    plt.imshow(seg_img, cmap='gray')
    plt.title('Segmented Image')
    plt.show()


def preprocess_img(img):
    img = cv2.resize(img, (512, 512))
    return img


def batch_generator(filelist, n_classes, batch_size):
    while True:
        X = []
        Y = []
        for i in range(batch_size):
            fn = random.choice(filelist)
            img = cv2.imread(f'/content/drive/MyDrive/PageSegData/PageImg/{fn}.JPG', 0)
            ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
            img = cv2.resize(img, (512, 512))
            img = np.expand_dims(img, axis=-1)
            img = img / 255

            seg = cv2.imread(f'/content/drive/MyDrive/PageSegData/PageSeg/{fn}_mask.png', 1)
            seg = get_segmented_img(seg, n_classes)

            X.append(img)
            Y.append(seg)
        yield np.array(X), np.array(Y)


def predict_mask(path_to_img, model):
    img = cv2.imread(path_to_img)
    ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    img = cv2.resize(img, (512, 512))
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    pred = np.squeeze(np.squeeze(pred, axis=0), axis=-1)

    plt.imsave(os.path.splitext(path_to_img)[0] + '_mask.jpg', pred)
    return os.path.splitext(path_to_img)[0] + '_mask.jpg'


def print_contours(path_to_img, path_to_mask):
    img = cv2.imread(path_to_mask)
    cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, img)
    ori_img = cv2.imread(path_to_img)
    ori_img = cv2.resize(ori_img, (512, 512))
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    coordinates = []
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a white rectangle to visualize the bounding rect
        cv2.rectangle(ori_img, (x, y), (x + w, y + h), 255, 1)
        coordinates.append([x, y, (x + w), (y + h)])

    #  cv2.drawContours(img, contours, -1, (255, 255, 0), 1)

    cv2.imwrite(os.path.splitext(path_to_img)[0] + '_cont.jpg', ori_img)
    return os.path.splitext(path_to_img)[0] + '_cont.jpg'


# line_img_array = []


def segment_into_lines(filename, model):
    line_img_array = []
    # Loading the image and performing thresholding on it and then resizing.
    img = cv2.imread(f'{filename}', 0)
    ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    img = cv2.resize(img, (512, 512))
    # Expanding the dimension to account for the batch dimension.
    img = np.expand_dims(img, axis=-1)
    # Expanding dimension along channel axis.
    img = np.expand_dims(img, axis=0)
    # Predict the segmentation mask.
    pred = model.predict(img)
    # Remove the batch and channel dimension for performing the binarization.
    pred = np.squeeze(np.squeeze(pred, axis=0), axis=-1)

    # Performing the binarization of the predicted mask for contour detection.
    coordinates = []
    img = cv2.normalize(src=pred, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, img)
    # Opening the original image to get the original dimension information.
    ori_img = cv2.imread(f'{filename}', 0)

    (H, W) = ori_img.shape[:2]
    (newW, newH) = (512, 512)
    rW = W / float(newW)
    rH = H / float(newH)

    # Contour detection and bouding box generation.
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(ori_img, (int(x*rW), int(y*rH)), (int((x+w)*rW),int((y+h)*rH)), (255,0,0), 1)
        coordinates.append((int(x * rW), int(y * rH), int((x + w) * rW), int((y + h) * rH)))
    # cv2.imwrite("output.jpg",ori_img)

    # Cropping the lines from the original image using the bouding boxes generated above.
    for i in range(len(coordinates) - 1, -1, -1):
        coors = coordinates[i]

        p_img = ori_img[coors[1]:coors[3], coors[0]:coors[2]].copy()

        line_img_array.append(p_img)

    return line_img_array
