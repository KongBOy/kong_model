"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import pprint
import scipy.misc
import numpy as np
import copy
try:
    _imread = scipy.misc.imread
    print("here1")
except AttributeError:
    from imageio import imread as _imread
    print("here2")
pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for cyclegan
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image

#def load_test_data(image_path, fine_size_h=1068, fine_size_w=800):
def load_test_data(image_path, fine_size_h=128, fine_size_w=800):
    img = imread(image_path)
    img = scipy.misc.imresize(img, [fine_size_h, fine_size_w])
    img = img/127.5 - 1
    return img

#def load_train_data(image_path, load_size=286, fine_size_h=1068, fine_size_w=800, is_testing=False):
def load_train_data(image_path, load_size=286, fine_size_h=128, fine_size_w=800, is_testing=False):
    img_A = imread(image_path[0])
    img_B = imread(image_path[1])
    if not is_testing:
        # img_A = scipy.misc.imresize(img_A, [load_size, load_size])
        # img_B = scipy.misc.imresize(img_B, [load_size, load_size])
        img_A = scipy.misc.imresize(img_A, [fine_size_h+30, fine_size_w+30])
        img_B = scipy.misc.imresize(img_B, [fine_size_h+30, fine_size_w+30])
        h1 = int(np.ceil(np.random.uniform(1e-2, 30)))
        w1 = int(np.ceil(np.random.uniform(1e-2, 30)))
        img_A = img_A[h1:h1+fine_size_h, w1:w1+fine_size_w]
        img_B = img_B[h1:h1+fine_size_h, w1:w1+fine_size_w]

        if np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
    else:
        img_A = scipy.misc.imresize(img_A, [fine_size_h, fine_size_w])
        img_B = scipy.misc.imresize(img_B, [fine_size_h, fine_size_w])

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return _imread(path, flatten=True).astype(np.float)
    else:
        return _imread(path, mode='RGB').astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

import cv2


image_path = "datasets/wei_focus_w=330,h=440/testB/00071-left=0280-top=0340.jpg"
img = imread(image_path)
img_cv2 = cv2.imread(image_path).astype(np.float)

print(img)

print(img_cv2)
print(img == img_cv2)
print(img.dtype)
print(img_cv2.dtype)


for go_row in range(img.shape[0]):
    for go_col in range(img.shape[1]):
        for go_ch in range(img.shape[2]):
            if(img[go_row,go_col,go_ch] != img_cv2[go_row,go_col,go_ch]):
                print("not same")
print("finish imread")

img = scipy.misc.imresize(img, [100, 100])
img_cv2 = cv2.resize(img_cv2, (100, 100), interpolation=cv2.INTER_LANCZOS4)
# for go_row in range(img.shape[0]):
#     for go_col in range(img.shape[1]):
#         for go_ch in range(img.shape[2]):
#             if(img[go_row,go_col,go_ch] != img_cv2[go_row,go_col,go_ch]):
#                 print(go_row,go_col,go_ch,"not same")
cv2.imshow("scipy",img)
cv2.waitKey(0)
cv2.imshow("cv2",img_cv2)
cv2.waitKey(0)
print("finish resize")