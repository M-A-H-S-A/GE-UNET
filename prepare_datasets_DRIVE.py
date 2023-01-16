import os
import cv2
import numpy as np
#from  scipy.misc.pilutil import *

from skimage.io import imread

data_location = 'grape/datasets/'

training_images_loc = data_location + 'DRIVE/train/images/'
training_label_loc = data_location + 'DRIVE/train/labels/'

validate_images_loc = data_location + 'DRIVE/validate/images/'
validate_label_loc = data_location + 'DRIVE/validate/labels/'
train_files = os.listdir(training_images_loc)
train_data = []
train_label = []
validate_files = os.listdir(validate_images_loc)
validate_data = []
validate_label = []
desired_size = 592



def get_data_training(train_data,
                      train_label,
                      validate_data,
                      validate_label):
    for i in train_files:
        im=imread(training_images_loc + i)
        label=imread(training_label_loc + i.split('_')[0] + '_manual1.png', pilmode="L")
        old_size=im.shape[:2]  # old_size is in (height, width) format
        delta_w=desired_size - old_size[1]
        delta_h=desired_size - old_size[0]

        top, bottom=delta_h // 2, delta_h - (delta_h // 2)
        left, right=delta_w // 2, delta_w - (delta_w // 2)

        color=[0, 0, 0]
        color2=[0]
        new_im=cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                  value=color)

        new_label=cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                     value=color2)

        train_data.append(cv2.resize(new_im, (desired_size, desired_size)))

        temp=cv2.resize(new_label, (desired_size, desired_size))
        _, temp=cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
        train_label.append(temp)

    for i in validate_files:
        im=imread(validate_images_loc + i)
        label=imread(validate_label_loc + i.split('_')[0] + '_manual1.png', pilmode="L")
        old_size=im.shape[:2]  # old_size is in (height, width) format
        delta_w=desired_size - old_size[1]
        delta_h=desired_size - old_size[0]

        top, bottom=delta_h // 2, delta_h - (delta_h // 2)
        left, right=delta_w // 2, delta_w - (delta_w // 2)

        color=[0, 0, 0]
        color2=[0]
        new_im=cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                  value=color)

        new_label=cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                     value=color2)

        validate_data.append(cv2.resize(new_im, (desired_size, desired_size)))

        temp=cv2.resize(new_label, (desired_size, desired_size))
        _, temp=cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
        validate_label.append(temp)

    train_data=np.array(train_data)
    train_label=np.array(train_label)

    validate_data=np.array(validate_data)
    validate_label=np.array(validate_label)

    x_train=train_data.astype('float32') / 255.
    y_train=train_label.astype('float32') / 255.
    x_train=np.reshape(x_train, (
        len(x_train), desired_size, desired_size, 3))  # adapt this if using `channels_first` image data format
    y_train=np.reshape(y_train,
                       (len(y_train), desired_size, desired_size, 1))  # adapt this if using `channels_first` im

    x_validate=validate_data.astype('float32') / 255.
    y_validate=validate_label.astype('float32') / 255.
    x_validate=np.reshape(x_validate, (
        len(x_validate), desired_size, desired_size, 3))  # adapt this if using `channels_first` image data format
    y_validate=np.reshape(y_validate,
                          (len(y_validate), desired_size, desired_size, 1))  # adapt this if using `channels_first` im

    return x_train,y_train,x_validate,y_validate
