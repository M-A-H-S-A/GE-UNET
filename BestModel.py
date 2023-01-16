import numpy as np
import configparser
import random
import collections
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras import layers
from keras.layers import Input, concatenate, UpSampling2D, Dropout, AveragePooling2D, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, \
    Conv2D, Add, Activation, Lambda, Conv1D, Layer, MaxPooling2D, AveragePooling2D, BatchNormalization, add, Conv2DTranspose, Convolution2D
from keras import optimizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as K
from matplotlib import pyplot as plt
from keras.utils.vis_utils import plot_model as plot
from keras_drop_block import DropBlock2D
import time
import datetime
start_time = time.time()
print ("Start: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
import os
import sys
import cv2
from skimage.io import imread

config = configparser.RawConfigParser()
config.readfp(open(r'./configuration.txt'))
name_experiment = config.get('experiment name', 'name')
#===========================================
#name of the experiment

data_location = ''

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
for i in train_files:
    im = imread(training_images_loc + i)
    label = imread(training_label_loc + i.split('_')[0] + '_manual1.png',pilmode="L")
    old_size = im.shape[:2]  # old_size is in (height, width) format
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    color2 = [0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    new_label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=color2)

    train_data.append(cv2.resize(new_im, (desired_size, desired_size)))

    temp = cv2.resize(new_label, (desired_size, desired_size))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    train_label.append(temp)

for i in validate_files:
    im = imread(validate_images_loc + i)
    label = imread(validate_label_loc + i.split('_')[0] + '_manual1.png',pilmode="L")
    old_size = im.shape[:2]  # old_size is in (height, width) format
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    color2 = [0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    new_label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=color2)

    validate_data.append(cv2.resize(new_im, (desired_size, desired_size)))

    temp = cv2.resize(new_label, (desired_size, desired_size))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    validate_label.append(temp)

train_data = np.array(train_data)
train_label = np.array(train_label)

validate_data = np.array(validate_data)
validate_label = np.array(validate_label)

x_train = train_data.astype('float32') / 255.
y_train = train_label.astype('float32') / 255.
x_train = np.reshape(x_train, (
len(x_train), desired_size, desired_size, 3))  # adapt this if using `channels_first` image data format
y_train = np.reshape(y_train, (len(y_train), desired_size, desired_size, 1))  # adapt this if using `channels_first` im

x_validate = validate_data.astype('float32') / 255.
y_validate = validate_label.astype('float32') / 255.
x_validate = np.reshape(x_validate, (
len(x_validate), desired_size, desired_size, 3))  # adapt this if using `channels_first` image data format
y_validate = np.reshape(y_validate,
                        (len(y_validate), desired_size, desired_size, 1))  # adapt this if using `channels_first` im

# ====================DropBlock2D========================

class DropBlock2D(keras.layers.Layer):
    """See: https://arxiv.org/pdf/1810.12890.pdf"""

    def __init__(self,
                 block_size,
                 keep_prob,
                 sync_channels=False,
                 data_format=None,
                 **kwargs):
        """Initialize the layer.
        :param block_size: Size for each mask block.
        :param keep_prob: Probability of keeping the original feature.
        :param sync_channels: Whether to use the same dropout for all channels.
        :param data_format: 'channels_first' or 'channels_last' (default).
        :param kwargs: Arguments for parent class.
        """
        super(DropBlock2D, self).__init__(**kwargs)
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.sync_channels = sync_channels
        self.data_format = data_format
        self.supports_masking = True
        self.height = self.width = self.ones = self.zeros = None

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            self.height, self.width = input_shape[2], input_shape[3]
        else:
            self.height, self.width = input_shape[1], input_shape[2]
        self.ones = K.ones((self.height, self.width), name='ones')
        self.zeros = K.zeros((self.height, self.width), name='zeros')
        super().build(input_shape)

    def get_config(self):
        config = {'block_size': self.block_size,
                  'keep_prob': self.keep_prob,
                  'sync_channels': self.sync_channels,
                  'data_format': self.data_format}
        base_config = super(DropBlock2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def _get_gamma(self):
        """Get the number of activation units to drop"""
        height, width = K.cast(self.height, K.floatx()), K.cast(self.width, K.floatx())
        block_size = K.constant(self.block_size, dtype=K.floatx())
        return ((1.0 - self.keep_prob) / (block_size ** 2)) *\
               (height * width / ((height - block_size + 1.0) * (width - block_size + 1.0)))

    def _compute_valid_seed_region(self):
        positions = K.concatenate([
            K.expand_dims(K.tile(K.expand_dims(K.arange(self.height), axis=1), [1, self.width]), axis=-1),
            K.expand_dims(K.tile(K.expand_dims(K.arange(self.width), axis=0), [self.height, 1]), axis=-1),
        ], axis=-1)
        half_block_size = self.block_size // 2
        valid_seed_region = K.switch(
            K.all(
                K.stack(
                    [
                        positions[:, :, 0] >= half_block_size,
                        positions[:, :, 1] >= half_block_size,
                        positions[:, :, 0] < self.height - half_block_size,
                        positions[:, :, 1] < self.width - half_block_size,
                    ],
                    axis=-1,
                ),
                axis=-1,
            ),
            self.ones,
            self.zeros,
        )
        return K.expand_dims(K.expand_dims(valid_seed_region, axis=0), axis=-1)

    def _compute_drop_mask(self, shape):
        mask = K.random_binomial(shape, p=self._get_gamma())
        mask *= self._compute_valid_seed_region()
        mask = keras.layers.MaxPool2D(
            pool_size=(self.block_size, self.block_size),
            padding='same',
            strides=1,
            data_format='channels_last',
        )(mask)
        return 1.0 - mask

    def call(self, inputs, training=None):

        def dropped_inputs():
            outputs = inputs
            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 2, 3, 1])
            shape = K.shape(outputs)
            if self.sync_channels:
                mask = self._compute_drop_mask([shape[0], shape[1], shape[2], 1])
            else:
                mask = self._compute_drop_mask(shape)
            outputs = outputs * mask *\
                (K.cast(K.prod(shape), dtype=K.floatx()) / K.sum(mask))
            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 3, 1, 2])
            return outputs

        return K.in_train_phase(dropped_inputs, inputs, training=training)


# ====================SAUNet========================

def spatial_attention(input_feature):
    kernel_size=7

    if K.image_data_format() == "channels_first":
        channel=input_feature.shape[1]
        cbam_feature=Permute((2, 3, 1))(input_feature)
    else:
        channel=input_feature.shape[-1]
        cbam_feature=input_feature

    avg_pool=Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool=Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat=Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2

    cbam_feature=Conv2D(1, (7, 7),
                        strides=1,
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer='he_normal',
                        use_bias=False)(concat)
    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature=Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

def UNET(input_size=(desired_size, desired_size, 3)):
    inputs=Input(input_size)
    x=inputs
    x=Conv2D(64, (5, 5), activation='LeakyReLU', padding="same", kernel_initializer='he_normal')(x)
    x=DropBlock2D(block_size=9, keep_prob=0.9)(x)
    x=BatchNormalization()(x)
    x=AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    x=Conv2D(16, (5, 5), activation='selu', padding="same", kernel_initializer='he_normal')(x)
    x=DropBlock2D(block_size=9, keep_prob=0.9)(x)
    x=BatchNormalization()(x)
    x=Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="valid")(x)
    x=Dropout(0.4)(x)
    x=BatchNormalization()(x)
    outputs=Conv2D(1, (1, 1), padding="same", activation='sigmoid', data_format=None)

    model=Model(input=inputs, output=outputs)
    model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy', F2_Score()])

    return model


model = UNET(input_size=(desired_size, desired_size, 3))
print( "Check: final output of the network:")
print(model.output_shape)
plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
json_string = model.to_json()
open('./'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)
checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment +'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased
history=model.fit(x_train, y_train, epochs=500, batch_size=8, verbose=2, validation_data=(x_validate, y_validate), shuffle=True)
model.save_weights('./'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)

def plot_acc(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'./'+name_experiment+'/'+name_experiment +'_accuracy.png')
    plt.clf()

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')

training_time = time.time()
print("--- %s seconds for training---" % (time.time() - training_time))
print(plot_acc(history))
print(plot_loss(history))